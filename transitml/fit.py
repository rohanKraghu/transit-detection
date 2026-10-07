"""Planet parameter fits: a limb-darkened transit model sampled with MCMC.

The search and the classifier say *whether* a star has a transit-like
signal.  This module says what the signal implies physically: the
planet-to-star radius ratio, the impact parameter, the stellar density the
transit shape requires, the duration and the ephemeris, each with a
credible interval.

The model is ``batman`` (Kreidberg 2015): a circular orbit and quadratic limb
darkening, integrated over the exposure time.  It has seven parameters::

    t0, period, k = Rp/R*, b, log10(T14 / day), q1, q2

``q1, q2`` are Kipping's (2013) limb-darkening parameters, uniform on the
unit square, which covers every physical quadratic law exactly once.  The
geometry is sampled as the total duration T14 and b, which with k and the
period fix a/R* and so the stellar density.  In a shallow transit the
ingress is barely resolved, and b and the density slide along a curved
ridge that a sampler crosses slowly; T14 is measured well, so in these
coordinates the ridge is nearly straight.  The prior is still flat in
log10 of the stellar density (the change of variables carries its
Jacobian), a quantity a reader can check against the star: a transit whose
shape needs a density far from the host's is a classic sign of an
eclipsing binary or a blend (Seager & Mallen-Ornelas 2003).

The fit uses the *detrended* light curve, only within ``window_durations``
of each transit.  Detrending is never perfect at the level of a transit
depth, so each transit window gets its own polynomial baseline (a
quadratic by default), marginalised analytically with a flat prior on its
coefficients: the depth and its interval then carry the uncertainty of
where the baseline sits under each transit, instead of trusting the
detrended level.  The noise is
set from the data rather than trusted from ``flux_err``: the scatter of the
cadences outside every fitted window (or, when there are too few, the
residuals of a first maximum-a-posteriori fit) fixes the white-noise level,
and the time-averaging method (Pont et al. 2006; Winn et al. 2008) inflates
it by beta, the factor by which binned residuals scatter more than white
noise would.  Without that step the intervals are too
narrow on any star with red noise.  The sampler is ``emcee`` (Foreman-Mackey
et al. 2013) with differential-evolution moves, which cope with the strong
correlation between k, b and the density in shallow transits.

``batman`` and ``emcee`` are needed only here, and imported only when a fit
runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .physics import G_CGS, RHO_SUN_CGS, SECONDS_PER_DAY, density_from_gravity
from .preprocess import FlattenedLightCurve

#: The sampled parameters, in order.
PARAMETERS: tuple[str, ...] = ("t0", "period", "k", "b", "log10_t14", "q1", "q2")

#: Earth radii per solar radius.
R_EARTH_PER_R_SUN: float = 109.076


class FitError(ValueError):
    """The light curve cannot support a fit of this signal."""


@dataclass(frozen=True)
class FitConfig:
    """Settings for :func:`fit_transit`.  The defaults suit one TESS sector."""

    n_walkers: int = 32
    #: The chain runs at least ``min_steps``, then in blocks of ``block_steps``
    #: until it is ``convergence_tau`` autocorrelation times long after burn-in,
    #: or reaches ``max_steps``.
    min_steps: int = 4000
    max_steps: int = 30000
    block_steps: int = 2000
    #: Burn-in is this many autocorrelation times (at least ``min_burn`` steps).
    burn_tau: float = 3.0
    min_burn: int = 1000
    #: Keep cadences within this many search durations of a transit centre.
    window_durations: float = 2.5
    #: Baseline marginalised in each transit window: ``"offset"``, ``"line"``
    #: or ``"quadratic"``.  The quadratic follows the curvature detrending
    #: leaves under a transit; see the coverage study in the README.
    baseline: str = "quadratic"
    #: Exposure time integrated over; ``None`` means the median cadence and
    #: ``0`` means instantaneous samples.
    exposure_minutes: float | None = None
    #: Spacing of the sub-exposure samples when integrating.
    supersample_minutes: float = 3.0
    #: Flat prior on the period: within this fraction of the search value.
    period_window: float = 0.02
    #: Flat prior on log10 of the stellar density in g/cm^3.
    log10_rho_range: tuple[float, float] = (-2.0, 2.0)
    #: Largest radius ratio allowed; 0.5 admits grazing binaries.
    max_k: float = 0.5
    #: A fit needs at least this many cadences inside the transits.
    min_in_transit: int = 5
    #: The chain counts as converged when it is this many autocorrelation times long.
    convergence_tau: float = 50.0
    #: Posterior draws kept with the result, for plots.
    n_keep: int = 4000
    seed: int = 0


# --------------------------------------------------------------------------
# Geometry, vectorised over posterior samples
# --------------------------------------------------------------------------
def q_to_u(q1, q2):
    """Kipping (2013): unit-square ``(q1, q2)`` to quadratic ``(u1, u2)``."""
    root = np.sqrt(q1)
    return 2.0 * root * q2, root * (1.0 - 2.0 * q2)


def u_to_q(u1, u2):
    """Inverse of :func:`q_to_u`."""
    total = u1 + u2
    return total**2, u1 / (2.0 * total)


def a_over_rs(period_days, rho_cgs):
    """a/R* from Kepler's third law for a star of mean density ``rho``."""
    p_sec = np.asarray(period_days, dtype=float) * SECONDS_PER_DAY
    return (G_CGS * np.asarray(rho_cgs, dtype=float) * p_sec**2 / (3.0 * np.pi)) ** (1.0 / 3.0)


def a_from_t14(period, k, b, t14):
    """a/R* that gives a total duration ``t14`` for this period, k and b (circular orbit).

    From ``sin(pi T14 / P) = sqrt((1 + k)^2 - b^2) / (a sin i)`` and ``a cos i = b``.
    """
    period, k, b, t14 = (np.asarray(v, dtype=float) for v in (period, k, b, t14))
    s = np.sin(np.pi * t14 / period)
    return np.sqrt(b**2 + ((1.0 + k) ** 2 - b**2) / s**2)


def density(period_days, a):
    """Stellar density in g/cm^3 implied by a/R* and the period (inverse of :func:`a_over_rs`)."""
    p_sec = np.asarray(period_days, dtype=float) * SECONDS_PER_DAY
    return 3.0 * np.pi * np.asarray(a, dtype=float) ** 3 / (G_CGS * p_sec**2)


def log_density_jacobian(period, k, b, t14, a):
    """``|d ln rho / d ln T14|`` at fixed period, k and b.

    Sampling log T14 with a flat prior on log rho needs this factor in the
    prior density.
    """
    x = np.pi * t14 / period
    chord2 = (1.0 + k) ** 2 - b**2
    return 3.0 * x * chord2 * np.cos(x) / (np.sin(x) ** 3 * a**2)


def durations(period, a, k, b):
    """Total and full (flat-bottom) durations in days, exact for a circular orbit.

    ``T = P / pi * arcsin(sqrt((1 +- k)^2 - b^2) / (a sin i))`` with ``cos i = b / a``;
    zero where the chord does not exist (no transit, or no flat bottom).
    """
    period, a, k, b = (np.asarray(v, dtype=float) for v in (period, a, k, b))
    sin_i = np.sqrt(np.clip(1.0 - (b / a) ** 2, 1e-12, None))

    def chord(radius):
        squared = radius**2 - b**2
        arg = np.sqrt(np.clip(squared, 0.0, None)) / (a * sin_i)
        return np.where(squared > 0, period / np.pi * np.arcsin(np.clip(arg, 0.0, 1.0)), 0.0)

    return chord(1.0 + k), chord(1.0 - k)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
def _batman_params(t0, period, k, a, inc_deg, u1, u2):
    import batman

    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = t0, period, k, a, inc_deg
    params.ecc, params.w = 0.0, 90.0
    params.limb_dark, params.u = "quadratic", [u1, u2]
    return params


class TransitModel:
    """``batman`` on a fixed set of times, evaluated for any parameter vector."""

    def __init__(self, time: NDArray[np.float64], exposure_days: float, supersample_days: float):
        import batman

        self.time = np.asarray(time, dtype=float)
        self.exposure_days = float(exposure_days)
        self.supersample = (
            max(1, int(math.ceil(exposure_days / supersample_days))) if exposure_days > 0 else 1
        )
        self._params = _batman_params(float(np.median(self.time)), 3.0, 0.1, 10.0, 89.0, 0.4, 0.2)
        if self.supersample > 1:
            self._model = batman.TransitModel(
                self._params, self.time, supersample_factor=self.supersample, exp_time=exposure_days
            )
        else:
            self._model = batman.TransitModel(self._params, self.time)

    def flux(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Relative flux for ``(t0, period, k, b, log10_t14, q1, q2[, f0])``; ``f0`` defaults to 1."""
        t0, period, k, b, log_t14, q1, q2 = theta[:7]
        f0 = theta[7] if len(theta) > 7 else 1.0
        a = float(a_from_t14(period, k, b, 10.0**log_t14))
        u1, u2 = q_to_u(q1, q2)
        p = self._params
        p.t0, p.per, p.rp, p.a = t0, period, k, a
        p.inc = math.degrees(math.acos(min(1.0, b / a)))
        p.u = [float(u1), float(u2)]
        return f0 * self._model.light_curve(p)


# --------------------------------------------------------------------------
# Posterior
# --------------------------------------------------------------------------
#: Polynomial order of each baseline kind.
BASELINE_ORDERS = {"offset": 0, "line": 1, "quadratic": 2}


class _Baselines:
    """Per-window polynomial baselines, profiled out of the residuals.

    For residuals ``r`` and weights ``w = 1 / sigma^2`` the best polynomial
    in each window is a weighted least-squares fit; what is left is
    ``chi^2 = sum w r^2 - b^T M^-1 b`` window by window.  With a flat prior on
    the coefficients this is the marginal likelihood up to a constant (the
    design does not depend on the transit parameters).  A window with too
    few cadences for the requested order gets a lower one.
    """

    def __init__(self, window: NDArray[np.int64], x: NDArray[np.float64],
                 sigma: NDArray[np.float64], kind: str):
        if kind not in BASELINE_ORDERS:
            raise ValueError(f"baseline must be one of {sorted(BASELINE_ORDERS)}, got {kind!r}")
        self.window = np.asarray(window)
        self.n = int(self.window.max()) + 1 if self.window.size else 0
        w = 1.0 / np.asarray(sigma, dtype=float) ** 2
        x = np.asarray(x, dtype=float)
        order = BASELINE_ORDERS[kind]
        counts = np.bincount(self.window, minlength=self.n)
        # Columns x^j, zeroed in windows too short to support degree j.
        allowed = np.minimum(order, np.maximum(counts - 2, 0) // 2)
        self.basis = np.stack(
            [np.where(allowed[self.window] >= j, x**j, 0.0) for j in range(order + 1)], axis=1
        )
        self.weighted = self.basis * w[:, None]
        moments = np.zeros((self.n, order + 1, order + 1))
        for j in range(order + 1):
            for k in range(order + 1):
                moments[:, j, k] = np.bincount(
                    self.window, self.weighted[:, j] * self.basis[:, k], self.n
                )
        self.inverse = np.linalg.pinv(moments)
        self.w = w

    def remove(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Residuals with each window's best polynomial taken out."""
        b = np.stack(
            [np.bincount(self.window, self.weighted[:, j] * r, self.n)
             for j in range(self.basis.shape[1])],
            axis=1,
        )
        coefficients = np.einsum("wjk,wk->wj", self.inverse, b)
        return r - np.sum(coefficients[self.window] * self.basis, axis=1)

    def chi2(self, r: NDArray[np.float64]) -> float:
        left = self.remove(r)
        return float(np.sum(self.w * left**2))


@dataclass
class _Problem:
    model: TransitModel
    flux: NDArray[np.float64]
    sigma: NDArray[np.float64]
    t0_centre: float
    t0_halfwidth: float
    period_centre: float
    config: FitConfig
    baselines: _Baselines | None = None

    def residuals(self, theta) -> NDArray[np.float64]:
        """Data minus model, with each window's best baseline removed."""
        r = self.flux - self.model.flux(theta)
        return self.baselines.remove(r) if self.baselines is not None else r

    def log_prior(self, theta) -> float:
        t0, period, k, b, log_t14, q1, q2 = theta
        c = self.config
        if abs(t0 - self.t0_centre) > self.t0_halfwidth:
            return -np.inf
        if abs(period / self.period_centre - 1.0) > c.period_window:
            return -np.inf
        if not (0.0 < k < c.max_k and 0.0 <= b < 1.0 + k):
            return -np.inf
        if not (0.0 < q1 < 1.0 and 0.0 < q2 < 1.0):
            return -np.inf
        t14 = 10.0**log_t14
        if not 0.0 < t14 < 0.5 * period:
            return -np.inf
        a = float(a_from_t14(period, k, b, t14))
        if a <= 1.0 + k:
            return -np.inf  # the planet's orbit would lie inside the star
        log_rho = math.log10(float(density(period, a)))
        if not (c.log10_rho_range[0] < log_rho < c.log10_rho_range[1]):
            return -np.inf
        # Flat in log10(rho), expressed in the sampled log10(T14).
        return math.log(float(log_density_jacobian(period, k, b, t14, a)))

    def log_likelihood(self, theta) -> float:
        r = self.flux - self.model.flux(theta)
        if self.baselines is not None:
            return -0.5 * self.baselines.chi2(r)
        z = r / self.sigma
        return -0.5 * float(z @ z)

    def __call__(self, theta) -> float:
        prior = self.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        value = prior + self.log_likelihood(theta)
        return value if np.isfinite(value) else -np.inf


def _robust_rms(z: NDArray[np.float64], clip: float = 5.0) -> float:
    """RMS after dropping points beyond ``clip`` robust sigmas; 1.0 if undefined.

    The plain RMS is used for the level because it is far more efficient than
    the median absolute deviation; the MAD only decides what to drop.
    """
    z = np.asarray(z, dtype=float)
    mad = 1.4826 * float(np.median(np.abs(z - np.median(z))))
    keep = np.abs(z - np.median(z)) <= clip * mad if mad > 0 else np.ones(z.size, bool)
    rms = float(np.sqrt(np.mean(z[keep] ** 2))) if keep.any() else float("nan")
    return rms if np.isfinite(rms) and rms > 0 else 1.0


def time_averaging_beta(
    time: NDArray[np.float64],
    residuals: NDArray[np.float64],
    bin_days: NDArray[np.float64],
    gap_days: float,
    min_bins: int = 8,
) -> float:
    """Red-noise factor beta from binned residuals (Winn et al. 2008).

    For each bin width the residuals are averaged in bins of that width,
    within each stretch of data (a gap longer than ``gap_days`` starts a new
    one), and the scatter of the bin means is compared with what white noise
    of the unbinned scatter would give.  Beta is the median ratio over the
    bin widths, and never below 1.  Returns 1.0 when no width has
    ``min_bins`` full bins.
    """
    time = np.asarray(time, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    sigma_1 = float(np.std(residuals))
    if sigma_1 == 0.0 or time.size < 2 * min_bins:
        return 1.0
    segment = np.concatenate([[0], np.cumsum(np.diff(time) > gap_days)])
    cadence = float(np.median(np.diff(time)))
    ratios = []
    for width in np.asarray(bin_days, dtype=float):
        expected_count = width / cadence
        if expected_count < 2:
            continue
        means, counts = [], []
        for seg in np.unique(segment):
            in_seg = segment == seg
            t, r = time[in_seg], residuals[in_seg]
            index = np.floor((t - t[0]) / width).astype(int)
            for i in np.unique(index):
                chunk = r[index == i]
                if chunk.size >= 0.75 * expected_count:
                    means.append(chunk.mean())
                    counts.append(chunk.size)
        n_bins = len(means)
        if n_bins < min_bins:
            continue
        observed = float(np.std(means))
        expected = sigma_1 / math.sqrt(float(np.mean(counts))) * math.sqrt(n_bins / (n_bins - 1.0))
        ratios.append(observed / expected)
    return max(1.0, float(np.median(ratios))) if ratios else 1.0


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
def _summary(values: NDArray[np.float64]) -> dict[str, float]:
    p = np.percentile(values, [2.5, 16.0, 50.0, 84.0, 97.5])
    return {
        "median": float(p[2]),
        "lower": float(p[1]),
        "upper": float(p[3]),
        "lower95": float(p[0]),
        "upper95": float(p[4]),
    }


@dataclass
class FitResult:
    """A posterior summary, plus the draws and data a plot needs."""

    target_id: str
    #: Every sampled and derived quantity: median, 68% and 95% intervals.
    parameters: dict[str, dict[str, float]]
    #: The maximum-a-posteriori parameter vector.
    map_parameters: dict[str, float]
    noise: dict[str, float]
    sampler: dict[str, Any]
    density_check: dict[str, Any] | None
    warnings: list[str]
    exposure_minutes: float
    baseline: str = "quadratic"
    #: Posterior draws, ``(n_keep, len(PARAMETERS))``; not written to JSON.
    samples: NDArray[np.float64] = field(
        repr=False, default_factory=lambda: np.empty((0, len(PARAMETERS)))
    )
    #: The fitted cadences, their adopted errors and each window's best
    #: baseline under the MAP model; not written to JSON.
    data: dict[str, NDArray[np.float64]] = field(repr=False, default_factory=dict)

    @property
    def converged(self) -> bool:
        return bool(self.sampler.get("converged"))

    def value(self, name: str) -> float:
        return self.parameters[name]["median"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "model": "batman, circular orbit, quadratic limb darkening, "
            f"integrated over {self.exposure_minutes:.1f}-minute exposures, "
            f"with a {self.baseline} baseline under each transit, marginalised",
            "parameters": self.parameters,
            "map": self.map_parameters,
            "noise": self.noise,
            "sampler": self.sampler,
            "density_check": self.density_check,
            "warnings": self.warnings,
            "interval_note": "median with the central 68% (lower, upper) and 95% "
            "(lower95, upper95) posterior intervals",
        }


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------
def _initial_guess(period, epoch, duration, depth) -> NDArray[np.float64]:
    k = math.sqrt(max(depth, 1e-6) / 1.1)
    k = min(max(k, 0.005), 0.4)
    b = 0.4
    t14 = min(max(duration, 1e-3), 0.2 * period)
    q1, q2 = u_to_q(0.40, 0.25)
    return np.array([epoch, period, k, b, math.log10(t14), q1, q2])


def _clip_to_prior(theta, problem: _Problem) -> NDArray[np.float64]:
    c = problem.config
    theta = np.array(theta, dtype=float)
    theta[0] = np.clip(theta[0], problem.t0_centre - 0.9 * problem.t0_halfwidth,
                       problem.t0_centre + 0.9 * problem.t0_halfwidth)
    theta[2] = np.clip(theta[2], 1e-3, 0.95 * c.max_k)
    theta[3] = np.clip(theta[3], 0.0, 0.95)
    theta[4] = min(theta[4], math.log10(0.4 * theta[1]))
    theta[5:7] = np.clip(theta[5:7], 0.02, 0.98)
    return theta


def _maximise(problem: _Problem, start: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Nelder-Mead from a few impact parameters; returns the best point."""
    from scipy.optimize import minimize

    best, best_value = start, problem(start)
    for b in (0.1, 0.5, 0.85):
        trial = _clip_to_prior(np.r_[start[:3], b, start[4:]], problem)
        if not np.isfinite(problem(trial)):
            continue
        result = minimize(
            lambda th: -problem(th) if np.isfinite(problem(th)) else 1e300,
            trial,
            method="Nelder-Mead",
            options={"maxiter": 6000, "maxfev": 12000, "xatol": 1e-7, "fatol": 1e-4, "adaptive": True},
        )
        value = problem(result.x)
        if np.isfinite(value) and value > best_value:
            best, best_value = np.asarray(result.x, dtype=float), value
    return best, best_value


def _derived(samples: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    t0, period, k, b, log_t14, q1, q2 = samples.T
    a = a_from_t14(period, k, b, 10.0**log_t14)
    rho = density(period, a)
    log_rho = np.log10(rho)
    t14, t23 = durations(period, a, k, b)
    u1, u2 = q_to_u(q1, q2)
    return {
        "t0": t0,
        "period": period,
        "k": k,
        "b": b,
        "rho_star": rho,
        "log10_rho_star": log_rho,
        "a_over_rs": a,
        "inclination_deg": np.degrees(np.arccos(np.clip(b / a, 0.0, 1.0))),
        "t14_hours": 24.0 * t14,
        "t23_hours": 24.0 * t23,
        "u1": u1,
        "u2": u2,
        "q1": q1,
        "q2": q2,
    }


def mid_transit_depth(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fractional depth at mid-transit for each draw (instantaneous, not exposure-averaged)."""
    model = TransitModel(np.array([0.0]), 0.0, 1.0)
    out = np.empty(len(samples))
    for i, theta in enumerate(samples):
        shifted = np.r_[0.0, theta[1:7], 1.0]
        out[i] = 1.0 - float(model.flux(shifted)[0])
    return out


def fit_transit(
    flat: FlattenedLightCurve,
    period: float,
    epoch: float,
    duration: float,
    depth: float,
    config: FitConfig | None = None,
    *,
    stellar_density: tuple[float, float] | None = None,
    stellar_radius: float | None = None,
) -> FitResult:
    """Fit one periodic transit in a detrended light curve.

    ``period, epoch, duration, depth`` come from the search and seed the fit.
    ``stellar_density`` (mean and standard deviation, g/cm^3) is not a prior:
    the fit is run without it and then compared with it, so the comparison
    is a test.  ``stellar_radius`` (solar radii) converts the radius ratio
    to Earth radii.

    Raises :class:`FitError` when the light curve has too few cadences in
    transit to fit.
    """
    import emcee

    config = config or FitConfig()
    time = np.asarray(flat.time, dtype=float)
    flux = np.asarray(flat.flux, dtype=float)
    err = np.asarray(flat.flux_err, dtype=float)
    if not (np.isfinite(period) and period > 0 and np.isfinite(duration) and duration > 0):
        raise FitError(f"{flat.target_id}: no usable signal to fit")

    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    window = np.abs(phase) < config.window_durations * duration
    in_transit = np.abs(phase) < 0.5 * duration
    if int(np.count_nonzero(in_transit)) < config.min_in_transit:
        raise FitError(
            f"{flat.target_id}: {int(np.count_nonzero(in_transit))} cadences in transit; "
            f"a fit needs {config.min_in_transit}"
        )
    t, f = time[window], flux[window]
    e = err[window]
    if not np.all(np.isfinite(e) & (e > 0)):
        e = np.full(t.size, float(flat.scatter))
    n_transits = int(np.unique(np.round((time[in_transit] - epoch) / period)).size)

    # Reference epoch: the transit nearest the middle of the fitted data, which
    # decorrelates t0 from the period.
    t0_centre = epoch + round((float(np.median(t)) - epoch) / period) * period
    cadence = float(np.median(np.diff(time)))
    exposure_min = cadence * 1440.0 if config.exposure_minutes is None else config.exposure_minutes
    model = TransitModel(t, exposure_min / 1440.0, config.supersample_minutes / 1440.0)
    # Each transit's window gets its own baseline, in units of the window's half-width.
    transit_number = np.round((t - t0_centre) / period).astype(int)
    window_id = np.unique(transit_number, return_inverse=True)[1]
    half_width = config.window_durations * duration
    x_window = (t - t0_centre - transit_number * period) / half_width

    problem = _Problem(
        model=model,
        flux=f,
        sigma=e,
        t0_centre=t0_centre,
        t0_halfwidth=max(duration, 2.0 * cadence),
        period_centre=period,
        config=config,
        baselines=_Baselines(window_id, x_window, e, config.baseline),
    )
    start = _clip_to_prior(_initial_guess(period, t0_centre, duration, depth), problem)
    theta_map, _ = _maximise(problem, start)

    # Noise from the data: rescale the errors to the scatter of the cadences
    # outside every fitted window (or, if too few, to the MAP residuals), then
    # inflate by the red-noise factor measured on the same cadences.  Neither
    # changes where the maximum is.
    outside = ~window
    err_all = err if np.all(np.isfinite(err) & (err > 0)) else np.full(time.size, float(flat.scatter))
    if np.count_nonzero(outside) >= max(3 * t.size, 100):
        noise_t = time[outside]
        noise_z = (flux[outside] - np.median(flux[outside])) / err_all[outside]
        noise_from = "cadences outside the fitted windows"
    else:
        noise_t = t
        noise_z = problem.residuals(theta_map) / e
        noise_from = "residuals of the best fit"
    scale = _robust_rms(noise_z)
    width = max(10.0 ** theta_map[4], duration)
    beta = time_averaging_beta(
        noise_t, noise_z / scale, np.linspace(0.25, 1.0, 6) * width, gap_days=3.0 * cadence
    )
    problem.sigma = e * scale * beta
    problem.baselines = _Baselines(window_id, x_window, problem.sigma, config.baseline)

    # Sample.
    rng = np.random.default_rng(config.seed)
    ndim = len(PARAMETERS)
    jitter = np.array([
        1e-3 * duration, 1e-5 * period, 1e-3, 1e-2, 2e-3, 1e-2, 1e-2,
    ])
    walkers = []
    while len(walkers) < config.n_walkers:
        trial = theta_map + jitter * rng.standard_normal(ndim)
        if np.isfinite(problem(trial)):
            walkers.append(trial)
    sampler = emcee.EnsembleSampler(
        config.n_walkers,
        ndim,
        problem,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
    )
    sampler.random_state = np.random.RandomState(config.seed).get_state()
    sampler.run_mcmc(np.array(walkers), config.min_steps, progress=False)
    while True:
        steps = sampler.iteration
        # The burn-in comes from a first estimate of tau; tau is then measured
        # on the chain that is kept, and the larger of the two is used, so the
        # verdict below is the one the loop stopped on.
        tau_max = float(np.nanmax(sampler.get_autocorr_time(discard=steps // 3, quiet=True, tol=0)))
        burn = min(max(config.min_burn, int(math.ceil(config.burn_tau * tau_max))), steps // 2)
        tau_max = max(
            tau_max, float(np.nanmax(sampler.get_autocorr_time(discard=burn, quiet=True, tol=0)))
        )
        kept = steps - burn
        converged = bool(np.isfinite(tau_max) and kept >= config.convergence_tau * tau_max)
        if converged or steps + config.block_steps > config.max_steps:
            break
        sampler.run_mcmc(None, config.block_steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)

    keep = rng.choice(len(chain), size=min(config.n_keep, len(chain)), replace=False)
    samples = chain[keep]
    derived = _derived(chain)
    derived["depth_ppm"] = 1e6 * mid_transit_depth(samples)
    if stellar_radius is not None:
        derived["rp_earth"] = derived["k"] * stellar_radius * R_EARTH_PER_R_SUN
    parameters = {name: _summary(values) for name, values in derived.items()}

    warnings = []
    if not converged:
        warnings.append(
            f"chain is {kept} steps, {kept / tau_max if tau_max > 0 else 0:.0f} autocorrelation "
            f"times; {config.convergence_tau:.0f} are needed for converged intervals"
        )
    edge = config.period_window * period
    if np.any(np.abs(chain[:, 1] - period) > 0.98 * edge):
        warnings.append("the period posterior reaches the edge of its prior")
    if parameters["t14_hours"]["upper"] / 24.0 > 2.0 * config.window_durations * duration * 0.8:
        warnings.append("the fitted duration approaches the width of the fitted window")
    if parameters["b"]["median"] > 1.0 - parameters["k"]["median"]:
        warnings.append("the fit prefers a grazing transit, where k and b are poorly constrained")

    density_check = None
    if stellar_density is not None:
        mean, sd = stellar_density
        fitted = derived["rho_star"]
        # How often a fitted draw falls below a draw of the star's density
        # (log-normal with the stated fractional spread), two-sided.
        star = mean * np.exp(sd / mean * rng.standard_normal(fitted.size))
        below = float(np.mean(fitted < star))
        tail = 2.0 * min(below, 1.0 - below)
        density_check = {
            "stellar_density": float(mean),
            "stellar_density_sd": float(sd),
            "ratio": _summary(fitted / mean),
            "two_sided_tail": tail,
            "consistent": bool(tail > 0.003),
            "note": "fitted density against the star's; a transit that needs a very "
            "different density suggests an eclipsing binary, a blend or an eccentric orbit",
        }

    return FitResult(
        target_id=flat.target_id,
        parameters=parameters,
        map_parameters={name: float(v) for name, v in zip(PARAMETERS, theta_map)},
        noise={
            "n_points": int(t.size),
            "n_in_transit": int(np.count_nonzero(in_transit)),
            "n_transits": n_transits,
            "sigma_ppm": float(1e6 * np.median(e * scale)),
            "beta": float(beta),
            "measured_from": noise_from,
        },
        sampler={
            "n_walkers": config.n_walkers,
            "n_steps": int(steps),
            "n_burn": int(burn),
            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "autocorr_time_max": tau_max,
            "n_effective": float(kept * config.n_walkers / tau_max) if tau_max > 0 else None,
            "converged": converged,
            "seed": config.seed,
        },
        density_check=density_check,
        warnings=warnings,
        exposure_minutes=float(exposure_min),
        baseline=config.baseline,
        samples=samples,
        data={
            "time": t,
            "flux": f,
            "sigma": problem.sigma.copy(),
            "baseline": (f - model.flux(theta_map)) - problem.residuals(theta_map),
        },
    )


#: Fractional uncertainty of a catalogue density worked out from log g (or
#: mass) and radius.  Against the published densities of 15 TESS hosts, the
#: TIC's log g and radius came within 30% (HD 1397, a subgiant, 28% low).
CATALOGUE_DENSITY_FRACTION = 0.3


def stellar_priors_from_meta(meta: dict[str, Any]) -> tuple[tuple[float, float] | None, float | None]:
    """Stellar density (mean, sd) and radius when the light curve carries them.

    A density the curve records as ``rho_star_cgs`` (synthetic and injected
    curves) gets a 10% uncertainty.  A survey curve carries the catalogue's
    log g (or mass) and radius instead, as a MAST download does from the TIC;
    their density gets :data:`CATALOGUE_DENSITY_FRACTION`.  A temperature
    alone gives no density here: a main-sequence guess would flag every
    evolved star.
    """

    def number(key: str) -> float | None:
        try:
            value = float(meta.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    rho, radius = number("rho_star_cgs"), number("r_star_rsun")
    density = (rho, 0.1 * rho) if rho is not None else None
    if density is None and radius is not None:
        mass, logg = number("m_star_msun"), number("logg_cgs")
        if mass is not None:
            rho = RHO_SUN_CGS * mass / radius**3
        elif logg is not None:
            rho = density_from_gravity(logg, radius)
        if rho is not None:
            density = (rho, CATALOGUE_DENSITY_FRACTION * rho)
    return density, radius


def default_exposure_minutes(meta: dict[str, Any]) -> float | None:
    """The exposure a fit should integrate over, for a curve with this metadata.

    Curves from this package's generator or injector (they carry ``kind``)
    hold their eclipse as instantaneous samples of a trapezoid, so 0.  For
    survey data ``None``: the median cadence, which for TESS is the exposure.
    An explicit ``exposure_minutes`` in the metadata wins.
    """
    if meta.get("exposure_minutes") is not None:
        return float(meta["exposure_minutes"])
    return 0.0 if "kind" in meta else None


def quick_config(**changes: Any) -> FitConfig:
    """A short-chain configuration for tests and previews (not converged intervals)."""
    return replace(
        FitConfig(n_walkers=24, min_steps=600, max_steps=600, min_burn=200, n_keep=500), **changes
    )
