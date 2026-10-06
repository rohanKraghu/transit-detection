"""Box Least Squares search and the vetting statistics derived from it.

Design note (why classical features and not a 1D CNN) is in the README; the
short version is that the feature set below is a direct encoding of the tests a
human vetter applies, and it is trainable on ~100 positives, which a CNN is not.

The features fall into four groups:

**Detection strength** -- how significant is the best periodic box?
    ``bls_sde``, ``bls_depth_snr``, ``bls_depth_over_scatter``, ``delta_loglike``

**Geometry / physical plausibility** -- could a planet produce this?
    ``duration_over_period``, ``log_duration_ratio`` (measured duration versus
    the duration Kepler's third law allows at that period), ``log_depth``

**False-positive discriminants** -- the eclipsing-binary tests
    ``odd_even_sigma``, ``secondary_sigma``, ``half_period_depth_ratio``,
    ``flat_bottom_fraction`` (V-shaped or not), ``harmonic_delta_loglike``.
    The two significances are divided by the red-noise ``beta`` (floored at 1),
    so correlated noise cannot masquerade as a binary signature.

**Noise characterisation** -- is the "detection" just correlated noise?
    ``red_noise_beta`` (Pont, Zucker & Queloz 2006), ``log_scatter``,
    ``power_contrast``, ``max_single_event_fraction``
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import numpy as np
from astropy.timeseries import BoxLeastSquares
from numpy.typing import NDArray
from scipy import stats

from .config import BLSConfig
from .physics import RHO_SUN_CGS, expected_central_duration
from .preprocess import FlattenedLightCurve, robust_sigma

#: Fixed column order.  The model, the permutation-importance plot and the
#: dataset all key off this, so it is defined once, here.
FEATURE_NAMES: tuple[str, ...] = (
    # detection strength
    "bls_sde",
    "bls_depth_snr",
    "bls_depth_over_scatter",
    "delta_loglike",
    "power_contrast",
    # geometry / plausibility
    "log_depth",
    "bls_duration",
    "duration_over_period",
    "log_duration_ratio",
    "log_period",
    "n_transits",
    "min_points_per_transit",
    # false-positive discriminants
    "odd_even_sigma",
    "secondary_sigma",
    "half_period_depth_ratio",
    "flat_bottom_fraction",
    "harmonic_delta_loglike",
    "harmonic_amp_over_depth",
    # noise characterisation
    "red_noise_beta",
    "log_scatter",
    "max_single_event_fraction",
    "flux_skew",
    "clipped_fraction",
)


def period_grid(baseline_days: float, config: BLSConfig) -> NDArray[np.float64]:
    """Log-spaced trial periods.

    Log spacing rather than linear because the width of a BLS peak scales with
    P^2 / (baseline * duration): a linear grid massively oversamples long
    periods and undersamples short ones.  The upper bound is set so that at
    least two transits fit in the baseline -- a single event cannot be confirmed
    as periodic, and including such periods just adds false positives.
    """
    p_max = max(config.min_period_days * 1.5, baseline_days * config.max_period_fraction_of_baseline)
    return np.logspace(
        np.log10(config.min_period_days), np.log10(p_max), config.n_periods
    )


#: Environment variable choosing who computes the BLS periodogram: ``astropy``
#: (the default), ``cpu`` or ``gpu`` (:mod:`transitml.fastbls`).  An execution
#: detail, not part of the configuration: every engine returns the same
#: periodogram, so features and trained models do not depend on it.
BLS_ENGINE_ENV = "TRANSITML_BLS_ENGINE"


def bls_engine() -> str:
    """The BLS engine this process uses, from ``TRANSITML_BLS_ENGINE``."""
    engine = os.environ.get(BLS_ENGINE_ENV, "astropy").strip().lower() or "astropy"
    if engine not in ("astropy", "cpu", "gpu"):
        raise ValueError(f"{BLS_ENGINE_ENV}={engine!r}; expected astropy, cpu or gpu")
    return engine


def run_bls(lc: FlattenedLightCurve, config: BLSConfig | None = None) -> dict[str, Any]:
    """Run the periodic search and return the peak solution plus the power array.

    BLS by default; ``config.search == "tls"`` runs Transit Least Squares
    instead (:mod:`transitml.tls`) and returns the same fields.  The BLS
    periodogram itself comes from astropy or from :mod:`transitml.fastbls`,
    as :func:`bls_engine` says.
    """
    config = config or BLSConfig()
    err = np.where(lc.flux_err > 0, lc.flux_err, lc.scatter)
    periods = period_grid(lc.baseline_days, config)
    if config.search == "tls":
        from .tls import run_tls

        try:
            return {**run_tls(lc, config, float(periods.max())), "searched_with": "tls"}
        except RuntimeError:
            # TLS found nothing it could fit (rare: flat or very short data);
            # the BLS solution stands in for this light curve.
            config = replace(config, search="bls")
    if config.search != "bls":
        raise ValueError(f"unknown search {config.search!r}; expected 'bls' or 'tls'")
    bls = BoxLeastSquares(lc.time, lc.flux, err)
    durations = np.asarray(config.durations_days, dtype=float)
    # Durations longer than the shortest trial period are meaningless.
    durations = durations[durations < 0.5 * periods.min()]
    if durations.size == 0:
        durations = np.array([0.05 * periods.min()])
    engine = bls_engine()
    if engine == "astropy":
        result = bls.power(periods, durations, objective="snr")
    else:
        from .fastbls import array_module, bls_power

        result = _AsResult(
            bls_power(lc.time, lc.flux, err, periods, durations, xp=array_module(engine)),
            periods,
        )

    best = int(np.nanargmax(result.power))
    return {
        "searched_with": "bls",
        "bls": bls,
        "periods": np.asarray(result.period, dtype=float),
        "power": np.asarray(result.power, dtype=float),
        "best_index": best,
        "period": float(result.period[best]),
        "duration": float(result.duration[best]),
        "transit_time": float(result.transit_time[best]),
        "depth": float(result.depth[best]),
        "depth_err": float(result.depth_err[best]),
        "depth_snr": float(result.depth_snr[best]),
        "log_likelihood": float(result.log_likelihood[best]),
    }


class _AsResult:
    """Attribute access to :func:`transitml.fastbls.bls_power` output, like astropy's."""

    def __init__(self, values: dict[str, NDArray[np.float64]], periods: NDArray[np.float64]):
        self.period = periods
        for key, value in values.items():
            setattr(self, key, value)


def signal_detection_efficiency(power: NDArray[np.float64]) -> float:
    """Robust peak significance of the periodogram (the BLS/TLS "SDE").

    ``(peak - median) / (1.4826 * MAD)``.  Using MAD rather than the standard
    deviation matters: the peak itself, plus its aliases, inflate ``np.std``
    enough to suppress the SDE of the strongest signals -- which is precisely
    backwards.
    """
    finite = power[np.isfinite(power)]
    if finite.size < 10:
        return float("nan")
    sigma = robust_sigma(finite)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float((np.nanmax(finite) - np.median(finite)) / sigma)


def power_contrast(
    periods: NDArray[np.float64], power: NDArray[np.float64], best_period: float
) -> float:
    """Peak power divided by the best power at an unrelated period.

    Periods within 10% of the peak and of its 1/2x, 2x and 3x aliases are
    excluded.  A genuine transit produces one isolated peak; correlated noise
    and residual stellar variability produce forests of comparable peaks, so
    this is near 1 for junk and well above 1 for real signals.
    """
    finite = np.isfinite(power)
    mask = finite.copy()
    for multiple in (0.5, 1.0, 2.0, 3.0):
        target = best_period * multiple
        mask &= ~(np.abs(periods - target) < 0.1 * target)
    if not mask.any():
        return float("nan")
    peak = float(np.nanmax(power[finite]))
    background = float(np.nanmax(power[mask]))
    if background <= 0:
        return float("nan")
    return peak / background


def red_noise_beta(
    lc: FlattenedLightCurve, duration_days: float, in_transit: NDArray[np.bool_]
) -> float:
    """Pont, Zucker & Queloz (2006) beta factor on the out-of-transit residuals.

    Bin the residuals into bins one transit-duration wide.  If the noise were
    white, the scatter of the bin means would fall as ``sigma / sqrt(N)``.  The
    ratio of the observed binned scatter to that expectation is ``beta``.
    ``beta ~ 1`` means white noise; ``beta >> 1`` means the light curve has
    correlated structure on exactly the timescale a transit lives on, so the
    nominal depth SNR is overstated by that factor.

    This is the single most important "is this detection real?" statistic in
    ground-based and space-based transit work, and the model gets to use it.
    """
    out = ~in_transit
    if out.sum() < 20 or duration_days <= 0:
        return float("nan")
    time, flux = lc.time[out], lc.flux[out]
    sigma = robust_sigma(flux)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")

    # Bin at the transit duration, but never narrower than a few cadences: at
    # 30-minute sampling the shortest trial durations hold one or two points,
    # and the scatter of one-point "bin means" is just the white noise again.
    cadence = float(np.median(np.diff(lc.time))) if lc.time.size > 1 else duration_days
    width = max(duration_days, 3.0 * cadence)
    edges = np.arange(time[0], time[-1] + width, width)
    if edges.size < 4:
        return float("nan")
    which = np.digitize(time, edges)
    means, counts = [], []
    for b in np.unique(which):
        sel = which == b
        if sel.sum() >= 2:
            means.append(float(np.mean(flux[sel])))
            counts.append(int(sel.sum()))
    if len(means) < 4:
        return float("nan")
    expected = sigma / np.sqrt(np.mean(counts))
    observed = float(np.std(means))
    return float(observed / expected) if expected > 0 else float("nan")


def flat_bottom_fraction(
    lc: FlattenedLightCurve, period: float, duration: float, transit_time: float
) -> float:
    """Fit a trapezoid to the folded event and return T23 / T14.

    ~0.8 for a box-shaped (central) transit, ~0 for a V-shaped grazing event.
    V shapes are the classic eclipsing-binary signature, though grazing planets
    produce them too -- which is why this is one feature among many rather than
    a veto.

    Implementation: the model is linear in depth once the shape is fixed, so we
    grid over the shape parameter only and solve for depth in closed form.  A
    full non-linear fit per light curve would dominate the runtime and buy
    nothing.
    """
    if period <= 0 or duration <= 0:
        return float("nan")
    phase = (lc.time - transit_time + 0.5 * period) % period - 0.5 * period
    window = np.abs(phase) < 1.5 * duration
    if window.sum() < 8:
        return float("nan")

    x = np.abs(phase[window])
    y = lc.flux[window] - 1.0
    half = duration / 2.0

    best_q, best_chi2 = float("nan"), np.inf
    for q in np.linspace(0.0, 0.95, 20):
        ingress = max(half * (1.0 - q), 1e-6)
        profile = np.clip((half - x) / ingress, 0.0, 1.0)
        denom = float(np.sum(profile**2))
        if denom <= 0:
            continue
        depth = -float(np.sum(profile * y)) / denom
        chi2 = float(np.sum((y + depth * profile) ** 2))
        if chi2 < best_chi2:
            best_chi2, best_q = chi2, float(q)
    return best_q


def _pair_sigma(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Significance of the difference between two (value, uncertainty) pairs."""
    va, ea = float(np.ravel(a)[0]), float(np.ravel(a)[1])
    vb, eb = float(np.ravel(b)[0]), float(np.ravel(b)[1])
    denom = np.sqrt(ea**2 + eb**2)
    return float(abs(va - vb) / denom) if denom > 0 else float("nan")


def beta_inflation(beta: float) -> float:
    """Error-bar inflation factor for the binary tests: ``max(beta, 1)``.

    The odd/even and secondary statistics astropy returns are significances
    against *white-noise* error bars.  When the residuals are correlated on the
    transit timescale, a transit-length average scatters ``beta`` times more
    than white noise predicts (Pont, Zucker & Queloz 2006), so those error bars
    are too small by that factor and the significances too large by it.

    The factor is floored at 1: a measured ``beta < 1`` is sampling scatter on
    a white-noise light curve, not evidence that the noise is better than
    white, and must not *inflate* a binary signature.  An undefined ``beta``
    (too few out-of-transit bins) leaves the statistic unscaled.
    """
    if not np.isfinite(beta):
        return 1.0
    return float(max(beta, 1.0))


def extract_features(
    lc: FlattenedLightCurve, config: BLSConfig | None = None
) -> dict[str, float]:
    """Run BLS on a detrended light curve and return the full feature vector.

    Every value is a plain float; missing/ill-defined statistics are returned as
    NaN rather than being imputed, because :class:`HistGradientBoostingClassifier`
    learns a split direction for missing values and "the test could not be
    computed" is itself informative (it usually means too few in-transit points).
    """
    config = config or BLSConfig()
    res = run_bls(lc, config)
    bls, period, duration = res["bls"], res["period"], res["duration"]
    transit_time, depth = res["transit_time"], res["depth"]
    scatter = lc.scatter if lc.scatter > 0 else robust_sigma(lc.flux)

    stats_dict = bls.compute_stats(period, duration, transit_time)
    per_transit_ll = np.asarray(stats_dict["per_transit_log_likelihood"], dtype=float)
    per_transit_n = np.asarray(stats_dict["per_transit_count"], dtype=float)
    observed = per_transit_n > 0
    n_transits = int(observed.sum())

    phase = (lc.time - transit_time + 0.5 * period) % period - 0.5 * period
    in_transit = np.abs(phase) < 0.5 * duration

    # Log-likelihood gain of the box model over a flat line, in sigma-like units.
    flat_ll = -0.5 * np.sum(((lc.flux - 1.0) / np.maximum(lc.flux_err, 1e-12)) ** 2)
    delta_ll = float(res["log_likelihood"] - flat_ll)

    # Measured duration versus the longest duration Kepler's third law permits
    # for a solar-density star at this period.  >> 1 is physically impossible
    # for a planet transit and flags binaries, blends and residual systematics.
    expected = expected_central_duration(period, RHO_SUN_CGS)
    duration_ratio = duration / expected if expected > 0 else np.nan

    secondary_pair = stats_dict["depth_phased"]
    sec_value, sec_err = float(np.ravel(secondary_pair)[0]), float(np.ravel(secondary_pair)[1])
    half_depth = float(np.ravel(stats_dict["depth_half"])[0])
    harmonic_amp = float(np.ravel(stats_dict["harmonic_amplitude"])[0])
    harmonic_dll = float(np.ravel(stats_dict["harmonic_delta_log_likelihood"])[0])

    # The binary tests are judged against the empirical noise on the transit
    # timescale, not the white-noise error bars: divide by beta (>= 1).
    # NaN in stays NaN out.
    #
    # The beta used for this is measured with a wider mask than the
    # ``red_noise_beta`` feature: both the primary and the phase-0.5 window are
    # excluded, each two box-widths wide.  With the feature's exact-box mask,
    # ingress/egress cadences of a slightly mis-fitted period or duration leak
    # into the "out-of-transit" bins, and a single leaked cadence of a deep
    # event dominates the binned scatter: on a pure white-noise planet light
    # curve that mask returns beta ~ 1.5-2.4.  Excluding the secondary window
    # stops a real secondary eclipse from inflating the beta that is then used
    # to discount it.
    beta = red_noise_beta(lc, duration, in_transit)
    noise_mask = (np.abs(phase) < duration) | (np.abs(np.abs(phase) - 0.5 * period) < duration)
    inflation = beta_inflation(red_noise_beta(lc, duration, noise_mask))
    odd_even_raw = _pair_sigma(stats_dict["depth_odd"], stats_dict["depth_even"])
    secondary_raw = float(sec_value / sec_err) if sec_err > 0 else np.nan

    total_ll = float(np.sum(np.abs(per_transit_ll[observed]))) if n_transits else 0.0
    max_single = (
        float(np.max(np.abs(per_transit_ll[observed])) / total_ll)
        if total_ll > 0
        else np.nan
    )

    features: dict[str, float] = {
        # --- detection strength ---
        "bls_sde": signal_detection_efficiency(res["power"]),
        "bls_depth_snr": float(res["depth_snr"]),
        "bls_depth_over_scatter": float(depth / scatter) if scatter > 0 else np.nan,
        "delta_loglike": delta_ll,
        "power_contrast": power_contrast(res["periods"], res["power"], period),
        # --- geometry / plausibility ---
        "log_depth": float(np.log10(max(depth, 1e-8))),
        "bls_duration": float(duration),
        "duration_over_period": float(duration / period),
        "log_duration_ratio": float(np.log10(duration_ratio))
        if np.isfinite(duration_ratio) and duration_ratio > 0
        else np.nan,
        "log_period": float(np.log10(period)),
        "n_transits": float(n_transits),
        "min_points_per_transit": float(per_transit_n[observed].min())
        if n_transits
        else np.nan,
        # --- false-positive discriminants ---
        "odd_even_sigma": odd_even_raw / inflation,
        "secondary_sigma": secondary_raw / inflation,
        "half_period_depth_ratio": float(half_depth / depth) if depth != 0 else np.nan,
        "flat_bottom_fraction": flat_bottom_fraction(lc, period, duration, transit_time),
        "harmonic_delta_loglike": harmonic_dll,
        "harmonic_amp_over_depth": float(harmonic_amp / depth) if depth > 0 else np.nan,
        # --- noise characterisation ---
        "red_noise_beta": beta,
        "log_scatter": float(np.log10(max(scatter, 1e-9))),
        "max_single_event_fraction": max_single,
        "flux_skew": float(stats.skew(lc.flux)),
        "clipped_fraction": float(lc.n_clipped / max(lc.time.size + lc.n_clipped, 1)),
    }
    assert set(features) == set(FEATURE_NAMES), "feature dict does not match FEATURE_NAMES"
    return {name: float(features[name]) for name in FEATURE_NAMES}
