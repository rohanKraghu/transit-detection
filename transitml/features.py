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
    The two significances are divided by the red-noise ``beta`` or by the
    event-to-event depth scatter ratio, whichever is larger (floored at 1), so
    correlated noise and inconsistent events cannot masquerade as a binary
    signature.  The secondary counts only what is deeper than the planet's own
    occultation could be around its star.

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

from .config import BLSConfig, PreprocessConfig
from .data.base import LightCurve
from .physics import RHO_SUN_CGS, expected_central_duration, max_occultation_fraction
from .preprocess import FlattenedLightCurve, flatten, robust_sigma

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

    On a log grid with step d(ln P), a transit's phase drifts by
    ``baseline * d(ln P)`` between neighbouring trials whatever the period, so
    a baseline longer than ``grid_baseline_days`` gets more than
    ``n_periods``: as many as keep that drift to what ``n_periods`` allow on
    ``grid_baseline_days``.
    """
    p_max = _max_period(baseline_days, config)
    reference = config.grid_baseline_days
    allowed = reference * np.log(_max_period(reference, config) / config.min_period_days)
    allowed /= config.n_periods - 1
    steps = np.log(p_max / config.min_period_days) * baseline_days / allowed
    n = max(config.n_periods, int(np.ceil(steps - 1e-9)) + 1)
    return np.logspace(np.log10(config.min_period_days), np.log10(p_max), n)


def _max_period(baseline_days: float, config: BLSConfig) -> float:
    return max(config.min_period_days * 1.5, baseline_days * config.max_period_fraction_of_baseline)


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
    as :func:`bls_engine` says.  A BLS search of a curve longer than
    ``config.max_search_baseline_days`` is a :func:`windowed_search`.
    """
    config = config or BLSConfig()
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
    limit = config.max_search_baseline_days
    if limit is not None and lc.baseline_days > limit:
        return windowed_search(lc, config)
    durations = _durations(config, periods)
    result = _periodogram(lc, periods, durations)
    best = int(np.nanargmax(result.power))
    return _solution(lc, result, best, periods)


def _durations(config: BLSConfig, periods: NDArray[np.float64]) -> NDArray[np.float64]:
    """The trial durations shorter than half the shortest trial period."""
    durations = np.asarray(config.durations_days, dtype=float)
    # Durations longer than the shortest trial period are meaningless.
    durations = durations[durations < 0.5 * periods.min()]
    if durations.size == 0:
        durations = np.array([0.05 * periods.min()])
    return durations


def _periodogram(
    lc: FlattenedLightCurve, periods: NDArray[np.float64], durations: NDArray[np.float64]
) -> Any:
    """The BLS periodogram of ``lc`` on ``periods``, from the engine :func:`bls_engine` names."""
    err = np.where(lc.flux_err > 0, lc.flux_err, lc.scatter)
    engine = bls_engine()
    if engine == "astropy":
        return BoxLeastSquares(lc.time, lc.flux, err).power(periods, durations, objective="snr")
    from .fastbls import array_module, bls_power

    return _AsResult(
        bls_power(lc.time, lc.flux, err, periods, durations, xp=array_module(engine)),
        periods,
    )


def _solution(
    lc: FlattenedLightCurve, result: Any, index: int, periods: NDArray[np.float64]
) -> dict[str, Any]:
    """:func:`run_bls`'s dictionary: the solution at ``result[index]`` and the periodogram."""
    err = np.where(lc.flux_err > 0, lc.flux_err, lc.scatter)
    return {
        "searched_with": "bls",
        "bls": BoxLeastSquares(lc.time, lc.flux, err),
        "periods": np.asarray(periods, dtype=float),
        "power": np.asarray(result.power, dtype=float),
        "best_index": index,
        "period": float(result.period[index]),
        "duration": float(result.duration[index]),
        "transit_time": float(result.transit_time[index]),
        "depth": float(result.depth[index]),
        "depth_err": float(result.depth_err[index]),
        "depth_snr": float(result.depth_snr[index]),
        "log_likelihood": float(result.log_likelihood[index]),
    }


class _AsResult:
    """Attribute access to :func:`transitml.fastbls.bls_power` output, like astropy's."""

    def __init__(self, values: dict[str, NDArray[np.float64]], periods: NDArray[np.float64]):
        self.period = periods
        for key, value in values.items():
            setattr(self, key, value)


#: Period ratios at which two trial periods are taken for one signal.
_ALIAS_RATIOS = np.array([1.0, 0.5, 2.0, 1.0 / 3.0, 3.0])
#: Multiples of a windowed search's candidate that the whole curve tries.
_REFINED_ALIASES = (1.0, 0.5, 2.0)


def distinct_peaks(
    periods: NDArray[np.float64], power: NDArray[np.float64], k: int
) -> list[int]:
    """Indices of the ``k`` strongest peaks of a periodogram that are different signals.

    Walking down from the strongest trial period, one is skipped when it lies
    within 1% of a period already taken or of its 1/3, 1/2, 2 or 3 times
    alias: the neighbouring trial periods of one peak, and the aliases of one
    signal, are one candidate, not several.
    """
    finite = np.isfinite(power)
    order = np.argsort(np.where(finite, power, -np.inf), kind="stable")[::-1]
    taken: list[int] = []
    for i in order:
        if not finite[i] or len(taken) == k:
            break
        ratio = periods[i] / periods[taken] if taken else np.empty(0)
        if np.any(np.abs(ratio[:, None] - _ALIAS_RATIOS) < 0.01 * _ALIAS_RATIOS):
            continue
        taken.append(int(i))
    return taken


def densest_window(time: NDArray[np.float64], length_days: float) -> tuple[float, float]:
    """``(start, stop)`` of the ``length_days`` stretch holding the most cadences.

    It starts on a cadence; of equally full stretches, the earliest.
    """
    time = np.sort(np.asarray(time, dtype=float))
    stops = np.searchsorted(time, time + length_days, side="right")
    first = int(np.argmax(stops - np.arange(time.size)))
    return float(time[first]), float(time[first] + length_days)


def _grid_step(baseline_days: float, config: BLSConfig) -> float:
    """Step in ln(period) of :func:`period_grid` on a baseline of ``baseline_days``."""
    periods = period_grid(baseline_days, config)
    return float(np.log(periods[-1] / periods[0]) / (periods.size - 1))


def windowed_search(lc: FlattenedLightCurve, config: BLSConfig) -> dict[str, Any]:
    """BLS on a curve too long for one grid: candidates from its densest stretch, chosen on all of it.

    1. The usual grid search (:func:`period_grid`) runs on the densest
       ``config.max_search_baseline_days`` of the curve (:func:`densest_window`).
    2. Each of that periodogram's ``config.candidate_peaks`` strongest
       :func:`distinct_peaks`, its half and its double, is searched again on
       the whole curve, on a grid as fine as a single grid over the whole
       curve would have been, across the candidate's peak: half its duration
       over the window's baseline, and at least two window steps, either
       side.  The half and double count because a stretch with gaps often
       cannot tell a period from them, where the rest of the data can.
    3. The highest of those whole-curve peaks is the signal.

    The ``"periods"`` and ``"power"`` returned are the window's periodogram,
    so the peak-significance features describe the search that proposed the
    signal, and ``"best_index"`` is the candidate that won; the rest of the
    solution, and ``"bls"``, are from the whole curve.  A few hundred
    periods around each of ten candidates are a few thousand, where one grid
    over five years would need about 250,000.  A signal too weak to be among
    the window's candidates, or with fewer than two transits in it, is not
    found.
    """
    start, stop = densest_window(lc.time, float(config.max_search_baseline_days))
    keep = (lc.time >= start) & (lc.time <= stop)
    window = replace(
        lc, time=lc.time[keep], flux=lc.flux[keep], flux_err=lc.flux_err[keep], trend=lc.trend[keep]
    )
    periods = period_grid(window.baseline_days, config)
    coarse = _periodogram(window, periods, _durations(config, periods))
    power = np.asarray(coarse.power, dtype=float)
    window_step = float(np.log(periods[1] / periods[0]))
    fine = min(_grid_step(lc.baseline_days, config), window_step)
    longest = _max_period(lc.baseline_days, config)
    # (power, candidate, period, durations tried there)
    best: tuple[float, int, float, NDArray[np.float64]] | None = None
    for i in distinct_peaks(periods, power, config.candidate_peaks):
        half = max(0.5 * float(coarse.duration[i]) / window.baseline_days, 2 * window_step)
        n = int(np.ceil(half / fine))
        for ratio in _REFINED_ALIASES:
            grid = periods[i] * ratio * np.exp(fine * np.arange(-n, n + 1))
            grid = grid[(grid >= config.min_period_days) & (grid <= longest)]
            if grid.size == 0:
                continue
            durations = _durations(config, grid)
            result = _periodogram(lc, grid, durations)
            j = int(np.nanargmax(result.power))
            if best is None or result.power[j] > best[0]:
                best = (float(result.power[j]), i, float(grid[j]), durations)
    if best is None:
        index = int(np.nanargmax(power))
        return {**_solution(window, coarse, index, periods), "search_window": (start, stop)}
    _, index, period, durations = best
    one = np.array([period])
    result = _periodogram(lc, one, durations)
    return {
        **_solution(lc, result, 0, one),
        "periods": periods,
        "power": power,
        "best_index": index,
        "search_window": (start, stop),
    }


def flatten_masked(
    lc: LightCurve,
    preprocess: PreprocessConfig | None = None,
    bls: BLSConfig | None = None,
) -> FlattenedLightCurve:
    """Detrend, find the strongest signal, then detrend again with it masked.

    The robust fit keeps a transit out of the trend by giving its cadences zero
    weight, but only if the first, unweighted iteration leaves them as
    outliers.  Against a data gap it does not: the edge spline function bends
    into the dip, the baseline cadences beside it are then clipped as upward
    outliers, and the refit erases the event.  The second pass keeps every
    cadence within ``mask_half_width_durations`` BLS durations of a transit
    out of the fit, so the trend is interpolated across the events instead of
    fitted to them.

    Returns the blind detrend when masking is switched off, when the strongest
    peak is not a dip or is weaker than ``mask_min_sde``, or when the mask
    would cover more than ``mask_max_fraction`` of the cadences.
    """
    return _masked_detrend(lc, preprocess, bls)[0]


def detrend_and_search(
    lc: LightCurve,
    preprocess: PreprocessConfig | None = None,
    bls: BLSConfig | None = None,
) -> tuple[FlattenedLightCurve, dict[str, Any]]:
    """:func:`flatten_masked`, plus the BLS search of the curve it returns.

    Deciding whether to mask already searched the blind detrend, so when that
    is the curve kept (about nine in ten on the synthetic run) its search is
    handed on instead of run again.  BLS is the pipeline's bottleneck.
    """
    bls = bls or BLSConfig()
    flat, search = _masked_detrend(lc, preprocess, bls)
    return flat, search if search is not None else run_bls(flat, bls)


def _masked_detrend(
    lc: LightCurve, preprocess: PreprocessConfig | None, bls: BLSConfig | None
) -> tuple[FlattenedLightCurve, dict[str, Any] | None]:
    """:func:`flatten_masked`'s detrend, and the search of it if one was run."""
    preprocess = preprocess or PreprocessConfig()
    bls = bls or BLSConfig()
    lc = lc.finite()
    blind = flatten(lc, preprocess)
    if not preprocess.mask_signal:
        return blind, None
    found = run_bls(blind, bls)
    if not found["depth"] > 0:
        return blind, found
    peak = found["power"][found["best_index"]]
    if not signal_detection_efficiency(found["power"], peak) >= preprocess.mask_min_sde:
        return blind, found
    period, epoch = found["period"], found["transit_time"]
    phase = (lc.time - epoch + 0.5 * period) % period - 0.5 * period
    exclude = np.abs(phase) < preprocess.mask_half_width_durations * found["duration"]
    if not exclude.any() or exclude.mean() > preprocess.mask_max_fraction:
        return blind, found
    return flatten(lc, preprocess, exclude=exclude), None


def signal_detection_efficiency(
    power: NDArray[np.float64], peak: float | None = None
) -> float:
    """Robust peak significance of the periodogram (the BLS/TLS "SDE").

    ``(peak - median) / (1.4826 * MAD)``.  Using MAD rather than the standard
    deviation matters: the peak itself, plus its aliases, inflate ``np.std``
    enough to suppress the SDE of the strongest signals -- which is precisely
    backwards.  ``peak`` is the power of the peak taken as the signal; the
    highest by default.
    """
    finite = power[np.isfinite(power)]
    if finite.size < 10:
        return float("nan")
    sigma = robust_sigma(finite)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    top = float(np.nanmax(finite)) if peak is None else float(peak)
    return float((top - np.median(finite)) / sigma)


def power_contrast(
    periods: NDArray[np.float64],
    power: NDArray[np.float64],
    best_period: float,
    peak: float | None = None,
) -> float:
    """Peak power divided by the best power at an unrelated period.

    Periods within 10% of the peak and of its 1/2x, 2x and 3x aliases are
    excluded.  A genuine transit produces one isolated peak; correlated noise
    and residual stellar variability produce forests of comparable peaks, so
    this is near 1 for junk and well above 1 for real signals.  ``peak`` is
    the power at ``best_period``; the highest by default.
    """
    finite = np.isfinite(power)
    mask = finite.copy()
    for multiple in (0.5, 1.0, 2.0, 3.0):
        target = best_period * multiple
        mask &= ~(np.abs(periods - target) < 0.1 * target)
    if not mask.any():
        return float("nan")
    peak = float(np.nanmax(power[finite])) if peak is None else float(peak)
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


def event_depths(
    lc: FlattenedLightCurve, period: float, duration: float, transit_time: float
) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
    """Box depth of every observed event, one event at a time.

    The weighted mean flux inside the box, against the weighted mean of every
    cadence outside both this box and the one half a period away (so that
    neither eclipse sits in the other's baseline), with the white-noise error
    from ``flux_err``: the BLS depth estimate, applied per event.  Pass
    ``transit_time + period / 2`` for the events at phase 0.5.  Returns
    ``(epoch numbers, depths, errors)``.
    """
    err = np.where(lc.flux_err > 0, lc.flux_err, lc.scatter)
    weight = 1.0 / np.maximum(err, 1e-12) ** 2
    epoch = np.round((lc.time - transit_time) / period).astype(int)
    offset = lc.time - transit_time - epoch * period
    inside = np.abs(offset) < 0.5 * duration
    outside = ~inside & (np.abs(np.abs(offset) - 0.5 * period) >= 0.5 * duration)
    empty = np.array([], dtype=float)
    if not inside.any() or not outside.any():
        return np.array([], dtype=int), empty, empty
    baseline = float(np.sum(weight[outside] * lc.flux[outside]) / np.sum(weight[outside]))
    epochs = np.unique(epoch[inside])
    depths, errors = np.empty(epochs.size), np.empty(epochs.size)
    for i, k in enumerate(epochs):
        sel = inside & (epoch == k)
        total = float(np.sum(weight[sel]))
        depths[i] = baseline - float(np.sum(weight[sel] * lc.flux[sel])) / total
        errors[i] = 1.0 / np.sqrt(total)
    return epochs, depths, errors


def depth_scatter_ratio(
    epochs: NDArray[np.int_],
    depths: NDArray[np.float64],
    errors: NDArray[np.float64],
    *,
    by_parity: bool,
) -> float:
    """Birge ratio ``sqrt(chi2 / dof)`` of per-event depths about their mean.

    1 when the events agree to within their white-noise errors.  Above 1 when
    they scatter more than that: red noise on the transit timescale, a
    ``flux_err`` that understates the real scatter, stellar pulsation, or the
    way a 30-minute cadence samples the ingress differently in every event.
    Each of these makes a difference between two groups of events look more
    significant than it is, by this factor.

    With ``by_parity``, odd and even events are each compared with their own
    mean, so an alternating depth, the binary signature itself, does not count
    as scatter.  NaN with fewer than two degrees of freedom.
    """
    groups = [epochs % 2 == 0, epochs % 2 == 1] if by_parity else [np.ones(epochs.size, bool)]
    chi2, dof = 0.0, 0
    for group in groups:
        if not group.any():
            continue
        weight = 1.0 / errors[group] ** 2
        mean = float(np.sum(weight * depths[group]) / np.sum(weight))
        chi2 += float(np.sum(weight * (depths[group] - mean) ** 2))
        dof += int(group.sum()) - 1
    return float(np.sqrt(chi2 / dof)) if dof >= 2 else float("nan")


def secondary_excess(secondary_depth: float, allowance: float) -> float:
    """The part of the phase-0.5 depth that no planet's occultation could produce.

    A hot Jupiter's own occultation is a secondary eclipse, and on a bright
    star a significant one, so the secondary test discounts the deepest one a
    planet could show (``allowance``, see
    :func:`~transitml.physics.max_occultation_fraction`).  A dip within it
    reads as zero, a brightening is left as it is, and with no allowance
    (NaN, an unknown star) the whole depth counts.
    """
    if not np.isfinite(allowance) or allowance <= 0:
        return float(secondary_depth)
    return float(secondary_depth - np.clip(secondary_depth, 0.0, allowance))


def extract_features(
    lc: FlattenedLightCurve,
    config: BLSConfig | None = None,
    *,
    search: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Run BLS on a detrended light curve and return the full feature vector.

    Every value is a plain float; missing/ill-defined statistics are returned as
    NaN rather than being imputed, because :class:`HistGradientBoostingClassifier`
    learns a split direction for missing values and "the test could not be
    computed" is itself informative (it usually means too few in-transit points).

    ``search`` is ``run_bls(lc, config)`` when the caller already has it (see
    :func:`detrend_and_search`); it is not checked against ``lc``.
    """
    config = config or BLSConfig()
    res = search if search is not None else run_bls(lc, config)
    bls, period, duration = res["bls"], res["period"], res["duration"]
    transit_time, depth = res["transit_time"], res["depth"]
    scatter = lc.scatter if lc.scatter > 0 else robust_sigma(lc.flux)
    # The periodogram peak the signal came from: the highest one, except
    # after a windowed search (whose period is then refined on all the data).
    peak = float(res["power"][res["best_index"]])
    peak_period = float(res["periods"][res["best_index"]])

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
    #
    # The events themselves are the other check on those error bars.  Each
    # binary test compares groups of events (odd with even, phase 0.5 with
    # flat), so it is overstated by however much more the events scatter
    # than their white-noise errors allow.  On real TESS photometry of bright
    # hot Jupiters that factor is 2-7 even after a clean detrend, and without
    # it confirmed planets read as 10-40 sigma odd/even "binaries".
    beta = red_noise_beta(lc, duration, in_transit)
    noise_mask = (np.abs(phase) < duration) | (np.abs(np.abs(phase) - 0.5 * period) < duration)
    inflation = beta_inflation(red_noise_beta(lc, duration, noise_mask))
    primary_ratio = depth_scatter_ratio(
        *event_depths(lc, period, duration, transit_time), by_parity=True
    )
    secondary_ratio = depth_scatter_ratio(
        *event_depths(lc, period, duration, transit_time + 0.5 * period), by_parity=False
    )
    # Both ratios are floored at 1 and NaN-safe, exactly like beta.
    odd_even_inflation = max(inflation, beta_inflation(primary_ratio))
    secondary_inflation = max(inflation, beta_inflation(secondary_ratio))
    odd_even_raw = _pair_sigma(stats_dict["depth_odd"], stats_dict["depth_even"])
    allowance = max_occultation_fraction(period, lc.teff_k, lc.density_cgs) * max(depth, 0.0)
    secondary_raw = (
        float(secondary_excess(sec_value, allowance) / sec_err) if sec_err > 0 else np.nan
    )

    total_ll = float(np.sum(np.abs(per_transit_ll[observed]))) if n_transits else 0.0
    max_single = (
        float(np.max(np.abs(per_transit_ll[observed])) / total_ll)
        if total_ll > 0
        else np.nan
    )

    features: dict[str, float] = {
        # --- detection strength ---
        "bls_sde": signal_detection_efficiency(res["power"], peak),
        "bls_depth_snr": float(res["depth_snr"]),
        "bls_depth_over_scatter": float(depth / scatter) if scatter > 0 else np.nan,
        "delta_loglike": delta_ll,
        "power_contrast": power_contrast(res["periods"], res["power"], peak_period, peak),
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
        "odd_even_sigma": odd_even_raw / odd_even_inflation,
        "secondary_sigma": secondary_raw / secondary_inflation,
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
