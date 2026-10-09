"""Detrending: remove the star without removing the planet.

This is the crux of the whole project.  A transit is a 0.05-1% dip lasting a
few hours; the rotational modulation it sits on is routinely 10-100x deeper and
varies over days.  Anything that removes the star too aggressively removes the
planet with it, and no downstream model recovers a signal that preprocessing
has already destroyed.

What is actually done
---------------------
One **robust fit** of one trend model, plus a one-sided outlier clip::

    trend(t) = cubic B-spline(t) + Fourier terms at the star's rotation period

fitted by iteratively reweighted least squares with a Tukey biweight loss, and
subtracted.  The spline handles slow, non-periodic drift (spot evolution,
thermal trends, scattered light); the Fourier terms handle coherent variability
too fast for a spline with a safe knot spacing to follow.  They are fitted
*together*, in one design matrix, so there is no order-of-operations question
about which stage gets to claim which signal.

Three decisions, each with a cheaper wrong answer
-------------------------------------------------
**A robust fit, not a running median.**
    The running median is the obvious choice and it is wrong, in a way that is
    easy to miss.  The median of a window over a *monotone* series is that
    window's centre value -- exactly.  On a star varying fast enough that its
    trend moves by more than the photometric noise across one window, the
    running median therefore reproduces the data, noise and transit included,
    and removing it removes the transit.  Measured on a 0.8% variable carrying
    a 3000 ppm transit, a 0.75 d running median recovers **41%** of the
    injected depth; the robust spline fit recovers **101%**, i.e. unbiased to
    within the measurement's own noise.  Both are pinned by
    ``test_running_median_eats_the_transit_on_a_steep_star``.

    The robust fit has no such failure mode because it never tries to pass
    through the data.  The transit becomes a cluster of large, one-sided
    residuals, the biweight drives their weights to zero, and the trend ends up
    fitted to out-of-transit cadences only.  This is the "mask the transit,
    then fit the baseline" recipe Kepler and TESS pipelines apply explicitly --
    here the mask is discovered rather than supplied.

**Knot spacing of at least ~3x the longest transit duration.**
    That protection only works while the transit *is* an outlier.  Give the
    spline knots tight enough to bend into a 3-hour dip and the first
    (unweighted) iteration fits the transit, the residuals there come out
    small, the biweight sees nothing to reject, and IRLS converges to the wrong
    answer.  At 0.75 d knots and a 0.26 d maximum duration, no basis function
    is narrow enough to absorb the event.  That floor is also why the Fourier
    terms are needed at all: no safe knot spacing can follow a 12-hour rotator.

**Clip outliers upward only.**
    Flares and cosmic rays are positive excursions; transits are negative ones.
    A symmetric sigma clip -- the default in most tutorials -- deletes exactly
    the cadences the search depends on, and deletes them hardest for the
    deepest, most detectable transits.  Clipping is one-sided everywhere here.

Everything is done additively on ``flux / median(flux) - 1``.  At the 1% level
these signals live at, additive and multiplicative removal agree to better than
one part in 10^4, and additive removal makes "the depth is preserved" a
statement that can be checked exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import LombScargle
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.ndimage import median_filter

from .config import PreprocessConfig
from .data.base import LightCurve


@dataclass
class FlattenedLightCurve:
    """A detrended light curve plus the trend that was removed.

    Attributes
    ----------
    time, flux, flux_err:
        The retained cadences.  ``flux`` is the detrended relative flux,
        centred on 1.0.
    trend:
        The removed baseline on the input flux scale, so that
        ``flux + trend - 1`` reproduces the normalised input exactly.
    scatter:
        Robust (MAD-based) scatter of the detrended flux: the empirical noise
        floor, and what significance should be measured against -- not the
        reported ``flux_err``, which ignores red noise entirely.
    n_clipped:
        Number of upward outliers removed.
    rotation_periods:
        Periods given Fourier terms in the trend model.  Empty when the star
        showed no coherent fast variability.
    n_masked:
        Retained cadences the caller asked to keep out of the trend fit (zero
        for a blind detrend).  A few may have been given back to keep the fit
        constrained; see :func:`release_starved`.
    teff_k, density_cgs:
        The host star's effective temperature and mean density, carried over
        from the input curve (:attr:`~transitml.data.base.LightCurve.star`);
        NaN when unknown.
    """

    target_id: str
    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    flux_err: NDArray[np.float64]
    trend: NDArray[np.float64]
    scatter: float
    n_clipped: int
    label: int | None = None
    rotation_periods: tuple[float, ...] = ()
    n_masked: int = 0
    teff_k: float = float("nan")
    density_cgs: float = float("nan")

    @property
    def baseline_days(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.time.size else 0.0


# --------------------------------------------------------------------------
# Noise estimators
# --------------------------------------------------------------------------
def robust_sigma(x: NDArray[np.float64]) -> float:
    """MAD-based standard-deviation estimate (``1.4826 * MAD``).

    Used rather than ``np.std`` because a single flare inflates the standard
    deviation enough to make any threshold built on it useless.
    """
    if x.size == 0:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def point_to_point_sigma(flux: NDArray[np.float64]) -> float:
    """Per-cadence noise from successive differences: ``robust_sigma(diff) / sqrt(2)``.

    The obvious estimator -- the scatter about a fitted trend -- is circular
    here: when the trend model is failing, which is exactly the case we need to
    detect, its residuals are inflated by the variability we are measuring
    against.  Differencing suppresses anything smooth on the cadence timescale,
    so this stays close to the white-noise floor whatever the star is doing.
    Same idea as the CDPP statistic the Kepler pipeline reports.
    """
    if flux.size < 3:
        return float("nan")
    return robust_sigma(np.diff(flux)) / np.sqrt(2.0)


def split_on_gaps(
    time: NDArray[np.float64], gap_threshold_days: float
) -> list[NDArray[np.int_]]:
    """Index arrays for contiguous segments separated by large gaps.

    Fitting one trend across the TESS mid-sector downlink gap would let each
    half of the sector influence the other; every segment gets its own spline.
    """
    if time.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(time) > gap_threshold_days) + 1
    return [seg for seg in np.split(np.arange(time.size), breaks) if seg.size > 0]


# --------------------------------------------------------------------------
# Robust least squares
# --------------------------------------------------------------------------
def robust_least_squares(
    design: NDArray[np.float64] | SegmentedDesign,
    values: NDArray[np.float64],
    *,
    iterations: int = 6,
    tuning: float = 4.685,
    exclude: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit ``design @ c ~ values`` with a Tukey biweight loss.  Returns ``(c, weights)``.

    ``tuning = 4.685`` is the textbook constant giving 95% efficiency against
    Gaussian noise.  Residuals beyond ``tuning`` robust sigma receive exactly
    zero weight, so a transit -- a small minority of cadences with large,
    one-signed residuals -- drops out of the fit within two or three
    iterations, with no externally supplied noise scale and no threshold to
    tune.

    ``exclude`` marks rows known in advance to hold signal (the cadences of a
    transit already found).  They carry zero weight from the first, otherwise
    unweighted, iteration on, and are left out of the residual scale.
    """
    include = np.ones(values.size) if exclude is None else (~exclude).astype(float)
    weights = include.copy()
    coefficients = np.zeros(design.shape[1])
    for _ in range(max(iterations, 1)):
        root = np.sqrt(weights)
        if isinstance(design, SegmentedDesign):
            coefficients = design.weighted_lstsq(values, root)
        else:
            coefficients, *_ = np.linalg.lstsq(
                design * root[:, None], values * root, rcond=None
            )
        residual = values - design @ coefficients
        scale = robust_sigma(residual if exclude is None else residual[~exclude])
        if not np.isfinite(scale) or scale <= 0:
            break
        u = residual / (tuning * scale)
        updated = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0) * include
        if updated.sum() < design.shape[1] + 8:
            break
        converged = np.allclose(updated, weights, atol=1e-3)
        weights = updated
        if converged:
            break
    return coefficients, weights


# --------------------------------------------------------------------------
# Trend model
# --------------------------------------------------------------------------
def spline_blocks(
    time: NDArray[np.float64], config: PreprocessConfig
) -> list[tuple[NDArray[np.int_], NDArray[np.float64]]]:
    """``(rows, basis)`` for each gap-free segment: the pieces of :func:`spline_basis`."""
    degree = config.spline_degree
    blocks: list[tuple[NDArray[np.int_], NDArray[np.float64]]] = []
    for segment in split_on_gaps(time, config.gap_threshold_days):
        x = time[segment]
        span = float(x[-1] - x[0]) if x.size else 0.0
        if x.size < degree + 3 or span <= 0:
            blocks.append((segment, np.ones((segment.size, 1))))
            continue
        n_interior = max(int(np.ceil(span / config.knot_spacing_days)) - 1, 0)
        n_interior = min(n_interior, max(x.size - degree - 2, 0))
        interior = np.linspace(x[0], x[-1], n_interior + 2)[1:-1]
        knots = np.concatenate(
            [np.full(degree + 1, x[0]), interior, np.full(degree + 1, x[-1])]
        )
        blocks.append((segment, BSpline.design_matrix(x, knots, degree, extrapolate=False).toarray()))
    return blocks


def spline_basis(
    time: NDArray[np.float64], config: PreprocessConfig
) -> NDArray[np.float64]:
    """Cubic B-spline basis, block-diagonal over gap-free segments.

    Each segment gets its own basis functions, so none spans a data gap and the
    two halves of a sector are detrended independently while still being fitted
    in one linear system.
    """
    return SegmentedDesign.spline(time, config).toarray()


@dataclass(frozen=True)
class SegmentedDesign:
    """The trend model's design matrix, kept segment by segment.

    The spline columns are block-diagonal over gap-free segments; only the
    rotation columns (``shared``) span the whole curve.  Solved in that form, a
    weighted fit is one small least-squares problem per segment plus one with a
    column per rotation term, so its cost grows with the curve's length instead
    of with its cube: a 34-sector curve, about 41,000 cadences and 1,300 spline
    columns, took about 400 s to detrend as one dense matrix.  The fit is the
    dense one's (the same minimum-norm solution), up to rounding.
    """

    n_rows: int
    segments: tuple[NDArray[np.int_], ...]
    blocks: tuple[NDArray[np.float64], ...]
    shared: NDArray[np.float64]

    @classmethod
    def spline(cls, time: NDArray[np.float64], config: PreprocessConfig) -> SegmentedDesign:
        pieces = spline_blocks(time, config) or [(np.arange(time.size), np.ones((time.size, 1)))]
        return cls(
            time.size,
            tuple(rows for rows, _ in pieces),
            tuple(block for _, block in pieces),
            np.zeros((time.size, 0)),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, sum(b.shape[1] for b in self.blocks) + self.shared.shape[1]

    def with_columns(self, columns: NDArray[np.float64]) -> SegmentedDesign:
        """The same design with ``columns`` added to the shared ones."""
        return SegmentedDesign(
            self.n_rows, self.segments, self.blocks, np.hstack([self.shared, columns])
        )

    def toarray(self) -> NDArray[np.float64]:
        parts = []
        for rows, block in zip(self.segments, self.blocks, strict=True):
            part = np.zeros((self.n_rows, block.shape[1]))
            part[rows] = block
            parts.append(part)
        return np.hstack([*parts, self.shared])

    def __matmul__(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.empty(self.n_rows)
        start = 0
        for rows, block in zip(self.segments, self.blocks, strict=True):
            out[rows] = block @ coefficients[start : start + block.shape[1]]
            start += block.shape[1]
        return out + self.shared @ coefficients[start:]

    def weighted_lstsq(
        self, values: NDArray[np.float64], root: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Minimise ``|root * (values - design @ c)|``: ``np.linalg.lstsq``, by segment.

        Each segment's spline is fitted to the values and to every shared
        column at once; the shared coefficients then come from what the
        splines leave of both, and each segment's from its own fits less the
        shared part.
        """
        targets = np.column_stack([values, self.shared]) * root[:, None]
        left = np.empty_like(targets)
        local = []
        for rows, block in zip(self.segments, self.blocks, strict=True):
            weighted = block * root[rows, None]
            solution, *_ = np.linalg.lstsq(weighted, targets[rows], rcond=None)
            left[rows] = targets[rows] - weighted @ solution
            local.append(solution)
        shared = (
            np.linalg.lstsq(left[:, 1:], left[:, 0], rcond=None)[0]
            if self.shared.shape[1]
            else np.zeros(0)
        )
        return np.concatenate([s[:, 0] - s[:, 1:] @ shared for s in local] + [shared])

    def column_mass(self, rows: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Each column's squared weight over the rows marked in ``rows``."""
        local = [np.sum(b[rows[r]] ** 2, axis=0) for r, b in zip(self.segments, self.blocks, strict=True)]
        return np.concatenate([*local, np.sum(self.shared[rows] ** 2, axis=0)])

    def rows_using(self, columns: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Rows where any of the marked columns is non-zero."""
        out = np.zeros(self.n_rows, dtype=bool)
        start = 0
        for rows, block in zip(self.segments, self.blocks, strict=True):
            wanted = columns[start : start + block.shape[1]]
            if wanted.any():
                out[rows] |= np.any(block[:, wanted] != 0.0, axis=1)
            start += block.shape[1]
        return out | np.any(self.shared[:, columns[start:]] != 0.0, axis=1)


def harmonic_basis(
    time: NDArray[np.float64], period: float, n_harmonics: int
) -> NDArray[np.float64]:
    """Columns ``[cos(k w t), sin(k w t)]`` for ``k = 1..n_harmonics``.

    No constant column: the spline already spans the overall level, and a
    second one would make the design rank-deficient.
    """
    omega = 2.0 * np.pi / period
    columns: list[NDArray[np.float64]] = []
    for k in range(1, n_harmonics + 1):
        columns.append(np.cos(k * omega * time))
        columns.append(np.sin(k * omega * time))
    return np.column_stack(columns)


def winsorize(values: NDArray[np.float64], n_sigma: float = 3.0) -> NDArray[np.float64]:
    """Clip to the central ``n_sigma`` robust sigma, for periodogram input only.

    A deep transit is itself a periodic signal and can define the Lomb-Scargle
    peak of an otherwise quiet star.  Clipping its extremes before the search
    keeps the rotation model from being aimed at the planet.  The unclipped
    values are used for the fit.
    """
    scale = robust_sigma(values)
    if not np.isfinite(scale) or scale <= 0:
        return values
    centre = float(np.median(values))
    return np.clip(values, centre - n_sigma * scale, centre + n_sigma * scale)


def dominant_period(
    time: NDArray[np.float64],
    residual: NDArray[np.float64],
    config: PreprocessConfig,
) -> float:
    """Lomb-Scargle peak period of the spline residual, within the rotation window.

    Bounded above by ``rotation_max_period_days``: slower variability is the
    spline's job, and modelling it with sinusoids would only add ringing.
    Bounded below by ``rotation_min_period_days``, roughly the shortest period a
    30-minute cadence resolves usefully.
    """
    p_min, p_max = config.rotation_min_period_days, config.rotation_max_period_days
    if time.size < 32 or p_min >= p_max or time[-1] <= time[0]:
        return float("nan")
    frequency = np.linspace(1.0 / p_max, 1.0 / p_min, config.rotation_n_frequencies)
    power = LombScargle(time, residual).power(frequency)
    if not np.any(np.isfinite(power)):
        return float("nan")
    return float(1.0 / frequency[int(np.nanargmax(power))])


def bayesian_information_criterion(
    residual: NDArray[np.float64], weights: NDArray[np.float64], n_params: int
) -> float:
    """BIC over the inlier set, used to compare rotation-period candidates.

    The candidates differ in how many Fourier terms they carry, so a plain
    residual comparison would always prefer the richest model.  BIC makes it
    pay for the extra parameters.
    """
    inlier = weights > 0.5
    n = int(inlier.sum())
    if n <= n_params + 1:
        return float("inf")
    rss = float(np.sum(residual[inlier] ** 2))
    if rss <= 0:
        return float("-inf")
    return n * np.log(rss / n) + n_params * np.log(n)


def release_starved(
    design: NDArray[np.float64] | SegmentedDesign,
    exclude: NDArray[np.bool_],
    min_support: float,
) -> NDArray[np.bool_]:
    """Unmask the cadences of any basis function a mask leaves almost unconstrained.

    The cubic B-spline at a segment edge is concentrated on the first knot
    interval: ``(1 - t)**3`` keeps under 1% of its squared weight beyond the
    interval's midpoint.  A mask over the start of a segment can therefore
    leave that coefficient fixed by a few cadences where its basis function is
    nearly zero, and the trend inside the mask becomes an unconstrained
    extrapolation.  Any column keeping less than ``min_support`` of its squared
    weight outside the mask gets its cadences back, so there the fit is the
    blind one again rather than an arbitrary one.
    """
    if isinstance(design, SegmentedDesign):
        mass = design.column_mass(np.ones(design.n_rows, dtype=bool))
        kept = design.column_mass(~exclude)
    else:
        mass = np.sum(design**2, axis=0)
        kept = np.sum(design[~exclude] ** 2, axis=0)
    starved = kept < min_support * mass
    if not starved.any():
        return exclude
    if isinstance(design, SegmentedDesign):
        return exclude & ~design.rows_using(starved)
    return exclude & ~np.any(design[:, starved] != 0.0, axis=1)


def fit_trend(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    config: PreprocessConfig,
    *,
    exclude: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], list[float]]:
    """Fit ``spline + rotation harmonics`` robustly.  Returns ``(trend, periods)``.

    The rotation term is added only when it earns its place:

    1. Fit the spline alone; take the Lomb-Scargle peak of its residual.
    2. Resolve that period's harmonic aliases.  A spotted star is not a
       sinusoid, and with two spot groups on opposite hemispheres the *tallest*
       periodogram peak sits at half the rotation period.  Modelling the star
       there removes the even harmonics and leaves the fundamental standing --
       which is exactly what a transit search then locks onto.  Candidates
       ``P, 2P, 3P`` are compared at matched bandwidth (``m`` times as many
       harmonics for an ``m`` times longer period) and chosen by BIC.
    3. Refit spline and harmonics together, keeping the result only if the
       robust residual scale improves by ``rotation_min_improvement``.

    Step 3 is what protects a quiet star: with the transit down-weighted out of
    the fit there is nothing coherent left, the harmonics buy no improvement,
    and the trend stays spline-only.

    ``exclude`` marks cadences to keep out of every fit (see :func:`flatten`).
    The trend is still evaluated there.
    """
    design = SegmentedDesign.spline(time, config)
    if exclude is not None:
        exclude = release_starved(design, exclude, config.mask_min_support)
        if not exclude.any():
            exclude = None
    fitted = slice(None) if exclude is None else ~exclude
    coefficients, _ = robust_least_squares(
        design,
        signal,
        iterations=config.irls_iterations,
        tuning=config.biweight_tuning,
        exclude=exclude,
    )
    trend = design @ coefficients
    scale = robust_sigma((signal - trend)[fitted])
    periods: list[float] = []

    for _ in range(config.rotation_max_terms):
        peak = dominant_period(time[fitted], winsorize((signal - trend)[fitted]), config)
        if not np.isfinite(peak) or any(abs(peak / p - 1.0) < 0.05 for p in periods):
            break

        best = None
        for multiple in (1, 2, 3):
            candidate = peak * multiple
            if candidate > 0.5 * float(time[-1] - time[0]):
                continue
            trial = design.with_columns(
                harmonic_basis(time, candidate, config.rotation_harmonics * multiple)
            )
            # Cheap scan: the candidates only need ranking, and three IRLS
            # cycles are enough for the weights to settle on the transit.
            trial_coefficients, trial_weights = robust_least_squares(
                trial, signal, iterations=3, tuning=config.biweight_tuning, exclude=exclude
            )
            trial_trend = trial @ trial_coefficients
            criterion = bayesian_information_criterion(
                signal - trial_trend, trial_weights, trial.shape[1]
            )
            if best is None or criterion < best[3]:
                best = (candidate, trial, trial_trend, criterion)

        if best is None:
            break
        candidate, trial_design, _, _ = best
        # Refit the winner properly before deciding whether to keep it.
        trial_coefficients, _ = robust_least_squares(
            trial_design,
            signal,
            iterations=config.irls_iterations,
            tuning=config.biweight_tuning,
            exclude=exclude,
        )
        trial_trend = trial_design @ trial_coefficients
        trial_scale = robust_sigma((signal - trial_trend)[fitted])
        if trial_scale > (1.0 - config.rotation_min_improvement) * scale:
            break  # the rotation term does not pay for itself

        design, trend, scale = trial_design, trial_trend, trial_scale
        periods.append(float(candidate))

    return trend, periods


def running_median_trend(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    window_days: float,
    gap_threshold_days: float = 0.25,
) -> NDArray[np.float64]:
    """Gap-aware running-median baseline.

    **This is the rejected alternative, kept for the test that rejects it.**
    It is not used by the pipeline.  See the module docstring and
    ``tests/test_preprocess.py::test_running_median_eats_the_transit_on_a_steep_star``:
    on a star whose trend moves by more than the photometric noise across one
    window, the window median degenerates to the window's centre value and the
    filter reproduces the data -- transit included.
    """
    trend = np.empty_like(flux)
    cadence = float(np.median(np.diff(time))) if time.size > 1 else 1.0
    if not np.isfinite(cadence) or cadence <= 0:
        cadence = 1.0
    size = max(int(round(window_days / cadence)), 3)
    size += 1 - size % 2
    for segment in split_on_gaps(time, gap_threshold_days):
        chunk = flux[segment]
        trend[segment] = (
            np.median(chunk)
            if chunk.size <= size
            else median_filter(chunk, size=size, mode="nearest")
        )
    return trend


# --------------------------------------------------------------------------
def flatten(
    lc: LightCurve,
    config: PreprocessConfig | None = None,
    *,
    exclude: NDArray[np.bool_] | None = None,
) -> FlattenedLightCurve:
    """Detrend a light curve.  See the module docstring for the reasoning.

    ``exclude`` (aligned with ``lc.time``) marks cadences to keep out of the
    trend fit altogether, such as the transits of a signal already found.  The
    trend is interpolated across them instead of being fitted to them, which
    the robust weights alone cannot guarantee at the edge of a segment (see
    :func:`transitml.features.flatten_masked`).

    Raises
    ------
    ValueError
        If the curve is too short, un-normalisable, or loses too many cadences
        to clipping to be worth searching.
    """
    config = config or PreprocessConfig()
    if exclude is not None:
        exclude = np.asarray(exclude, dtype=bool)
        if exclude.shape != lc.time.shape:
            raise ValueError(f"{lc.target_id}: exclude must match the cadences")
        exclude = exclude[np.isfinite(lc.time) & np.isfinite(lc.flux) & np.isfinite(lc.flux_err)]
    lc = lc.finite()
    if lc.n_cadences < 64:
        raise ValueError(f"{lc.target_id}: too few finite cadences ({lc.n_cadences})")

    median_flux = float(np.median(lc.flux))
    if median_flux == 0.0:
        raise ValueError(f"{lc.target_id}: median flux is zero; cannot normalise")

    time = lc.time
    signal = lc.flux / median_flux - 1.0
    flux_err = lc.flux_err / median_flux

    # Flares and cosmic rays are clipped against the robust trend, measured in
    # point-to-point sigma so that the peaks of a variable star are not
    # mistaken for outliers.  Upward only, always.
    noise = point_to_point_sigma(signal)
    keep = np.ones(time.size, dtype=bool)
    n_clipped = 0
    trend, periods = fit_trend(time, signal, config, exclude=exclude)

    if np.isfinite(noise) and noise > 0:
        for _ in range(max(config.clip_iterations, 1)):
            outlier = (signal[keep] - trend) > config.upper_clip_sigma * noise
            if not outlier.any():
                break
            n_clipped += int(outlier.sum())
            keep[np.flatnonzero(keep)[outlier]] = False
            if int(keep.sum()) < 64:
                raise ValueError(f"{lc.target_id}: too few cadences survive clipping")
            trend, periods = fit_trend(
                time[keep],
                signal[keep],
                config,
                exclude=None if exclude is None else exclude[keep],
            )

    time, signal, flux_err = time[keep], signal[keep], flux_err[keep]
    flat = signal - trend + 1.0
    teff, density = lc.star

    return FlattenedLightCurve(
        target_id=lc.target_id,
        time=time,
        flux=flat,
        flux_err=flux_err,
        trend=trend + 1.0,
        scatter=robust_sigma(flat),
        n_clipped=n_clipped,
        label=lc.label,
        rotation_periods=tuple(periods),
        n_masked=0 if exclude is None else int(exclude[keep].sum()),
        teff_k=teff,
        density_cgs=density,
    )
