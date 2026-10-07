"""Phase-folded views of a known signal: the input a shape-learning model needs.

The gradient-boosted model elsewhere in this project sees 23 scalars.  A
convolutional model sees the transit itself, binned onto a fixed grid, and the
format here follows AstroNet (Shallue & Vanderburg 2018) so results can be
compared with it:

* **global view**: the whole orbit folded at the period, ``2001`` bins.  Shows
  secondary eclipses, phase curves and anything else away from the transit.
* **local view**: ``+-4`` transit durations around the transit, ``201``
  overlapping bins each ``0.16`` durations wide.  Shows the shape: a U-shaped
  planet against a V-shaped grazing binary.

and adds three more local views, as ExoMiner (Valizadegan et al. 2022) does,
because each carries a Robovetter test in picture form:

* **odd** and **even**: the local view of odd- and even-numbered transits
  alone.  An eclipsing binary at twice the period shows two different depths.
* **secondary**: the local view half an orbit later, where a binary's
  secondary eclipse sits.

All five views share one scale: the median out-of-transit level is subtracted
and the result divided by the local view's depth, so the transit bottoms out
at about -1 in every view where it appears, and an even view at -0.5 means
the even transits are half as deep.  The scale itself is returned beside the
views, so the absolute depth is not lost.

Detrending
----------
The ephemeris is known, so the in-transit cadences of every signal on the
star are masked out of the trend fit instead of being fought off by robust
weights.  Each gap-free segment gets a cubic spline whose knots are spaced
well beyond the longest masked window, so no basis function sits entirely
inside a mask, fitted with the same Tukey biweight as
:mod:`transitml.preprocess` on the unmasked cadences.  There is no rotation
term: PDC flux already removes most of what it would catch on Kepler stars,
and a term fitted across a 4-year baseline is not worth the cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import PreprocessConfig
from .preprocess import robust_least_squares, robust_sigma, spline_basis, split_on_gaps

#: View names, in the order they are stored.
VIEW_NAMES: tuple[str, ...] = ("global", "local", "odd", "even", "secondary")


@dataclass(frozen=True)
class ViewConfig:
    """Binning and detrending parameters.  Defaults follow AstroNet."""

    global_bins: int = 2001
    local_bins: int = 201
    #: Half-width of the local view, in transit durations.
    local_durations: float = 4.0
    #: Width of each local bin, in transit durations (bins overlap).
    local_bin_width: float = 0.16
    #: In-transit mask half-width, in transit durations: 1.5 masks from 1.5
    #: durations before mid-transit to 1.5 after, which covers ingress and
    #: egress even with an ephemeris off by a fraction of a duration.
    mask_half_width: float = 1.5
    #: Spline knots are at least this far apart (days) and at least
    #: ``knot_mask_factor`` times the widest mask on the star.
    min_knot_spacing_days: float = 1.5
    knot_mask_factor: float = 2.0
    #: Split the series at gaps wider than this (Kepler's monthly downlinks
    #: and quarterly rolls).
    gap_threshold_days: float = 0.5


@dataclass(frozen=True)
class Ephemeris:
    """Period, epoch and duration of one signal, all in days."""

    period: float
    epoch: float
    duration: float


@dataclass
class Views:
    """The five views of one signal, plus what was needed to make them."""

    global_view: NDArray[np.float32]
    local: NDArray[np.float32]
    odd: NDArray[np.float32]
    even: NDArray[np.float32]
    secondary: NDArray[np.float32]
    #: Depth the views were divided by (relative flux), and the out-of-transit
    #: robust scatter of the detrended curve on the same scale before division.
    depth_scale: float
    scatter: float
    #: Fraction of local-view bins that held no cadence and were interpolated.
    local_empty_fraction: float

    def as_dict(self) -> dict[str, NDArray[np.float32]]:
        return {
            "global": self.global_view,
            "local": self.local,
            "odd": self.odd,
            "even": self.even,
            "secondary": self.secondary,
        }


def in_transit_mask(
    time: NDArray[np.float64], ephemerides: Sequence[Ephemeris], half_width: float
) -> NDArray[np.bool_]:
    """Cadences within ``half_width`` durations of any transit of any signal."""
    mask = np.zeros(time.size, dtype=bool)
    for eph in ephemerides:
        if not (eph.period > 0 and eph.duration > 0):
            continue
        phase = (time - eph.epoch + 0.5 * eph.period) % eph.period - 0.5 * eph.period
        mask |= np.abs(phase) < half_width * eph.duration
    return mask


def detrend_masked(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    mask: NDArray[np.bool_],
    knot_spacing_days: float,
    gap_threshold_days: float = 0.5,
) -> NDArray[np.float64]:
    """Divide out a robust spline fitted to the unmasked cadences, per segment.

    Returns relative flux centred on 1.  A segment with too few unmasked
    cadences for a spline is divided by its unmasked median instead (or the
    median of everything, if all of it is masked).
    """
    config = PreprocessConfig(
        knot_spacing_days=knot_spacing_days, gap_threshold_days=gap_threshold_days
    )
    trend = np.ones_like(flux)
    for segment in split_on_gaps(time, gap_threshold_days):
        t, f, m = time[segment], flux[segment], mask[segment]
        free = ~m
        fallback = float(np.median(f[free])) if free.any() else float(np.median(f))
        trend[segment] = fallback
        if int(free.sum()) < 16:
            continue
        design = spline_basis(t, config)
        coefficients, _ = robust_least_squares(design[free], f[free] - fallback)
        trend[segment] = fallback + design @ coefficients
    bad = ~np.isfinite(trend) | (trend <= 0)
    trend[bad] = 1.0
    return flux / trend


def _binned_median(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lo: float,
    hi: float,
    n_bins: int,
    width: float,
) -> tuple[NDArray[np.float64], int]:
    """Median of ``y`` in ``n_bins`` windows of ``width`` centred evenly on ``[lo, hi]``.

    ``x`` must be sorted.  Empty windows are filled by linear interpolation
    between their neighbours (or the nearest filled bin at the ends).
    Returns ``(values, n_empty)``.
    """
    spacing = (hi - lo) / n_bins
    centres = lo + spacing * (np.arange(n_bins) + 0.5)
    left = np.searchsorted(x, centres - 0.5 * width, side="left")
    right = np.searchsorted(x, centres + 0.5 * width, side="right")
    values = np.full(n_bins, np.nan)
    for i in range(n_bins):
        if right[i] > left[i]:
            values[i] = np.median(y[left[i] : right[i]])
    filled = np.isfinite(values)
    n_empty = int(n_bins - filled.sum())
    if not filled.any():
        return np.zeros(n_bins), n_empty
    if n_empty:
        values[~filled] = np.interp(centres[~filled], centres[filled], values[filled])
    return values, n_empty


def _fold(time: NDArray[np.float64], period: float, epoch: float) -> NDArray[np.float64]:
    """Phase in days, in ``[-period/2, period/2)``, transit at zero."""
    return (time - epoch + 0.5 * period) % period - 0.5 * period


def make_views(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    ephemeris: Ephemeris,
    config: ViewConfig | None = None,
) -> Views:
    """Global, local, odd, even and secondary views of a detrended curve.

    ``flux`` is relative flux centred on 1 (the output of :func:`detrend_masked`).

    Raises
    ------
    ValueError
        If the ephemeris is unusable or no cadence falls near the transit.
    """
    config = config or ViewConfig()
    period, epoch, duration = ephemeris.period, ephemeris.epoch, ephemeris.duration
    if not (np.isfinite(period) and period > 0 and np.isfinite(duration) and duration > 0):
        raise ValueError(f"unusable ephemeris {ephemeris}")
    signal = flux - 1.0

    phase = _fold(time, period, epoch)
    order = np.argsort(phase, kind="stable")
    phase_sorted, signal_sorted = phase[order], signal[order]

    half = min(config.local_durations * duration, 0.5 * period)
    width = config.local_bin_width * duration
    near = np.abs(phase) < half
    if not near.any():
        raise ValueError("no cadences within the local window")

    out_of_transit = np.abs(phase) > config.mask_half_width * duration
    baseline = float(np.median(signal[out_of_transit])) if out_of_transit.any() else 0.0
    scatter = robust_sigma(signal[out_of_transit]) if out_of_transit.any() else float("nan")

    local, n_empty = _binned_median(
        phase_sorted, signal_sorted, -half, half, config.local_bins, width
    )
    local -= baseline
    depth = float(-local.min())
    if not np.isfinite(depth) or depth <= 0:
        # No dip at all: keep the views on the noise scale instead of
        # dividing by zero or flipping the sign.
        depth = scatter if np.isfinite(scatter) and scatter > 0 else 1.0

    global_view, _ = _binned_median(
        phase_sorted,
        signal_sorted,
        -0.5 * period,
        0.5 * period,
        config.global_bins,
        period / config.global_bins,
    )

    number = np.round((time - epoch) / period).astype(np.int64)
    parity = []
    for keep in (number % 2 == 1, number % 2 == 0):
        p, s = phase[keep], signal[keep]
        o = np.argsort(p, kind="stable")
        if p.size and (np.abs(p) < half).any():
            view, _ = _binned_median(p[o], s[o], -half, half, config.local_bins, width)
        else:
            view = np.full(config.local_bins, baseline)
        parity.append(view)

    shifted = _fold(time, period, epoch + 0.5 * period)
    o = np.argsort(shifted, kind="stable")
    secondary, _ = _binned_median(
        shifted[o], signal[o], -half, half, config.local_bins, width
    )

    def scale(view: NDArray[np.float64], subtract: float) -> NDArray[np.float32]:
        return ((view - subtract) / depth).astype(np.float32)

    return Views(
        global_view=scale(global_view, baseline),
        local=scale(local, 0.0),
        odd=scale(parity[0], baseline),
        even=scale(parity[1], baseline),
        secondary=scale(secondary, baseline),
        depth_scale=depth,
        scatter=float(scatter),
        local_empty_fraction=n_empty / config.local_bins,
    )


def knot_spacing_for(ephemerides: Sequence[Ephemeris], config: ViewConfig) -> float:
    """Knot spacing wide enough that no spline basis function hides in a mask."""
    widest = max(
        (2.0 * config.mask_half_width * e.duration for e in ephemerides if e.duration > 0),
        default=0.0,
    )
    return max(config.min_knot_spacing_days, config.knot_mask_factor * widest)
