"""Single and duo transits: dips that happen once or twice in the window.

BLS folds the light curve on trial periods, so it needs at least two transits
and its grid stops at half the baseline.  A planet on a 40-day orbit shows
one transit in a 27-day sector, or none, and is excluded by construction.
Those planets are worth finding anyway: they are the long-period, temperate
ones, and a second sector or a radial-velocity campaign can pin the period
down once the event is known.

This module searches for individual box-shaped dips instead:

1. **Box scan.**  For each trial duration, every cadence is tried as the
   centre of a box.  The depth is the local baseline (the cadences on either
   side, one duration wide) minus the in-box mean, so slow residual trends do
   not masquerade as depth.  Its error uses the light curve's own scatter
   *binned at that duration*, which is what red noise inflates; a box that is
   less than ``min_coverage`` observed is skipped, so a dip that is half in a
   gap does not count.
2. **Events.**  The highest-SNR box is an event if its SNR reaches
   ``min_snr``; every box overlapping it is then removed and the next highest
   is taken, up to ``max_events``.  A dip with one sharp edge and an
   exponential recovery on the other side (the shape of an instrumental
   ramp after a momentum dump) is fitted both ways, and if the ramp fits
   better than any box by ``ramp_delta_chi2`` it is set aside as a ramp
   rather than reported.
3. **Periods.**  For a single event the period is unknown, but not
   unconstrained: any period that would put another transit on observed data
   that shows no such dip is ruled out, which gives a minimum period, and the
   duration gives a rough period for a central transit across a star of the
   assumed density.  Two events of matching depth and duration form a duo,
   whose period must be their separation divided by a whole number; each
   such alias is kept only if none of the transits it predicts lands on
   observed, flat data.

Only the vetting tool uses this.  The classifier still sees the strongest BLS
signal and the headline numbers do not depend on anything here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import SingleEventConfig
from .physics import G_CGS, SECONDS_PER_DAY, expected_central_duration
from .preprocess import FlattenedLightCurve, robust_sigma


@dataclass(frozen=True)
class SingleEvent:
    """One box-shaped dip found by :func:`single_event_search`."""

    #: Mid-event time, on the light curve's time axis.
    time: float
    duration: float
    depth: float
    depth_err: float
    snr: float
    #: Cadences inside the box.
    n_cadences: int
    #: Within one duration of a data gap or the end of the light curve, where
    #: scattered light, momentum dumps and detrending edge effects live.
    near_gap: bool
    #: Period of a central transit with this duration across a star of the
    #: assumed density (``P = pi^2 G rho T^3 / 3``).  A grazing transit is
    #: shorter, so the true period is more often longer than this.
    period_estimate: float
    #: Shortest period that puts no other transit on observed, flat data.
    min_period: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in asdict(self).items()}


@dataclass(frozen=True)
class DuoCandidate:
    """Two events that could be successive transits of one planet."""

    first: SingleEvent
    second: SingleEvent
    #: ``(second.time - first.time) / n`` for every ``n`` not ruled out by the data.
    allowed_periods: tuple[float, ...]
    #: Each allowed period's measured duration over the central-transit duration
    #: at the assumed density; far above 1 means the event is too long for it.
    duration_ratios: tuple[float, ...]

    @property
    def separation(self) -> float:
        return self.second.time - self.first.time

    def to_dict(self) -> dict[str, Any]:
        return {
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "separation": float(self.separation),
            "allowed_periods": [float(p) for p in self.allowed_periods],
            "duration_ratios": [float(r) for r in self.duration_ratios],
        }


@dataclass(frozen=True)
class SingleEventSearch:
    """Everything :func:`search_single_events` found in one light curve."""

    events: tuple[SingleEvent, ...]
    duos: tuple[DuoCandidate, ...]
    #: Mid-times of dips that cleared ``min_snr`` but are ramp-shaped, so
    #: they were not reported as events.
    ramps: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "duos": [d.to_dict() for d in self.duos],
            "ramps": [float(t) for t in self.ramps],
        }


# --------------------------------------------------------------------------
# The box statistic
# --------------------------------------------------------------------------
def _cadence(time: NDArray[np.float64]) -> float:
    return float(np.median(np.diff(time))) if time.size > 1 else 1.0


def binned_noise_factor(lc: FlattenedLightCurve, width: float) -> float:
    """How much more the light curve scatters at timescale ``width`` than white noise would.

    The robust scatter of bin means against ``scatter / sqrt(n)``, floored at
    1.  The robust estimator matters here: one deep event in a few bins must
    not raise the noise it is then judged against.
    """
    cadence = _cadence(lc.time)
    width = max(width, 3.0 * cadence)
    sigma = lc.scatter if lc.scatter > 0 else robust_sigma(lc.flux)
    edges = np.arange(lc.time[0], lc.time[-1] + width, width)
    which = np.digitize(lc.time, edges)
    counts = np.bincount(which)
    sums = np.bincount(which, weights=lc.flux)
    full = counts >= max(2, 0.5 * width / cadence)
    if full.sum() < 8 or not np.isfinite(sigma) or sigma <= 0:
        return 1.0
    means = sums[full] / counts[full]
    expected = sigma / np.sqrt(np.mean(counts[full]))
    return float(max(1.0, robust_sigma(means) / expected))


class _BoxStats:
    """Cumulative sums for O(1) box means anywhere on one light curve."""

    def __init__(self, lc: FlattenedLightCurve) -> None:
        self.time = lc.time
        self.cadence = _cadence(lc.time)
        self.sum = np.concatenate([[0.0], np.cumsum(lc.flux)])
        self.scatter = lc.scatter if lc.scatter > 0 else robust_sigma(lc.flux)

    def window(self, lo: NDArray, hi: NDArray) -> tuple[NDArray, NDArray]:
        """Count and sum of cadences with ``lo <= t < hi``."""
        a = np.searchsorted(self.time, lo, side="left")
        b = np.searchsorted(self.time, hi, side="left")
        return (b - a).astype(float), self.sum[b] - self.sum[a]

    def depth(
        self,
        centres: NDArray,
        duration: float,
        beta: float,
        min_coverage: float,
        two_sided: bool = True,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Depth, its error and in-box count of a box at each centre.

        Boxes less than ``min_coverage`` observed come back with a NaN depth,
        and so do boxes without a baseline on both sides unless ``two_sided``
        is off.  Detection insists on both sides, so a ramp into a gap is not
        an event; checking whether a predicted transit is absent does not, so
        a predicted transit just before a gap can still be ruled out.
        """
        half = duration / 2.0
        n_in, s_in = self.window(centres - half, centres + half)
        n_lo, s_lo = self.window(centres - half - duration, centres - half)
        n_hi, s_hi = self.window(centres + half, centres + half + duration)
        expected = duration / self.cadence
        need = max(2.0, min_coverage * expected)
        if two_sided:
            ok = (n_in >= need) & (n_lo >= 0.5 * need) & (n_hi >= 0.5 * need)
        else:
            ok = (n_in >= need) & (n_lo + n_hi >= need)
        with np.errstate(invalid="ignore", divide="ignore"):
            n_out = n_lo + n_hi
            depth = (s_lo + s_hi) / n_out - s_in / n_in
            err = self.scatter * beta * np.sqrt(1.0 / n_in + 1.0 / n_out)
        depth = np.where(ok, depth, np.nan)
        return depth, err, n_in


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------
def period_from_duration(duration_days: float, stellar_density_cgs: float) -> float:
    """Period of a central transit lasting ``duration_days`` (small planet).

    From ``T = P / (pi * a/R*)`` and Kepler's third law
    ``a/R* = (G rho P^2 / 3 pi)^(1/3)``: ``P = pi^2 G rho T^3 / 3``.
    """
    t = duration_days * SECONDS_PER_DAY
    return float(np.pi**2 * G_CGS * stellar_density_cgs * t**3 / 3.0 / SECONDS_PER_DAY)


def _ruled_out(
    stats: _BoxStats,
    epochs: NDArray,
    event: SingleEvent,
    beta: float,
    config: SingleEventConfig,
) -> bool:
    """True if any predicted transit at ``epochs`` lands on observed data without the dip."""
    if epochs.size == 0:
        return False
    depth, _, _ = stats.depth(
        epochs, event.duration, beta, config.min_coverage, two_sided=False
    )
    covered = np.isfinite(depth)
    return bool(np.any(depth[covered] < config.duo_veto_fraction * event.depth))


def minimum_period(
    stats: _BoxStats, event: SingleEvent, beta: float, config: SingleEventConfig
) -> float:
    """Shortest period whose other transits all miss observed, flat data."""
    start, end = float(stats.time[0]), float(stats.time[-1])
    longest = max(event.time - start, end - event.time) + event.duration
    step = max(stats.cadence / 4.0, 0.002)
    for period in np.arange(max(2.0 * event.duration, 0.5), longest + step, step):
        before = int((event.time - start) // period) + 1
        k = np.arange(-before, int((end - event.time) // period) + 2)
        k = k[k != 0]
        epochs = event.time + k * period
        epochs = epochs[(epochs > start - event.duration) & (epochs < end + event.duration)]
        if not _ruled_out(stats, epochs, event, beta, config):
            return float(period)
    return float(longest)


def duo_periods(
    stats: _BoxStats,
    first: SingleEvent,
    second: SingleEvent,
    beta: float,
    config: SingleEventConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Aliases ``separation / n`` that no observed, flat stretch rules out."""
    start, end = float(stats.time[0]), float(stats.time[-1])
    separation = second.time - first.time
    probe = SingleEvent(
        time=first.time,
        duration=0.5 * (first.duration + second.duration),
        depth=0.5 * (first.depth + second.depth),
        depth_err=0.0,
        snr=0.0,
        n_cadences=0,
        near_gap=False,
        period_estimate=float("nan"),
        min_period=float("nan"),
    )
    periods, ratios = [], []
    for n in range(1, config.duo_max_harmonic + 1):
        period = separation / n
        if period < 2.0 * probe.duration:
            break
        k = np.arange(int(np.floor((start - first.time) / period)) - 1,
                      int(np.ceil((end - first.time) / period)) + 2)
        k = k[(k != 0) & (k != n)]
        epochs = first.time + k * period
        epochs = epochs[(epochs > start - probe.duration) & (epochs < end + probe.duration)]
        if _ruled_out(stats, epochs, probe, beta, config):
            continue
        periods.append(float(period))
        central = expected_central_duration(period, config.stellar_density_cgs)
        ratios.append(float(probe.duration / central) if central > 0 else float("nan"))
    return tuple(periods), tuple(ratios)


# --------------------------------------------------------------------------
# Ramps
# --------------------------------------------------------------------------
_RAMP_TAUS_DAYS = np.geomspace(0.02, 0.6, 14)


def _best_chi2(y: NDArray, templates: NDArray) -> float:
    """Smallest residual sum of squares of ``y ~ c + a * template`` over the rows."""
    yc = y - y.mean()
    tc = templates - templates.mean(axis=1, keepdims=True)
    norm = np.einsum("ij,ij->i", tc, tc)
    proj = tc @ yc
    with np.errstate(invalid="ignore", divide="ignore"):
        explained = np.where(norm > 0, proj**2 / norm, 0.0)
    return float(yc @ yc - explained.max())


def ramp_preference(lc: FlattenedLightCurve, event: SingleEvent, beta: float) -> float:
    """How much better a ramp fits ``event`` than a box does, in chi-squared.

    Both models are fitted to the data within ``duration / 2 + max(duration,
    0.3 d)`` of the event, each with its own baseline and amplitude.  The box
    has free centre (within half a duration) and width (half to one and a
    half durations).  The ramp is a sharp step at a free time followed by an
    exponential recovery (time constant 0.02 to 0.6 d), or the same reversed
    in time.  The result is ``(chi2_box - chi2_ramp) / (scatter * beta)^2``:
    positive when the ramp is the better description.  A transit has two
    sharp edges and a ramp has one, so on real transits this is strongly
    negative.
    """
    d = event.duration
    near = np.abs(lc.time - event.time) < d / 2.0 + max(d, 0.3)
    t, y = lc.time[near], lc.flux[near]
    if t.size < 8:
        return float("-inf")
    cadence = _cadence(lc.time)
    centres = np.arange(event.time - d / 2.0, event.time + d / 2.0 + cadence, cadence / 2.0)
    widths = d * np.linspace(0.5, 1.5, 11)
    boxes = -(
        np.abs(t[None, None, :] - centres[None, :, None]) < widths[:, None, None] / 2.0
    ).reshape(-1, t.size).astype(float)
    starts = np.arange(event.time - d, event.time + d, cadence / 2.0)
    x = (t[None, None, :] - starts[None, :, None]) / _RAMP_TAUS_DAYS[:, None, None]
    after = -np.where(x >= 0.0, np.exp(-np.clip(x, 0.0, 50.0)), 0.0)
    before = -np.where(x <= 0.0, np.exp(np.clip(x, -50.0, 0.0)), 0.0)
    ramps = np.concatenate([after.reshape(-1, t.size), before.reshape(-1, t.size)])
    sigma2 = (lc.scatter * beta) ** 2
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return float("-inf")
    return (_best_chi2(y, boxes) - _best_chi2(y, ramps)) / sigma2


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------
def _near_gap(time: NDArray, centre: float, duration: float, cadence: float) -> bool:
    reach = 1.5 * duration
    if centre - reach < time[0] or centre + reach > time[-1]:
        return True
    window = time[(time > centre - reach - cadence) & (time < centre + reach + cadence)]
    return bool(window.size < 2 or np.max(np.diff(window)) > 4.0 * cadence)


def search_single_events(
    lc: FlattenedLightCurve, config: SingleEventConfig | None = None
) -> SingleEventSearch:
    """Find up to ``config.max_events`` individual dips, and pair them into duos.

    See the module docstring for the method.  An empty result means no box
    reached ``config.min_snr``.
    """
    config = config or SingleEventConfig()
    if lc.time.size < 20:
        return SingleEventSearch((), ())
    stats = _BoxStats(lc)
    centres = lc.time
    betas = {d: binned_noise_factor(lc, d) for d in config.durations_days}
    table = []
    for duration in config.durations_days:
        depth, err, n_in = stats.depth(centres, duration, betas[duration], config.min_coverage)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(np.isfinite(depth) & (err > 0), depth / err, -np.inf)
        table.append((duration, depth, err, n_in, snr))

    events: list[SingleEvent] = []
    ramps: list[float] = []
    taken: list[tuple[float, float]] = []
    while len(events) < config.max_events and len(taken) < 3 * config.max_events:
        best = None
        for duration, depth, err, n_in, snr in table:
            blocked = np.zeros(centres.size, dtype=bool)
            for t0, d0 in taken:
                blocked |= np.abs(centres - t0) < 0.5 * (d0 + duration)
            score = np.where(blocked, -np.inf, snr)
            i = int(np.argmax(score))
            if np.isfinite(score[i]) and (best is None or score[i] > best[0]):
                best = (float(score[i]), duration, float(depth[i]), float(err[i]),
                        int(n_in[i]), float(centres[i]))
        if best is None or best[0] < config.min_snr:
            break
        best = _refine(lc, stats, best, config)
        snr, duration, depth, err, n_in, centre = best
        event = SingleEvent(
            time=centre,
            duration=duration,
            depth=depth,
            depth_err=err,
            snr=snr,
            n_cadences=n_in,
            near_gap=_near_gap(lc.time, centre, duration, stats.cadence),
            period_estimate=period_from_duration(duration, config.stellar_density_cgs),
            min_period=float("nan"),
        )
        beta = betas.get(duration) or binned_noise_factor(lc, duration)
        taken.append((centre, duration))
        if ramp_preference(lc, event, beta) > config.ramp_delta_chi2:
            ramps.append(centre)
            continue
        events.append(_with_min_period(stats, event, beta, config))

    duos = []
    ordered = sorted(events, key=lambda e: e.time)
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            if not _matching(first, second, config):
                continue
            beta = binned_noise_factor(lc, 0.5 * (first.duration + second.duration))
            periods, ratios = duo_periods(stats, first, second, beta, config)
            if periods:
                duos.append(DuoCandidate(first, second, periods, ratios))
    return SingleEventSearch(tuple(events), tuple(duos), tuple(ramps))


def _refine(
    lc: FlattenedLightCurve,
    stats: _BoxStats,
    best: tuple[float, float, float, float, int, float],
    config: SingleEventConfig,
) -> tuple[float, float, float, float, int, float]:
    """Polish an event's duration and centre on a finer grid around the coarse best.

    The coarse grid is spaced by factors of ~1.5, so a 0.25-day transit is
    first seen as a 0.33-day box and its depth is diluted by a quarter.
    Durations from 0.7 to 1.4 times the coarse one and centres within half a
    duration are tried, and the highest SNR wins.  The longest coarse
    duration stays the ceiling.
    """
    _, duration0, _, _, _, centre0 = best
    longest = max(config.durations_days)
    trial = np.clip(duration0 * np.linspace(0.7, 1.4, 15), 0.02, longest)
    durations = np.unique(np.round(trial, 4))
    near = np.abs(lc.time - centre0) <= 0.5 * duration0
    centres = np.concatenate([lc.time[near], lc.time[near] + 0.5 * stats.cadence])
    out = best
    for duration in durations:
        beta = binned_noise_factor(lc, float(duration))
        depth, err, n_in = stats.depth(centres, float(duration), beta, config.min_coverage)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(np.isfinite(depth) & (err > 0), depth / err, -np.inf)
        i = int(np.argmax(snr))
        if snr[i] > out[0]:
            out = (float(snr[i]), float(duration), float(depth[i]), float(err[i]),
                   int(n_in[i]), float(centres[i]))
    return out


def _with_min_period(
    stats: _BoxStats, event: SingleEvent, beta: float, config: SingleEventConfig
) -> SingleEvent:
    return replace(event, min_period=minimum_period(stats, event, beta, config))


def _matching(a: SingleEvent, b: SingleEvent, config: SingleEventConfig) -> bool:
    """Depths agree within ``duo_depth_sigma`` plus a fractional floor, durations within a factor.

    The floor matters at high SNR, where the depth errors are tiny and the two
    transits of one planet still differ by several per cent through
    detrending and how the cadences sample ingress and egress.
    """
    sigma = float(np.hypot(a.depth_err, b.depth_err))
    allowed = config.duo_depth_sigma * sigma + config.duo_depth_fraction * 0.5 * (a.depth + b.depth)
    depth_ok = abs(a.depth - b.depth) <= max(allowed, 1e-12)
    ratio = max(a.duration, b.duration) / min(a.duration, b.duration)
    return bool(depth_ok and ratio <= config.duo_duration_ratio)


def drop_periodic(
    found: SingleEventSearch,
    lc: FlattenedLightCurve,
    signals: list[tuple[float, float, float]],
    config: SingleEventConfig | None = None,
    min_transit_snr: float = 3.0,
) -> SingleEventSearch:
    """Remove events that are transits of a periodic signal the data support.

    ``signals`` are ``(period, epoch, duration)`` from the periodic search.  A
    signal counts as periodic when at least two of its predicted transits land
    on observed data and each shows a dip of SNR ``min_transit_snr`` or more;
    then every event inside one of its transits is that signal, and is dropped
    along with any duo it belongs to.  A "periodic" signal whose only real dip
    is one event, the rest falling in gaps or on flat data, explains nothing,
    so a genuine single transit is never hidden behind its own BLS alias.
    """
    config = config or SingleEventConfig()
    stats = _BoxStats(lc)
    start, end = float(lc.time[0]), float(lc.time[-1])
    explained: set[float] = set()
    for period, epoch, duration in signals:
        k = np.arange(np.floor((start - epoch) / period), np.ceil((end - epoch) / period) + 1)
        epochs = epoch + k * period
        beta = binned_noise_factor(lc, duration)
        depth, err, _ = stats.depth(epochs, duration, beta, config.min_coverage)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(np.isfinite(depth) & (err > 0), depth / err, -np.inf)
        if np.count_nonzero(snr >= min_transit_snr) < 2:
            continue
        for event in found.events:
            phase = (event.time - epoch + 0.5 * period) % period - 0.5 * period
            if abs(phase) < 0.5 * (duration + event.duration):
                explained.add(event.time)
    events = tuple(e for e in found.events if e.time not in explained)
    duos = tuple(
        d for d in found.duos
        if d.first.time not in explained and d.second.time not in explained
    )
    return SingleEventSearch(events, duos, found.ramps)
