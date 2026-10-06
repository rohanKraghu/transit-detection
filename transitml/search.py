"""Iterative multi-planet search: find a signal, mask it, search again.

The classifier works on one signal per light curve, the strongest BLS peak, and
that stays true.  This module is for looking at a single star in more detail
(the ``vet`` command): after the strongest signal is found, its in-transit
cadences are masked and BLS is run again on what is left, so a second or third
planet with a different period is not hidden behind the first.

The search stops at ``max_signals`` or as soon as the best remaining peak is
not significant: its ``bls_sde`` (the robust periodogram peak significance the
classifier also uses) falls below ``min_sde``.  A peak below the threshold is
never reported, so a light curve with nothing in it yields an empty list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import BLSConfig, MultiPlanetConfig
from .features import run_bls, signal_detection_efficiency
from .preprocess import FlattenedLightCurve


@dataclass(frozen=True)
class CandidateSignal:
    """One periodic box-shaped signal found by the iterative search."""

    #: 1 for the primary (strongest) signal, 2 for the next, and so on.
    rank: int
    period: float
    epoch: float
    duration: float
    depth: float
    depth_snr: float
    sde: float
    #: Cadences left in the light curve when this signal was searched for.
    n_cadences_searched: int

    def to_dict(self) -> dict[str, Any]:
        return {k: (float(v) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def transit_mask(
    time: NDArray[np.float64],
    period: float,
    epoch: float,
    duration: float,
    half_width_durations: float,
) -> NDArray[np.bool_]:
    """True for cadences within ``half_width_durations * duration`` of a transit."""
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    return np.abs(phase) < half_width_durations * duration


def _subset(lc: FlattenedLightCurve, keep: NDArray[np.bool_]) -> FlattenedLightCurve:
    return replace(
        lc,
        time=lc.time[keep],
        flux=lc.flux[keep],
        flux_err=lc.flux_err[keep],
        trend=lc.trend[keep],
    )


def iterative_search(
    lc: FlattenedLightCurve,
    bls: BLSConfig | None = None,
    config: MultiPlanetConfig | None = None,
) -> list[CandidateSignal]:
    """Search a detrended light curve for up to ``config.max_signals`` signals.

    Each pass runs the same BLS search as :func:`transitml.features.run_bls`
    (so the first pass reproduces the primary signal the classifier sees),
    records the peak if its SDE is at least ``config.min_sde``, masks every
    cadence within ``config.mask_half_width_durations`` durations of its
    transits, and repeats.  It stops at the first peak below the threshold,
    when the peak has non-positive depth, or when fewer than
    ``config.min_cadences`` cadences remain.

    The scatter used for the error bars is the full curve's, so masking does
    not change the noise level later passes are judged against.
    """
    bls = bls or BLSConfig()
    config = config or MultiPlanetConfig()
    candidates: list[CandidateSignal] = []
    current = lc
    for rank in range(1, config.max_signals + 1):
        if current.time.size < config.min_cadences:
            break
        result = run_bls(current, bls)
        sde = signal_detection_efficiency(result["power"])
        if not np.isfinite(sde) or sde < config.min_sde or result["depth"] <= 0:
            break
        candidates.append(
            CandidateSignal(
                rank=rank,
                period=result["period"],
                epoch=result["transit_time"],
                duration=result["duration"],
                depth=result["depth"],
                depth_snr=result["depth_snr"],
                sde=sde,
                n_cadences_searched=int(current.time.size),
            )
        )
        masked = transit_mask(
            current.time,
            result["period"],
            result["transit_time"],
            result["duration"],
            config.mask_half_width_durations,
        )
        current = _subset(current, ~masked)
    return candidates
