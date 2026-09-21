"""Source-agnostic light-curve container and the interface every source implements.

The whole point of this module is that nothing downstream of it knows or cares
whether a light curve was simulated or downloaded from MAST.  ``preprocess``,
``features``, ``model`` and ``evaluate`` only ever see a :class:`LightCurve`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class LightCurve:
    """A single-sector photometric time series.

    Attributes
    ----------
    target_id:
        Opaque identifier.  For synthetic data ``"SYN-000123"``; for MAST data
        this would be ``"TIC 307210830"``.
    time:
        Barycentric time in days.  Strictly increasing, gaps allowed.
    flux:
        Relative flux, normalised so the out-of-transit level is ~1.0.
    flux_err:
        Per-cadence 1-sigma uncertainty on ``flux``.
    label:
        1 if the star hosts a transiting planet, 0 otherwise.  ``None`` for
        unlabelled survey data at inference time.
    meta:
        Free-form provenance/ground truth.  Synthetic curves record the injected
        parameters here so the evaluation can slice recall by true transit SNR;
        real curves would record sector, camera, crowding, and so on.
    """

    target_id: str
    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    flux_err: NDArray[np.float64]
    label: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.time.size
        if not (self.flux.size == n and self.flux_err.size == n):
            raise ValueError(
                f"{self.target_id}: time/flux/flux_err length mismatch "
                f"({n}/{self.flux.size}/{self.flux_err.size})"
            )
        if n and np.any(np.diff(self.time) <= 0):
            raise ValueError(f"{self.target_id}: time array must be strictly increasing")

    @property
    def n_cadences(self) -> int:
        return int(self.time.size)

    @property
    def baseline_days(self) -> float:
        """Total observing span, including gaps."""
        return float(self.time[-1] - self.time[0]) if self.time.size else 0.0

    def finite(self) -> "LightCurve":
        """Return a copy with non-finite cadences removed."""
        good = np.isfinite(self.time) & np.isfinite(self.flux) & np.isfinite(self.flux_err)
        return LightCurve(
            target_id=self.target_id,
            time=self.time[good],
            flux=self.flux[good],
            flux_err=self.flux_err[good],
            label=self.label,
            meta=dict(self.meta),
        )


class LightCurveSource(abc.ABC):
    """Abstract provider of light curves.

    Implementations must yield :class:`LightCurve` objects that are already
    normalised to a relative-flux scale.  They must *not* detrend: removing
    stellar variability is the pipeline's job and is deliberately kept out of
    the data layer so that the same detrending is applied to every source.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of light curves this source will yield."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[LightCurve]:
        """Yield light curves one at a time (streaming; do not materialise all)."""

    @property
    def name(self) -> str:
        return type(self).__name__
