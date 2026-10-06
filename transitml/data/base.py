"""Source-agnostic light-curve container and the interface every source implements.

The whole point of this module is that nothing downstream of it knows or cares
whether a light curve was simulated or downloaded from MAST.  ``preprocess``,
``features``, ``model`` and ``evaluate`` only ever see a :class:`LightCurve`.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from ..physics import RHO_SUN_CGS, density_from_gravity, main_sequence_density, main_sequence_teff


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
        real curves would record sector, camera, crowding, and so on.  The
        host star's parameters live here too (see :attr:`star`).
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
    def star(self) -> tuple[float, float]:
        """The host's effective temperature (K) and mean density (g/cm^3).

        See :func:`stellar_parameters`.  NaN for whatever is unknown.
        """
        return stellar_parameters(self.meta)

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


def stellar_parameters(meta: Mapping[str, Any]) -> tuple[float, float]:
    """Effective temperature (K) and mean density (g/cm^3) from a curve's metadata.

    The synthetic and injection sources record the star they drew as
    ``rho_star_cgs`` (and ``teff_k``); a real star has catalogue values, the
    temperature as ``teff_k`` and the density as ``rho_star_cgs`` or, failing
    that, from ``m_star_msun`` or ``logg_cgs`` with ``r_star_rsun``.  When only
    the temperature or only the density is known, the other is the
    main-sequence value.  Both are NaN when neither is known.
    """

    def number(key: str) -> float:
        try:
            value = float(meta.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")
        return value if np.isfinite(value) and value > 0 else float("nan")

    teff, density, radius = number("teff_k"), number("rho_star_cgs"), number("r_star_rsun")
    if not np.isfinite(density) and np.isfinite(radius):
        mass, logg = number("m_star_msun"), number("logg_cgs")
        if np.isfinite(mass):
            density = RHO_SUN_CGS * mass / radius**3
        elif np.isfinite(logg):
            density = density_from_gravity(logg, radius)
    if np.isfinite(teff) and not np.isfinite(density):
        density = main_sequence_density(teff)
    elif np.isfinite(density) and not np.isfinite(teff):
        teff = main_sequence_teff(density)
    return teff, density


def stitch_light_curves(curves: Sequence[LightCurve]) -> LightCurve:
    """Join several single-sector curves of one star into one multi-sector curve.

    Each curve is divided by its own median flux (errors scaled alike) before
    joining, so sectors observed through different apertures or with different
    contamination land on the same relative scale.  The time series are
    concatenated in time order and nothing is interpolated: the gaps between
    sectors stay gaps.  Detrending splits on gaps, so no spline segment spans
    one, and the BLS period grid then extends to half the stitched baseline.

    A cadence time that appears in more than one curve is kept once (the first
    occurrence in time order).

    Raises
    ------
    ValueError
        If ``curves`` is empty or the curves belong to different targets.
    """
    if not curves:
        raise ValueError("nothing to stitch")
    targets = {lc.target_id for lc in curves}
    if len(targets) > 1:
        raise ValueError(f"cannot stitch different targets: {sorted(targets)}")

    times, fluxes, errors = [], [], []
    for lc in curves:
        lc = lc.finite()
        if lc.n_cadences == 0:
            continue
        median = float(np.median(lc.flux))
        if median == 0.0:
            raise ValueError(f"{lc.target_id}: a sector has zero median flux")
        times.append(lc.time)
        fluxes.append(lc.flux / median)
        errors.append(lc.flux_err / abs(median))
    if not times:
        raise ValueError(f"{curves[0].target_id}: no finite cadences to stitch")

    time = np.concatenate(times)
    order = np.argsort(time, kind="stable")
    time, flux, flux_err = time[order], np.concatenate(fluxes)[order], np.concatenate(errors)[order]
    keep = np.concatenate([[True], np.diff(time) > 0])

    labels = {lc.label for lc in curves}
    meta = dict(curves[0].meta)
    meta.update(
        {
            "stitched": True,
            "n_sectors": len(times),
            "sectors": [lc.meta.get("sector") for lc in curves],
        }
    )
    return LightCurve(
        target_id=curves[0].target_id,
        time=time[keep],
        flux=flux[keep],
        flux_err=flux_err[keep],
        label=labels.pop() if len(labels) == 1 else None,
        meta=meta,
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
