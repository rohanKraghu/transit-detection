"""Synthetic target pixel files: Gaussian stars on a pixel grid, with eclipses.

Used to test the centroid analysis in :mod:`transitml.centroid` against known
truth.  The scene is a few point sources, each a circular Gaussian integrated
over square pixels, plus sky.  Any star may dip on a shared box ephemeris, so
one scene can hold a planet transiting the target or an eclipsing binary on a
neighbour a pixel or two away whose light leaks into the target's aperture: a
blend, the false positive centroid tests exist to catch.

Noise is per pixel and per cadence: photon noise on the star and sky counts
plus a read/sky floor, and optionally a pointing jitter that moves every star
together by a random fraction of a pixel each cadence.  Fluxes are electrons
per second and are returned per second, as SPOC target pixel files are, with
the sky already subtracted.

This is a test fixture, not a model of the TESS PRF, which is broader, not
circular, and varies across the field.  A flux-weighted centroid does not
depend on the PSF's shape for a symmetric PSF, which is what makes a Gaussian
an adequate stand-in for these tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from .tpf import TargetPixelData

#: Electrons per second from a T = 10 star in TESS (order of magnitude; Vanderspek et al. 2018).
TESS_ZERO_POINT_E_PER_S = 15_000.0


def tess_flux(tess_mag: float) -> float:
    """Electrons per second for a star of TESS magnitude ``tess_mag``."""
    return TESS_ZERO_POINT_E_PER_S * 10.0 ** (-0.4 * (tess_mag - 10.0))


@dataclass(frozen=True)
class PixelStar:
    """One point source in a synthetic stamp.

    ``column`` and ``row`` are stamp pixel coordinates (``(0, 0)`` is the
    centre of the first pixel).  ``eclipse_depth`` is the fractional dip of
    *this star's own* flux during each event on the scene's ephemeris; 0 means
    the star is constant.
    """

    column: float
    row: float
    flux: float
    eclipse_depth: float = 0.0


def gaussian_psf_image(
    shape: tuple[int, int], column: float, row: float, sigma: float
) -> NDArray[np.float64]:
    """A unit-flux circular Gaussian integrated over each pixel of a ``(rows, cols)`` stamp."""

    def axis_weights(n: int, centre: float) -> NDArray[np.float64]:
        edges = np.arange(n + 1, dtype=np.float64) - 0.5
        cdf = 0.5 * (1.0 + erf((edges - centre) / (np.sqrt(2.0) * sigma)))
        return np.diff(cdf)

    return np.outer(axis_weights(shape[0], row), axis_weights(shape[1], column))


def box_in_eclipse(
    time: NDArray[np.float64], period: float, epoch: float, duration: float
) -> NDArray[np.bool_]:
    """True for cadences inside a box event of the given ephemeris."""
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    return np.abs(phase) < 0.5 * duration


def synthetic_tpf(
    stars: Sequence[PixelStar],
    *,
    period: float,
    epoch: float,
    duration: float,
    shape: tuple[int, int] = (11, 11),
    baseline: float = 27.0,
    cadence_minutes: float = 30.0,
    psf_sigma: float = 0.75,
    sky_e_per_s: float = 200.0,
    read_noise_e: float = 10.0,
    jitter_pixels: float = 0.0,
    aperture_fraction: float = 0.08,
    seed: int = 0,
    target_id: str = "SYN-TPF",
) -> TargetPixelData:
    """Simulate a target pixel file.  ``stars[0]`` is the target.

    The aperture is every pixel whose noise-free mean flux exceeds
    ``aperture_fraction`` of the brightest pixel, which, like a real pipeline
    aperture, also takes in light from a close neighbour.  The target's true
    position is stored as ``target_position``, as the catalogue position would
    be for a real file.
    """
    if not stars:
        raise ValueError("a scene needs at least one star (the target)")
    rng = np.random.default_rng(seed)
    exposure = cadence_minutes * 60.0
    time = np.arange(0.0, baseline, cadence_minutes / 1440.0)
    n_t = time.size
    in_event = box_in_eclipse(time, period, epoch, duration)
    jitter = (
        rng.normal(0.0, jitter_pixels, size=(n_t, 2)) if jitter_pixels > 0 else None
    )

    model = np.zeros((n_t, *shape))
    for star in stars:
        brightness = np.full(n_t, star.flux)
        brightness[in_event] *= 1.0 - star.eclipse_depth
        if jitter is None:
            model += brightness[:, None, None] * gaussian_psf_image(
                shape, star.column, star.row, psf_sigma
            )
            continue
        for i in range(n_t):
            model[i] += brightness[i] * gaussian_psf_image(
                shape, star.column + jitter[i, 0], star.row + jitter[i, 1], psf_sigma
            )

    variance_e = (model + sky_e_per_s) * exposure + read_noise_e**2
    sigma = np.sqrt(variance_e) / exposure
    flux = model + rng.normal(0.0, 1.0, size=model.shape) * sigma

    mean_image = model.mean(axis=0)
    aperture = mean_image > aperture_fraction * mean_image.max()
    target = stars[0]
    return TargetPixelData(
        target_id=target_id,
        time=time,
        flux=flux,
        flux_err=sigma,
        aperture=aperture,
        target_position=(target.column, target.row),
        meta={
            "kind": "synthetic",
            "period": period,
            "epoch": epoch,
            "duration": duration,
            "stars": [
                {
                    "column": s.column,
                    "row": s.row,
                    "flux": s.flux,
                    "eclipse_depth": s.eclipse_depth,
                }
                for s in stars
            ],
        },
    )


Scenario = Literal["on_target", "blend", "none"]


def blend_scenario(
    kind: Scenario,
    *,
    neighbour_offset: tuple[float, float] = (1.6, 0.8),
    target_mag: float = 10.0,
    neighbour_delta_mag: float = 2.0,
    transit_depth: float = 0.005,
    binary_depth: float = 0.06,
    period: float = 3.1,
    epoch: float = 1.4,
    duration: float = 0.15,
    jitter_pixels: float = 0.005,
    seed: int = 0,
    **kwargs,
) -> tuple[TargetPixelData, dict]:
    """A target with one fainter neighbour, and an event on one of them (or neither).

    ``"on_target"``: the target is transited (``transit_depth`` of its flux).
    ``"blend"``: the neighbour, ``neighbour_delta_mag`` fainter and
    ``neighbour_offset`` pixels away, is an eclipsing binary
    (``binary_depth`` of its own flux), diluted in the aperture to a
    planet-like dip.  ``"none"``: nothing dips.  Returns the file and the
    truth: ephemeris and both positions.
    """
    shape = kwargs.pop("shape", (11, 11))
    centre = ((shape[1] - 1) / 2.0 + 0.1, (shape[0] - 1) / 2.0 - 0.2)
    neighbour_pos = (centre[0] + neighbour_offset[0], centre[1] + neighbour_offset[1])
    target_flux = tess_flux(target_mag)
    neighbour_flux = tess_flux(target_mag + neighbour_delta_mag)
    stars = [
        PixelStar(*centre, target_flux, transit_depth if kind == "on_target" else 0.0),
        PixelStar(
            *neighbour_pos, neighbour_flux, binary_depth if kind == "blend" else 0.0
        ),
    ]
    tpf = synthetic_tpf(
        stars,
        period=period,
        epoch=epoch,
        duration=duration,
        shape=shape,
        jitter_pixels=jitter_pixels,
        seed=seed,
        target_id=f"SYN-TPF-{kind}",
        **kwargs,
    )
    truth = {
        "period": period,
        "epoch": epoch,
        "duration": duration,
        "target_position": centre,
        "neighbour_position": neighbour_pos,
        "kind": kind,
    }
    return tpf, truth
