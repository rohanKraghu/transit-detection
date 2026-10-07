"""Target pixel files: the per-pixel time series behind a light curve.

A light curve is one number per cadence: the flux summed over an aperture.  A
target pixel file (TPF) keeps every pixel of a small stamp around the star, so
it shows *where* on the sky a dip happens.  That is what separates an eclipse
on the target from an eclipse on a fainter neighbour whose light leaks into
the aperture (a blend), which :mod:`transitml.centroid` tests for.

This module holds the container, a no-pickle ``.npz`` format for it, and the
conversion from a ``lightkurve.TargetPixelFile``.  As in
:mod:`transitml.data.mast`, ``lightkurve`` is imported lazily and is not in
``requirements.txt``.

Pixel conventions
-----------------
The flux cube is ``[time, row, column]`` (``[t, y, x]``), as lightkurve and
the FITS files store it.  Positions inside the stamp are ``(column, row)`` in
pixels with ``(0, 0)`` at the centre of the first pixel.  ``column0`` and
``row0`` are the CCD coordinates of that pixel, so ``column0 + x`` is the CCD
column of stamp column ``x``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import LightCurve

#: TESS plate scale, arcseconds per pixel.
TESS_PIXEL_SCALE_ARCSEC = 21.0

#: Bumped if the npz layout below ever changes.
TPF_FORMAT_VERSION = 1

_LIGHTKURVE_HINT = (
    "Downloading target pixel files requires `lightkurve` and outbound network "
    "access to the MAST archive. Install with `pip install lightkurve`, or pass "
    "a saved .npz with --tpf."
)


@dataclass
class TargetPixelData:
    """A target pixel file reduced to plain arrays.

    Attributes
    ----------
    target_id:
        Same convention as :class:`~transitml.data.base.LightCurve`.
    time:
        Cadence times in days, shape ``(n_t,)``, strictly increasing.
    flux:
        Background-subtracted flux per pixel, shape ``(n_t, n_rows, n_cols)``.
        NaN marks a missing value (a bad pixel or cadence).
    aperture:
        Boolean ``(n_rows, n_cols)`` mask of the pixels summed into the light
        curve.
    flux_err:
        Optional per-pixel 1-sigma uncertainty, same shape as ``flux``.
    column0, row0:
        CCD column and row of stamp pixel ``(0, 0)``.
    target_position:
        Catalogue position of the target in stamp pixels, ``(column, row)``,
        when known (from the TPF's WCS).  ``None`` otherwise.
    meta:
        Free-form provenance (sector, camera, CCD, ...).  Must be
        JSON-serialisable to survive :func:`save_tpf`.
    """

    target_id: str
    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    aperture: NDArray[np.bool_]
    flux_err: NDArray[np.float64] | None = None
    column0: int = 0
    row0: int = 0
    target_position: tuple[float, float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.flux = np.asarray(self.flux, dtype=np.float64)
        self.aperture = np.asarray(self.aperture, dtype=bool)
        if self.flux.ndim != 3:
            raise ValueError(
                f"{self.target_id}: flux must be [time, row, column], got {self.flux.shape}"
            )
        if self.time.shape != (self.flux.shape[0],):
            raise ValueError(
                f"{self.target_id}: {self.time.size} times for {self.flux.shape[0]} flux frames"
            )
        if self.aperture.shape != self.flux.shape[1:]:
            raise ValueError(
                f"{self.target_id}: aperture {self.aperture.shape} does not match "
                f"the stamp {self.flux.shape[1:]}"
            )
        if self.flux_err is not None:
            self.flux_err = np.asarray(self.flux_err, dtype=np.float64)
            if self.flux_err.shape != self.flux.shape:
                raise ValueError(
                    f"{self.target_id}: flux_err shape does not match flux"
                )
        if self.time.size and np.any(np.diff(self.time) <= 0):
            raise ValueError(
                f"{self.target_id}: time array must be strictly increasing"
            )
        if self.target_position is not None:
            col, row = (float(v) for v in self.target_position)
            self.target_position = (
                (col, row) if np.isfinite(col) and np.isfinite(row) else None
            )

    @property
    def n_cadences(self) -> int:
        return int(self.time.size)

    @property
    def shape(self) -> tuple[int, int]:
        """Stamp shape, ``(n_rows, n_cols)``."""
        return (int(self.flux.shape[1]), int(self.flux.shape[2]))

    def pixel_grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(column, row)`` stamp coordinates of every pixel centre, each ``(n_rows, n_cols)``."""
        rows, cols = np.indices(self.shape, dtype=np.float64)
        return cols, rows

    def to_light_curve(self) -> LightCurve:
        """Simple aperture photometry: the flux summed over the aperture, median-normalised.

        Cadences where any aperture pixel is NaN are dropped rather than
        summed short, which would look like a dip.
        """
        if not self.aperture.any():
            raise ValueError(f"{self.target_id}: empty aperture")
        pixels = self.flux[:, self.aperture]
        flux = pixels.sum(axis=1)
        if self.flux_err is not None:
            err = np.sqrt(np.sum(self.flux_err[:, self.aperture] ** 2, axis=1))
        else:
            err = np.full(flux.size, np.nan)
        good = np.isfinite(self.time) & np.isfinite(flux)
        time, flux, err = self.time[good], flux[good], err[good]
        if time.size == 0:
            raise ValueError(
                f"{self.target_id}: no cadence with every aperture pixel finite"
            )
        median = float(np.median(flux))
        if median <= 0.0:
            raise ValueError(f"{self.target_id}: aperture flux has non-positive median")
        flux, err = flux / median, err / median
        if not np.all(np.isfinite(err)):
            from ..preprocess import point_to_point_sigma  # avoids an import cycle

            err = np.full(flux.size, point_to_point_sigma(flux))
        meta = dict(self.meta)
        meta["source"] = "target pixel file aperture sum"
        return LightCurve(self.target_id, time, flux, err, label=None, meta=meta)


def bin_target_pixels(tpf: TargetPixelData, cadence_seconds: float) -> TargetPixelData:
    """The pixels averaged onto a slower cadence, as
    :func:`~transitml.data.base.bin_light_curve` does for a light curve.

    Each pixel of a binned frame is the mean of its finite values over the
    bin's frames (NaN where it has none), with the error of that mean.  Bins
    holding fewer than half the frames of a full one are dropped.  Pixels
    already at ``cadence_seconds`` or slower come back as they are.
    """
    if tpf.n_cadences < 2:
        return tpf
    width = cadence_seconds / 86400.0
    native = float(np.median(np.diff(tpf.time)))
    if native >= 0.9 * width:
        return tpf
    index = np.floor((tpf.time - tpf.time[0] + 0.5 * native) / width).astype(np.int64)
    starts = np.flatnonzero(np.concatenate([[True], np.diff(index) > 0]))
    counts = np.diff(np.append(starts, tpf.n_cadences))
    keep = counts >= 0.5 * width / native
    finite = np.isfinite(tpf.flux)
    n = np.add.reduceat(finite, starts, axis=0)[keep]
    total = np.add.reduceat(np.where(finite, tpf.flux, 0.0), starts, axis=0)[keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = np.where(n > 0, total / n, np.nan)
        flux_err = None
        if tpf.flux_err is not None:
            square = np.where(finite, tpf.flux_err**2, 0.0)
            flux_err = np.where(n > 0, np.sqrt(np.add.reduceat(square, starts, axis=0)[keep]) / n, np.nan)
    meta = dict(tpf.meta)
    meta["binned_from_seconds"] = round(native * 86400.0)
    return replace(
        tpf,
        time=np.add.reduceat(tpf.time, starts)[keep] / counts[keep],
        flux=flux,
        flux_err=flux_err,
        meta=meta,
    )


# --------------------------------------------------------------------------
# npz (no pickle)
# --------------------------------------------------------------------------
def save_tpf(tpf: TargetPixelData, path: str | Path) -> Path:
    """Write one :class:`TargetPixelData` to a compressed ``.npz`` without pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "format_version": np.int64(TPF_FORMAT_VERSION),
        "target_id": np.array(tpf.target_id, dtype=str),
        "time": tpf.time,
        "flux": tpf.flux,
        "aperture": tpf.aperture,
        "origin": np.array([tpf.column0, tpf.row0], dtype=np.int64),
        "target_position": np.array(
            tpf.target_position
            if tpf.target_position is not None
            else (np.nan, np.nan),
            dtype=np.float64,
        ),
        "meta": np.array(json.dumps(tpf.meta, default=str), dtype=str),
    }
    if tpf.flux_err is not None:
        arrays["flux_err"] = tpf.flux_err
    np.savez_compressed(path, **arrays)
    return path


def load_tpf(path: str | Path) -> TargetPixelData:
    """Inverse of :func:`save_tpf`."""
    with np.load(path, allow_pickle=False) as data:
        if "flux" not in data or data["flux"].ndim != 3:
            raise ValueError(
                f"{path}: not a target pixel file (no [time, row, column] flux cube)"
            )
        version = int(data["format_version"]) if "format_version" in data else 0
        if version > TPF_FORMAT_VERSION:
            raise ValueError(
                f"{path}: format version {version} is newer than this code"
            )
        position = data["target_position"].astype(float)
        column0, row0 = (int(v) for v in data["origin"])
        return TargetPixelData(
            target_id=str(data["target_id"]),
            time=data["time"].astype(np.float64),
            flux=data["flux"].astype(np.float64),
            aperture=data["aperture"].astype(bool),
            flux_err=data["flux_err"].astype(np.float64)
            if "flux_err" in data
            else None,
            column0=column0,
            row0=row0,
            target_position=(float(position[0]), float(position[1])),
            meta=json.loads(str(data["meta"])),
        )


# --------------------------------------------------------------------------
# lightkurve
# --------------------------------------------------------------------------
def _plain(value: Any) -> Any:
    """An astropy Quantity / Time / masked value as a plain float array."""
    for attr in ("value", "data"):
        if hasattr(value, attr) and not isinstance(value, np.ndarray):
            value = getattr(value, attr)
    array = np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan)
    return np.asarray(array, dtype=np.float64)


def _target_position(lk_tpf) -> tuple[float, float] | None:
    """The target's catalogue position in stamp pixels, from the TPF's WCS."""
    try:
        ra, dec = float(lk_tpf.ra), float(lk_tpf.dec)
        x, y = lk_tpf.wcs.all_world2pix([[ra, dec]], 0)[0]
    except Exception:  # noqa: BLE001 - missing header keys or WCS vary by product
        return None
    return (float(x), float(y)) if np.isfinite(x) and np.isfinite(y) else None


def from_lightkurve(lk_tpf, target_id: str | None = None) -> TargetPixelData:
    """Convert a ``lightkurve.TargetPixelFile`` into :class:`TargetPixelData`.

    The aperture is the pipeline's own (``pipeline_mask``) when it has one and
    otherwise lightkurve's threshold mask.  Cadences with a non-finite time are
    dropped and the rest sorted; per-pixel NaNs are kept as NaN.
    """
    time = _plain(lk_tpf.time)
    flux = _plain(lk_tpf.flux)
    flux_err = getattr(lk_tpf, "flux_err", None)
    flux_err = _plain(flux_err) if flux_err is not None else None
    if flux_err is not None and flux_err.shape != flux.shape:
        flux_err = None

    try:
        aperture = np.asarray(lk_tpf.pipeline_mask)
    except Exception:  # noqa: BLE001 - absent on some products (e.g. FFI cutouts)
        aperture = np.zeros(0)
    if (
        aperture.ndim != 2
        or aperture.shape != flux.shape[1:]
        or not aperture.astype(bool).any()
    ):
        aperture = np.asarray(lk_tpf.create_threshold_mask())

    good = np.isfinite(time)
    order = np.argsort(time[good], kind="stable")
    time, flux = time[good][order], flux[good][order]
    flux_err = flux_err[good][order] if flux_err is not None else None
    unique = (
        np.concatenate([[True], np.diff(time) > 0]) if time.size else np.zeros(0, bool)
    )

    meta_source = getattr(lk_tpf, "meta", {}) or {}
    getter = getattr(meta_source, "get", lambda *_: None)
    meta = {
        "kind": "real",
        "sector": getter("SECTOR"),
        "camera": getter("CAMERA"),
        "ccd": getter("CCD"),
        "tess_mag": getter("TESSMAG"),
    }
    meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in meta.items()}
    return TargetPixelData(
        target_id=target_id or str(getattr(lk_tpf, "targetid", "unknown")),
        time=time[unique],
        flux=flux[unique],
        flux_err=flux_err[unique] if flux_err is not None else None,
        aperture=aperture.astype(bool),
        column0=int(getattr(lk_tpf, "column", 0)),
        row0=int(getattr(lk_tpf, "row", 0)),
        target_position=_target_position(lk_tpf),
        meta=meta,
    )


def _import_lightkurve():  # pragma: no cover - requires the optional dependency
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise ImportError(_LIGHTKURVE_HINT) from exc
    return lk


def download_tpfs(
    target_id: str,
    *,
    mission: str = "TESS",
    author: str = "SPOC",
    exposure_time: int | None = None,
    sector: int | None = None,
    quality_bitmask: str = "default",
) -> list[TargetPixelData]:
    """Every matching target pixel file for one star from MAST; ``[]`` if none or on failure.

    Checked against the real archive on TOI benchmark hosts, where it returns
    the same pixels as the TESS-SPOC files fetched directly; the tests use a
    stand-in object.  A download error is a warning, not an exception, as in
    :class:`~transitml.data.mast.MASTLightCurveSource`.
    """
    lk = _import_lightkurve()
    try:
        search = lk.search_targetpixelfile(
            target_id,
            mission=mission,
            author=author,
            exptime=exposure_time,
            sector=sector,
        )
        if len(search) == 0:
            return []
        collection = search.download_all(quality_bitmask=quality_bitmask)
        return [from_lightkurve(tpf, target_id) for tpf in collection]
    except Exception as exc:  # noqa: BLE001 - network and FITS errors vary
        warnings.warn(
            f"{target_id}: no target pixel file ({type(exc).__name__}: {exc})",
            stacklevel=2,
        )
        return []
