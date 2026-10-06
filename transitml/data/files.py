"""Light curves from local files, for vetting a single star offline.

Two formats are read:

* **CSV** with a header row naming ``time`` and ``flux`` columns and,
  optionally, ``flux_err`` (names are matched case-insensitively; other
  columns are ignored).  Time is in days.  Flux may be in any units: it is
  divided by its median, as :class:`~transitml.data.mast.MASTLightCurveSource`
  does, so the result is on the same relative scale as every other source.
  When ``flux_err`` is missing, every cadence gets the point-to-point scatter.
* **npz** in the cache format written by
  :func:`transitml.data.injection.save_curves`.  A cache can hold many stars;
  :func:`read_light_curves` returns all of them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .base import LightCurve
from .injection import load_curves


def _normalised(target_id: str, time, flux, flux_err) -> LightCurve:
    """Finite, time-sorted, de-duplicated, median-normalised light curve."""
    good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[good], flux[good], flux_err[good]
    if time.size == 0:
        raise ValueError(f"{target_id}: no finite cadences")
    order = np.argsort(time, kind="stable")
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    unique = np.concatenate([[True], np.diff(time) > 0])
    time, flux, flux_err = time[unique], flux[unique], flux_err[unique]
    median = float(np.median(flux))
    if median == 0.0:
        raise ValueError(f"{target_id}: median flux is zero; cannot normalise")
    return LightCurve(
        target_id=target_id,
        time=time,
        flux=flux / median,
        flux_err=np.abs(flux_err / median),
        label=None,
        meta={"kind": "file"},
    )


def read_csv_light_curve(path: str | Path, target_id: str | None = None) -> LightCurve:
    """Read a ``time,flux[,flux_err]`` CSV into a normalised :class:`LightCurve`."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        if "time" not in columns or "flux" not in columns:
            raise ValueError(f"{path}: needs a header with 'time' and 'flux' columns")
        rows = list(reader)

    def column(name: str) -> np.ndarray:
        values = []
        for row in rows:
            try:
                values.append(float(row[columns[name]]))
            except (TypeError, ValueError):
                values.append(np.nan)
        return np.asarray(values, dtype=np.float64)

    time, flux = column("time"), column("flux")
    if "flux_err" in columns:
        flux_err = column("flux_err")
    else:
        from ..preprocess import point_to_point_sigma  # avoids an import cycle

        finite = flux[np.isfinite(flux)]
        sigma = point_to_point_sigma(finite) if finite.size else np.nan
        flux_err = np.full(flux.size, sigma)
    lc = _normalised(target_id or path.stem, time, flux, flux_err)
    lc.meta["source_file"] = str(path)
    return lc


def read_light_curves(path: str | Path, target_id: str | None = None) -> list[LightCurve]:
    """Every light curve in a ``.csv`` or ``.npz`` file.

    For an npz cache, ``target_id`` keeps only the curves of that star.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [read_csv_light_curve(path, target_id)]
    if suffix == ".npz":
        curves = load_curves(path)
        if target_id is not None:
            curves = [lc for lc in curves if lc.target_id == target_id]
            if not curves:
                raise ValueError(f"{path}: no curve with target id {target_id!r}")
        return curves
    raise ValueError(f"{path}: unsupported light-curve file (expected .csv or .npz)")
