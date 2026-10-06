"""Host-star parameters from the TESS Input Catalog.

The secondary-eclipse test allows for a planet's own occultation, and how deep
that can be depends on the star: its temperature, and its density, which with
the period fixes how close the planet orbits (see
:func:`~transitml.physics.max_occultation_fraction`).  Synthetic and injected
curves carry the star they were drawn for, and a MAST light curve carries the
TIC's values in its header.  A curve cache does not keep those, so the TOI
benchmark looks its stars up in the TIC (v8.2) itself and keeps them in a
small CSV beside the TOI table, read again on the next run.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from ..physics import RHO_SUN_CGS
from .base import LightCurve
from .injection import tic_number

#: A star table's columns after ``tic``, named as :func:`~transitml.data.base.stellar_parameters`
#: reads them from a light curve's metadata.
STAR_FIELDS: tuple[str, ...] = (
    "teff_k",
    "logg_cgs",
    "r_star_rsun",
    "m_star_msun",
    "rho_star_cgs",
)

#: TIC column for each field, and the factor that converts it.  The TIC gives
#: density in solar units.
_TIC_COLUMNS: dict[str, tuple[str, float]] = {
    "teff_k": ("Teff", 1.0),
    "logg_cgs": ("logg", 1.0),
    "r_star_rsun": ("rad", 1.0),
    "m_star_msun": ("mass", 1.0),
    "rho_star_cgs": ("rho", RHO_SUN_CGS),
}

_ASTROQUERY_HINT = (
    "Looking stars up in the TIC needs `astroquery` (installed with `lightkurve`) "
    "and outbound network access to MAST."
)

Star = dict[str, float]


def _number(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def read_star_table(path: str | Path) -> dict[int, Star]:
    """``{tic: {field: value}}`` from a star table; blanks read as NaN."""
    with open(path, newline="") as handle:
        return {
            int(row["tic"]): {name: _number(row.get(name)) for name in STAR_FIELDS}
            for row in csv.DictReader(handle)
        }


def write_star_table(path: str | Path, stars: Mapping[int, Mapping[str, float]]) -> None:
    """Write ``stars`` sorted by TIC ID, unknown values left blank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tic", *STAR_FIELDS))
        for tic in sorted(stars):
            values = (_number(stars[tic].get(name)) for name in STAR_FIELDS)
            writer.writerow((tic, *("" if math.isnan(v) else f"{v:.6g}" for v in values)))


def fetch_tic_stars(tic_ids: Iterable[int], *, chunk: int = 250) -> dict[int, Star]:
    """Look stars up in the TIC at MAST, ``chunk`` IDs per query.

    A star the TIC does not return is left out; one it returns without a
    value for some field gets NaN there.
    """
    try:
        from astroquery.mast import Catalogs
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_ASTROQUERY_HINT) from exc

    ids = sorted(set(tic_ids))
    stars: dict[int, Star] = {}
    for start in range(0, len(ids), chunk):  # pragma: no cover - needs network
        table = Catalogs.query_criteria(catalog="Tic", ID=ids[start : start + chunk])
        for row in table:
            stars[int(row["ID"])] = {
                name: _number(row[column]) * factor
                for name, (column, factor) in _TIC_COLUMNS.items()
            }
    return stars


def load_or_fetch_stars(
    tic_ids: Iterable[int],
    path: str | Path,
    *,
    fetch: Callable[[list[int]], Mapping[int, Star]] | None = None,
) -> dict[int, Star]:
    """Stars from the table at ``path``, looking up in the TIC only those it lacks.

    ``fetch`` defaults to :func:`fetch_tic_stars`.  Stars the TIC has no entry
    for are written with blank values, so they are not looked up again on
    every run; if the lookup itself fails, nothing is written and the error
    propagates.  Returns the stars asked for.
    """
    path = Path(path)
    table = read_star_table(path) if path.exists() else {}
    wanted = sorted(set(tic_ids))
    missing = [tic for tic in wanted if tic not in table]
    if missing:
        found = (fetch or fetch_tic_stars)(missing)
        for tic in missing:
            table[tic] = dict(found.get(tic, {name: math.nan for name in STAR_FIELDS}))
        write_star_table(path, table)
    return {tic: table[tic] for tic in wanted}


def with_star(lc: LightCurve, stars: Mapping[int, Mapping[str, float]]) -> LightCurve:
    """``lc`` with its star's known values in ``meta``; unchanged if it has none."""
    tic = tic_number(lc.target_id)
    star = stars.get(tic) if tic is not None else None
    if not star:
        return lc
    known = {name: value for name in STAR_FIELDS if not math.isnan(value := _number(star.get(name)))}
    return replace(lc, meta={**lc.meta, **known}) if known else lc
