"""Injection-recovery: synthetic eclipses in real photometry.

The README's weakest point is the noise model.  Synthetic red noise is a
stationary 1/f process, and real TESS systematics are not: scattered light on
the 13.7-day orbit, focus drift, pointing jitter, crowding.  A pipeline that
looks good on synthetic noise can look much worse on real noise, and the only
way to find out is to run it on real light curves.

Real light curves have no trustworthy labels, though.  Injection-recovery is
how the mission teams get them: take real photometry of stars with no known
planet, inject a known population of transits and eclipses into some of them,
and see what the pipeline recovers.  The noise is real; the labels are known
by construction.

:class:`InjectionSource` does exactly that, drawing planets and binaries from
the same :func:`~transitml.data.synthetic.planet_signal` and
:func:`~transitml.data.synthetic.binary_signal` the synthetic generator uses,
so the only thing that changes between the two experiments is the noise.

Two caveats are worth stating up front:

* **Uninjected curves are labelled 0, which is only as true as the exclusion
  list.**  A host of an unknown planet becomes a mislabelled negative.  At a
  few per cent of stars that is a small, known bias against precision; the
  TOI catalogue removes the hosts we already know about
  (:func:`load_excluded_tic_ids`).
* **Injected eclipses are multiplied into the flux** (``flux * (1 - dip)``),
  so they are diluted by any contaminating light already in the aperture,
  exactly as a real eclipse would be after the pipeline's crowding correction.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import EclipsingBinaryConfig, PlanetConfig, StarConfig
from ..physics import RHO_SUN_CGS
from .base import LightCurve, LightCurveSource
from .synthetic import CurveKind, binary_signal, planet_signal

_TIC_RE = re.compile(r"(\d+)")


def tic_number(target_id: str) -> int | None:
    """Return the integer TIC ID in ``"TIC 307210830"`` (or ``"307210830"``)."""
    match = _TIC_RE.search(str(target_id))
    return int(match.group(1)) if match else None


def robust_white_sigma(flux: NDArray[np.float64]) -> float:
    """Per-cadence white-noise scatter from point-to-point differences.

    Differencing removes anything slower than a cadence, so stellar
    variability and slow systematics do not inflate the estimate; the MAD
    keeps flares and the injected eclipse itself from doing so.
    """
    diff = np.diff(np.asarray(flux, dtype=float))
    if diff.size < 2:
        return float("nan")
    mad = np.median(np.abs(diff - np.median(diff)))
    return float(1.4826 * mad / np.sqrt(2.0))


class InjectionSource(LightCurveSource):
    """Inject a labelled planet / binary population into real light curves.

    Parameters
    ----------
    base_curves:
        Real, normalised light curves of stars believed to host no transiting
        planet.  A curve labelled 1 is rejected: it is a known host and would
        become a mislabelled negative.
    positive_rate, eclipsing_binary_rate:
        Fraction of curves that get a planet, and an eclipsing binary.  As in
        the synthetic generator the counts are exact and their positions are
        shuffled, so the realised rate is not a random variable.
    seed:
        Curve ``i`` uses an RNG derived from ``(seed, i)``, so any single
        injected curve can be regenerated on its own.
    star:
        Used only to draw a stellar radius (hence density, hence transit
        duration) when the base curve's metadata carries no ``rho_star_cgs``.
    """

    def __init__(
        self,
        base_curves: Sequence[LightCurve],
        positive_rate: float,
        eclipsing_binary_rate: float,
        *,
        seed: int,
        star: StarConfig | None = None,
        planet: PlanetConfig | None = None,
        eb: EclipsingBinaryConfig | None = None,
    ) -> None:
        if positive_rate < 0 or eclipsing_binary_rate < 0:
            raise ValueError("rates must be non-negative")
        if positive_rate + eclipsing_binary_rate > 1.0:
            raise ValueError("positive_rate + eclipsing_binary_rate must not exceed 1")
        known = [lc.target_id for lc in base_curves if lc.label == 1]
        if known:
            raise ValueError(
                f"{len(known)} base curve(s) are labelled as planet hosts "
                f"(e.g. {known[0]}); exclude known hosts before injecting"
            )
        self.base_curves = list(base_curves)
        self.positive_rate = float(positive_rate)
        self.eclipsing_binary_rate = float(eclipsing_binary_rate)
        self.seed = int(seed)
        self.star = star or StarConfig()
        self.planet = planet or PlanetConfig()
        self.eb = eb or EclipsingBinaryConfig()
        self._kinds = self._assign_kinds()

    def _assign_kinds(self) -> list[CurveKind]:
        n = len(self.base_curves)
        n_pos = int(round(n * self.positive_rate))
        n_eb = int(round(n * self.eclipsing_binary_rate))
        kinds: list[CurveKind] = (
            ["planet"] * n_pos + ["eclipsing_binary"] * n_eb + ["noise"] * (n - n_pos - n_eb)
        )
        np.random.default_rng(self.seed).shuffle(kinds)  # type: ignore[arg-type]
        return kinds

    @property
    def kinds(self) -> list[CurveKind]:
        """Ground-truth population label for every curve, in order."""
        return list(self._kinds)

    def __len__(self) -> int:
        return len(self.base_curves)

    def __iter__(self) -> Iterator[LightCurve]:
        for index in range(len(self)):
            yield self.generate(index)

    def _stellar_density(self, base: LightCurve, rng: np.random.Generator) -> float:
        rho = base.meta.get("rho_star_cgs")
        if rho is not None and np.isfinite(rho) and rho > 0:
            return float(rho)
        r_star = float(rng.uniform(*self.star.radius_range_rsun))
        return RHO_SUN_CGS * r_star**0.9 / r_star**3

    def generate(self, index: int) -> LightCurve:
        """Return base curve ``index`` with its assigned eclipse injected."""
        base = self.base_curves[index].finite()
        kind = self._kinds[index]
        rng = np.random.default_rng([self.seed, index])
        rho_star = self._stellar_density(base, rng)

        if kind == "planet":
            dip, signal_meta = planet_signal(base.time, rng, rho_star, self.planet)
        elif kind == "eclipsing_binary":
            dip, signal_meta = binary_signal(base.time, rng, rho_star, self.eb)
        else:
            dip, signal_meta = np.zeros_like(base.time), {}

        sigma_white = robust_white_sigma(base.flux)
        n_in = int(np.count_nonzero(dip > 0.5 * dip.max())) if dip.max() > 0 else 0
        lo, hi = np.percentile(base.flux, [5.0, 95.0])

        meta: dict[str, Any] = dict(base.meta)
        meta.update(signal_meta)
        meta.update(
            {
                "kind": kind,
                "base_target_id": base.target_id,
                "rho_star_cgs": rho_star,
                "sigma_white": sigma_white,
                "variability_amplitude": meta.get(
                    "variability_amplitude", float((hi - lo) / 2.0)
                ),
                "n_in_transit_cadences": n_in,
                "true_snr": float(dip.max() / sigma_white * np.sqrt(n_in))
                if n_in > 0 and sigma_white > 0
                else 0.0,
            }
        )
        return LightCurve(
            target_id=f"{base.target_id} inj{index:05d}",
            time=base.time,
            flux=base.flux * (1.0 - dip),
            flux_err=base.flux_err,
            label=1 if kind == "planet" else 0,
            meta=meta,
        )


def _tic_column(fieldnames: Sequence[str] | None) -> str | None:
    # MAST's TESS-SPOC target lists comment their header: ``#TIC_ID,RA,DEC``.
    names = ("tic id", "tic", "tic_id", "ticid")
    return next(
        (c for c in fieldnames or [] if c.strip().lstrip("#").strip().lower() in names), None
    )


def read_target_list(path: str | Path) -> list[str]:
    """Read target IDs: one per line, or a CSV with a ``tic``/``TIC ID`` column.

    Returns IDs normalised to ``"TIC <n>"``, in file order, without duplicates.
    """
    path = Path(path)
    text = path.read_text().splitlines()
    # Drop leading comments, except a commented header that names the TIC
    # column (MAST's ``#TIC_ID,RA,DEC``).
    while text and text[0].startswith("#") and _tic_column(text[0].split(",")) is None:
        text = text[1:]
    rows: list[str]
    first = text[0] if text else ""
    if "," in first:
        reader = csv.DictReader(text)
        column = _tic_column(reader.fieldnames)
        if column is None:
            raise ValueError(f"{path}: no TIC column in {reader.fieldnames}")
        rows = [row[column] for row in reader]
    else:
        rows = [line for line in text if line.strip() and not line.startswith("#")]
    seen: dict[str, None] = {}
    for row in rows:
        number = tic_number(row)
        if number is not None:
            seen.setdefault(f"TIC {number}", None)
    return list(seen)


def load_excluded_tic_ids(path: str | Path) -> set[int]:
    """TIC IDs to exclude, from an ExoFOP TOI table (or any CSV with a TIC column).

    ExoFOP's TOI export comments its header with ``#`` lines on some
    downloads; those are skipped.
    """
    text = Path(path).read_text().splitlines()
    lines = [line for line in text if line and not line.startswith("#")]
    reader = csv.DictReader(lines)
    column = _tic_column(reader.fieldnames)
    if column is None:
        raise ValueError(f"{path}: no 'TIC ID' column in {reader.fieldnames}")
    out: set[int] = set()
    for row in reader:
        number = tic_number(row[column])
        if number is not None:
            out.add(number)
    return out


def exclude_known_hosts(
    targets: Iterable[str], excluded: set[int]
) -> tuple[list[str], list[str]]:
    """Split ``targets`` into ``(kept, dropped)`` by TIC number."""
    kept: list[str] = []
    dropped: list[str] = []
    for target in targets:
        (dropped if tic_number(target) in excluded else kept).append(target)
    return kept, dropped


def save_curves(curves: Sequence[LightCurve], path: str | Path) -> None:
    """Save light curves to one ``.npz`` (no pickle), so a run can be repeated offline."""
    lengths = np.array([lc.n_cadences for lc in curves], dtype=np.int64)
    np.savez_compressed(
        path,
        lengths=lengths,
        time=np.concatenate([lc.time for lc in curves]) if curves else np.zeros(0),
        flux=np.concatenate([lc.flux for lc in curves]) if curves else np.zeros(0),
        flux_err=np.concatenate([lc.flux_err for lc in curves]) if curves else np.zeros(0),
        target_id=np.array([lc.target_id for lc in curves], dtype=str),
        label=np.array([-1 if lc.label is None else lc.label for lc in curves], dtype=np.int64),
        meta=np.array([json.dumps(lc.meta, default=str) for lc in curves], dtype=str),
    )


def load_curves(path: str | Path) -> list[LightCurve]:
    """Inverse of :func:`save_curves`."""
    with np.load(path, allow_pickle=False) as data:
        bounds = np.concatenate([[0], np.cumsum(data["lengths"])])
        curves = []
        for i, target_id in enumerate(data["target_id"]):
            sl = slice(bounds[i], bounds[i + 1])
            label = int(data["label"][i])
            curves.append(
                LightCurve(
                    target_id=str(target_id),
                    time=data["time"][sl].astype(np.float64),
                    flux=data["flux"][sl].astype(np.float64),
                    flux_err=data["flux_err"][sl].astype(np.float64),
                    label=None if label < 0 else label,
                    meta=json.loads(str(data["meta"][i])),
                )
            )
    return curves
