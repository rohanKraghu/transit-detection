"""Real labels: TESS Objects of Interest and their follow-up dispositions.

Injection-recovery (:mod:`transitml.data.injection`) gives labels that are
true by construction, but the signals are still drawn from our own planet and
binary models.  The other honest test is the one the vetting tool will face in
use: real signals that the TESS pipelines flagged, labelled by what follow-up
observations later showed them to be.

The TESS Follow-up Observing Program Working Group (TFOPWG) assigns every TOI
one of

======  =======================  =============
code    meaning                  label here
======  =======================  =============
CP      confirmed planet         1
KP      known planet             1
FP      false positive           0
FA      false alarm              0
PC      planet candidate         unlabelled
APC     ambiguous candidate      unlabelled
======  =======================  =============

Labels are per **star**, because the pipeline scores one light curve per star:

* a star is a positive if any of its TOIs is CP or KP;
* it is a negative only if every one of its TOIs is FP or FA;
* anything else (an open PC or APC, or a blank disposition, with no CP/KP
  beside it) is left out, because nobody knows yet what it is.

Two caveats travel with these labels, and the benchmark report repeats them:

* **Every star here was already flagged by a TESS pipeline.**  Both classes
  passed an automated detection and vetting step, so the negatives are the
  hard false positives that survived it, not random quiet stars.  This tests
  vetting, not detection.
* **The positive rate is set by the catalogue, not the sky.**  Roughly half of
  the labelled stars are planets, against a few per cent in a survey, so
  precision here does not transfer to a survey.  Recall on planets and the
  fraction of known false positives rejected do transfer.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .injection import tic_number

#: TFOPWG dispositions that make a TOI a planet, and those that make it not one.
POSITIVE_DISPOSITIONS: frozenset[str] = frozenset({"CP", "KP"})
NEGATIVE_DISPOSITIONS: frozenset[str] = frozenset({"FP", "FA"})

#: Accepted spellings of each field, in order of preference: ExoFOP's TOI
#: export first, then the NASA Exoplanet Archive ``toi`` table, then the short
#: ``TIC ID,TOI,Disposition`` list in ``data/real_injection/toi.csv``.
_COLUMNS: dict[str, tuple[str, ...]] = {
    "tic": ("TIC ID", "tid", "TIC"),
    "toi": ("TOI", "toi"),
    "disposition": ("TFOPWG Disposition", "tfopwg_disp", "Disposition"),
    "period": ("Period (days)", "pl_orbper"),
    "epoch_bjd": ("Epoch (BJD)", "pl_tranmid"),
    "duration_hours": ("Duration (hours)", "pl_trandurh"),
    "depth_ppm": ("Depth (ppm)", "pl_trandep"),
    "snr": ("Planet SNR",),
    "tess_mag": ("TESS Mag", "st_tmag"),
    "sectors": ("Sectors",),
}


@dataclass(frozen=True)
class TOI:
    """One row of the TOI catalogue."""

    tic: int
    toi: str
    disposition: str
    period: float = math.nan
    epoch_bjd: float = math.nan
    duration_hours: float = math.nan
    depth_ppm: float = math.nan
    snr: float = math.nan
    tess_mag: float = math.nan
    sectors: tuple[int, ...] = ()

    @property
    def label(self) -> int | None:
        """1 for CP/KP, 0 for FP/FA, ``None`` for anything still open."""
        if self.disposition in POSITIVE_DISPOSITIONS:
            return 1
        if self.disposition in NEGATIVE_DISPOSITIONS:
            return 0
        return None


@dataclass(frozen=True)
class BenchmarkTarget:
    """One labelled star, the sector to fetch for it, and its reference TOI.

    ``reference`` is the TOI whose period the search is checked against: the
    highest-SNR TOI that agrees with the star's label (a CP for a planet host,
    an FP or FA for a negative).
    """

    tic: int
    label: int
    sector: int
    reference: TOI
    tois: tuple[TOI, ...] = field(default=())

    @property
    def target_id(self) -> str:
        return f"TIC {self.tic}"


def _float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def parse_sectors(value: str | None) -> tuple[int, ...]:
    """``"14,15,41"`` -> ``(14, 15, 41)``; blanks and junk are ignored."""
    out: list[int] = []
    for part in str(value or "").replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return tuple(out)


def parse_sector_spec(spec: str) -> list[int]:
    """``"14"``, ``"14,15"`` or ``"14-26"`` -> sector numbers, in the order given."""
    sectors: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(p) for p in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"empty sector range {part!r}")
            sectors.extend(range(lo, hi + 1))
        else:
            sectors.append(int(part))
    if not sectors:
        raise ValueError(f"no sectors in {spec!r}")
    return list(dict.fromkeys(sectors))


def _resolve(fieldnames: Sequence[str]) -> dict[str, str]:
    present = {name.strip(): name for name in fieldnames}
    resolved: dict[str, str] = {}
    for key, options in _COLUMNS.items():
        for option in options:
            if option in present:
                resolved[key] = present[option]
                break
    return resolved


def read_toi_table(path: str | Path) -> list[TOI]:
    """Read a TOI catalogue export into :class:`TOI` rows.

    Accepts ExoFOP's ``download_toi.php?output=csv`` export (the only one
    that lists observed sectors), the NASA Exoplanet Archive ``toi`` table, or
    any CSV with a TIC column and a disposition column.  Leading ``#`` comment
    lines are skipped, as in :func:`~transitml.data.injection.load_excluded_tic_ids`.
    """
    path = Path(path)
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    reader = csv.DictReader(lines)
    columns = _resolve(reader.fieldnames or [])
    for required in ("tic", "disposition"):
        if required not in columns:
            raise ValueError(f"{path}: no {required} column in {reader.fieldnames}")

    rows: list[TOI] = []
    for row in reader:
        tic = tic_number(row[columns["tic"]])
        if tic is None:
            continue
        values = {key: row.get(name) for key, name in columns.items()}
        get = values.get
        rows.append(
            TOI(
                tic=tic,
                toi=(get("toi") or "").strip(),
                disposition=(get("disposition") or "").strip().upper(),
                # Single-transit TOIs carry a period of 0: unknown, not zero.
                period=_float(get("period")) if _float(get("period")) > 0 else math.nan,
                epoch_bjd=_float(get("epoch_bjd")),
                duration_hours=_float(get("duration_hours")),
                depth_ppm=_float(get("depth_ppm")),
                snr=_float(get("snr")),
                tess_mag=_float(get("tess_mag")),
                sectors=parse_sectors(get("sectors")),
            )
        )
    return rows


#: Why a false positive was retired, read from the free-text ExoFOP comment.
#: The signal is on another star (a nearby or background eclipsing binary, a
#: nearby planet candidate, a centroid offset, or a source named elsewhere) ...
_OFF_TARGET = re.compile(
    r"\bNEB\b|\bBEB\b|\bNPC\b|nearby eclipsing|nearby planet candidate|off.?target|"
    r"centroid offset|centroids? show|offset to|offset towards|(on|from) (a )?neighbo|"
    r"(correct|actual|true) source|centered on TIC|blend",
    re.IGNORECASE,
)
#: ... or it is a binary at the target (spectroscopic or photometric evidence).
_ON_TARGET = re.compile(
    r"SEB[12]|\bSB[12]\b|\bEB\b|heartbeat|v[- ]shaped|secondary|odd.?even|too large",
    re.IGNORECASE,
)

#: The three answers :func:`false_positive_reason` gives, in report order.
FALSE_POSITIVE_REASONS: tuple[str, ...] = ("off target", "binary on target", "not stated")


def false_positive_reason(comment: str) -> str:
    """Where a false positive's signal comes from, by keywords in its ExoFOP comment.

    ``"off target"`` when the comment puts it on another star (``NEB``,
    ``BEB``, ``NPC``, a centroid offset, "off target", "on neighbor", "the
    correct source is TIC ..."); otherwise ``"binary on target"`` when it
    names a binary or binary evidence (``SEB1``, ``SB2``,
    ``EB``, V-shaped, a secondary, odd/even, too large); otherwise ``"not
    stated"``.  The comments are written by hand for observers, not for
    parsing, so this is a coarse sort.
    """
    if _OFF_TARGET.search(comment):
        return "off target"
    if _ON_TARGET.search(comment):
        return "binary on target"
    return "not stated"


def read_toi_comments(path: str | Path) -> dict[str, str]:
    """``{TOI: comment}`` from a CSV with ``TOI`` and ``Comments`` columns.

    ExoFOP's full TOI export has both; ``data/toi_benchmark/toi_comments.csv``
    keeps just those two for the false positives.  Leading ``#`` lines are
    skipped.
    """
    lines = [line for line in Path(path).read_text().splitlines() if line and not line.startswith("#")]
    return {
        (row.get("TOI") or "").strip(): (row.get("Comments") or "").strip()
        for row in csv.DictReader(lines)
        if (row.get("TOI") or "").strip()
    }


def star_label(tois: Iterable[TOI]) -> int | None:
    """Per-star label from all of a star's TOIs (rules in the module docstring)."""
    labels = [t.label for t in tois]
    if not labels:
        return None
    if 1 in labels:
        return 1
    if all(label == 0 for label in labels):
        return 0
    return None


def _snr_key(toi: TOI) -> float:
    return toi.snr if math.isfinite(toi.snr) else -math.inf


def select_benchmark_targets(
    tois: Iterable[TOI],
    sectors: Sequence[int],
    *,
    exclude_tics: Iterable[int] = (),
) -> tuple[list[BenchmarkTarget], dict[str, int]]:
    """Labelled stars observed in one of ``sectors``, one sector per star.

    ``sectors`` is a preference order: each star is fetched in the first of
    them it was observed in, so a star is scored on exactly one light curve.
    ``exclude_tics`` removes stars the model was trained on, which would
    otherwise be scored on data they were fitted to.

    Returns ``(targets, counts)``; ``counts`` says how many stars each rule
    dropped, so the report can account for every TOI host.
    """
    by_star: dict[int, list[TOI]] = {}
    for toi in tois:
        by_star.setdefault(toi.tic, []).append(toi)

    excluded = set(exclude_tics)
    wanted = list(sectors)
    counts = {
        "stars_in_table": len(by_star),
        "unlabelled": 0,
        "not_in_sectors": 0,
        "in_training_set": 0,
        "selected": 0,
        "positives": 0,
        "negatives": 0,
    }
    targets: list[BenchmarkTarget] = []
    for tic in sorted(by_star):
        star_tois = by_star[tic]
        label = star_label(star_tois)
        if label is None:
            counts["unlabelled"] += 1
            continue
        observed = {s for t in star_tois for s in t.sectors}
        sector = next((s for s in wanted if s in observed), None)
        if sector is None:
            counts["not_in_sectors"] += 1
            continue
        if tic in excluded:
            counts["in_training_set"] += 1
            continue
        agreeing = [t for t in star_tois if t.label == label]
        reference = max(agreeing, key=_snr_key)
        targets.append(
            BenchmarkTarget(
                tic=tic, label=label, sector=sector, reference=reference, tois=tuple(star_tois)
            )
        )
        counts["selected"] += 1
        counts["positives" if label == 1 else "negatives"] += 1
    return targets, counts
