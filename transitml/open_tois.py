"""Ranking the TOIs nobody has resolved yet.

The TOI benchmark (:mod:`transitml.benchmark`) scores resolved TOIs, whose
follow-up dispositions say how well the model separates planets from false
positives.  This scores the open ones: every star with a PC or APC TOI and no
confirmed or known planet (:func:`~transitml.data.toi.select_open_targets`),
searched and scored exactly as a benchmark star is, by a model trained on
every resolved host of the same sectors.

What the ranking is, and is not:

* **There is no average precision here.**  Nobody knows yet which of these
  are planets, so the list is a ranking and a probability, nothing measured.
  How well the same model ranks is the benchmark's and its cross-validation's
  to say.
* **The probability is calibrated to resolved TOIs.**  ``P(planet)`` is the
  model's score through the Platt calibration fitted on its training hosts,
  at their planet rate.  Open TOIs are not drawn like them (the report
  compares magnitude, depth and SNR), so it is the chance a resolved TOI that
  looked like this one turned out a planet.
* **The centroid test runs on the top of the list only** (``n_centroids``),
  where a dip placed on another star matters most; a star without the test is
  not cleared by it.

One check is possible all the same.  Dispositions on ExoFOP lag the
literature, so some open TOIs are already confirmed planets in the NASA
Exoplanet Archive (:func:`read_archive_planets`).  Those labels were never
seen in training, and :meth:`OpenRanking.archive_check` asks how far up the
list they sit, against orderings by the quantities follow-up itself favours.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .data.loader import Dataset
from .data.toi import OPEN_DISPOSITIONS, BenchmarkTarget
from .evaluate import N_BOOTSTRAP, period_recovered
from .model import TrainedModel
from .toi_training import TOI_RELIABILITY_EDGES

#: Columns of the ranked table, in order.
OPEN_COLUMNS: tuple[str, ...] = (
    "rank",
    "target_id",
    "toi",
    "open_tois",
    "disposition",
    "sector",
    "author",
    "tess_mag",
    "toi_period_days",
    "toi_depth_ppm",
    "toi_snr",
    "bls_period_days",
    "period_recovered",
    "model_score",
    "probability",
    "kept",
    "odd_even_sigma",
    "secondary_sigma",
    "centroid_status",
    "centroid_offset_pixels",
    "centroid_offset_sigma",
    "centroid_flagged",
    "archive_planet",
)

#: An archive planet is the TOI when their periods differ by less than this
#: fraction.  Only the same period counts: a planet at twice or half the TOI's
#: period, or another planet of the star, is left out.
ARCHIVE_PERIOD_TOLERANCE: float = 0.01

#: Orderings the archive check compares the model with: the TOI's SNR and
#: depth and the star's brightness, which decide what follow-up can confirm.
ARCHIVE_ORDERINGS: tuple[tuple[str, str], ...] = (
    ("model", "model score"),
    ("toi_snr", "TOI SNR"),
    ("toi_depth_ppm", "TOI depth"),
    ("brightness", "brightness (-Tmag)"),
)

#: Quantities the report compares between the open stars and the resolved
#: ones the model learned from, as medians.
COMPARED: tuple[tuple[str, str], ...] = (
    ("tess_mag", "TESS magnitude"),
    ("toi_depth_ppm", "TOI depth (ppm)"),
    ("toi_snr", "TOI SNR"),
    ("toi_period_days", "TOI period (days)"),
)


@dataclass
class OpenRanking:
    """The open TOI hosts of some sectors, ranked by a model trained on the resolved ones."""

    sectors: list[int]
    selection: dict[str, int]
    n_without_curve: int
    threshold: float
    threshold_rule: str
    threshold_probability: float
    training_planet_rate: float
    #: One row per scored star (:data:`OPEN_COLUMNS`), best first.
    rows: list[dict[str, Any]]
    #: ``{quantity: {"open": median, "resolved": median}}`` (:data:`COMPARED`).
    comparison: dict[str, dict[str, float]]
    n_resolved: int
    n_centroids: int = 0
    stitched: bool = False
    joined: str | None = None
    #: Centroid statuses among the stars tested.
    centroid_status: dict[str, int] = field(default_factory=dict)
    #: The archive planet table the rows were matched against, if any.
    archive: str | None = None

    @property
    def n_stars(self) -> int:
        return len(self.rows)

    @property
    def n_kept(self) -> int:
        return sum(1 for r in self.rows if r["kept"])

    def probability_bins(self) -> list[dict[str, Any]]:
        """How many stars fall in each bin of :data:`TOI_RELIABILITY_EDGES`."""
        p = np.array([r["probability"] for r in self.rows], dtype=float)
        edges = TOI_RELIABILITY_EDGES
        out = []
        for i, (lo, hi) in enumerate(pairwise(edges)):
            last = i == len(edges) - 2
            sel = (p >= lo) & ((p <= hi) if last else (p < hi))
            out.append({"low": lo, "high": hi, "n": int(sel.sum())})
        return out

    def by_disposition(self) -> dict[str, dict[str, float]]:
        """Stars, median ``P(planet)`` and share above threshold, per open disposition.

        APC (an ambiguous candidate, where follow-up left doubt) against PC is
        the one hint of the truth the open list carries.
        """
        out: dict[str, dict[str, float]] = {}
        for disposition in sorted({r["disposition"] for r in self.rows}, key=lambda d: (d != "PC", d)):
            rows = [r for r in self.rows if r["disposition"] == disposition]
            out[disposition] = {
                "n": len(rows),
                "median_probability": _median([r["probability"] for r in rows]),
                "kept_fraction": sum(r["kept"] for r in rows) / len(rows),
            }
        return out

    def archive_check(self, n_resamples: int = N_BOOTSTRAP, seed: int = 0) -> dict[str, Any] | None:
        """Where the open TOIs the archive already lists as planets sit in the ranking.

        ROC-AUC of the archive planets against the rest, for the model's
        score and for :data:`ARCHIVE_ORDERINGS`; the share above the
        threshold; and the share of ``n_resamples`` paired resamples (each
        group resampled on its own) in which the model's AUC beats the TOI
        SNR's.  The rest are not all false positives, so the AUC is a floor
        on how well the model separates, not a measure of it.
        """
        if self.archive is None:
            return None
        confirmed = np.array([r["archive_planet"] is not None for r in self.rows])
        n = int(confirmed.sum())
        out: dict[str, Any] = {"archive": self.archive, "n_confirmed": n, "n_rest": len(confirmed) - n}
        if n == 0 or n == len(confirmed):
            return out
        orderings = {
            "model": [r["model_score"] for r in self.rows],
            "toi_snr": [r["toi_snr"] for r in self.rows],
            "toi_depth_ppm": [r["toi_depth_ppm"] for r in self.rows],
            "brightness": [-r["tess_mag"] for r in self.rows],
        }
        values = {k: _filled(v) for k, v in orderings.items()}
        y = confirmed.astype(int)
        kept = np.array([r["kept"] for r in self.rows])
        probability = np.array([r["probability"] for r in self.rows], dtype=float)
        rng = np.random.default_rng(seed)
        pos, neg = np.flatnonzero(confirmed), np.flatnonzero(~confirmed)
        wins = 0
        for _ in range(n_resamples):
            idx = np.r_[rng.choice(pos, pos.size), rng.choice(neg, neg.size)]
            wins += roc_auc_score(y[idx], values["model"][idx]) > roc_auc_score(
                y[idx], values["toi_snr"][idx]
            )
        out.update(
            {
                "kept_confirmed": float(kept[confirmed].mean()),
                "kept_rest": float(kept[~confirmed].mean()),
                "median_probability_confirmed": float(np.median(probability[confirmed])),
                "median_probability_rest": float(np.median(probability[~confirmed])),
                "auc": {k: float(roc_auc_score(y, v)) for k, v in values.items()},
                "model_ahead_of_snr": wins / n_resamples,
                "n_resamples": n_resamples,
            }
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "sectors": self.sectors,
            "stitched": self.stitched,
            "joined": self.joined,
            "selection": self.selection,
            "n_without_curve": self.n_without_curve,
            "n_stars": self.n_stars,
            "n_kept": self.n_kept,
            "threshold": self.threshold,
            "threshold_rule": self.threshold_rule,
            "threshold_probability": self.threshold_probability,
            "training_planet_rate": self.training_planet_rate,
            "n_resolved": self.n_resolved,
            "comparison": self.comparison,
            "probability_bins": self.probability_bins(),
            "by_disposition": self.by_disposition(),
            "archive_check": self.archive_check(),
            "n_centroids": self.n_centroids,
            "centroid_status": self.centroid_status,
            "stars": self.rows,
        }


def _filled(values: Sequence[float]) -> np.ndarray:
    """Scores with missing values put below every finite one, as :func:`~transitml.evaluate.score_curve` does."""
    a = np.asarray(values, dtype=float)
    ok = np.isfinite(a)
    return np.where(ok, a, (a[ok].min() if ok.any() else 0.0) - 1.0)


def read_archive_planets(path: Path | str) -> dict[str, list[tuple[str, float]]]:
    """Confirmed planets by host star, from the NASA Exoplanet Archive's planet table.

    A CSV with ``pl_name``, ``tic_id`` (``"TIC n"``) and ``pl_orbper``
    columns, such as ``pscomppars`` from the archive's TAP service; rows
    without a TIC ID are skipped and a missing period is NaN.
    """
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            tic = (row.get("tic_id") or "").strip()
            if not tic:
                continue
            try:
                period = float(row.get("pl_orbper") or "nan")
            except ValueError:
                period = float("nan")
            out[tic].append((row["pl_name"], period))
    return dict(out)


def archive_match(planets: Sequence[tuple[str, float]], period: float) -> str | None:
    """The planet in ``planets`` at the TOI's ``period`` (:data:`ARCHIVE_PERIOD_TOLERANCE`), if any."""
    if not np.isfinite(period) or period <= 0:
        return None
    for name, p in planets:
        if np.isfinite(p) and abs(p / period - 1.0) < ARCHIVE_PERIOD_TOLERANCE:
            return name
    return None


def _median(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def rank_open_hosts(
    dataset: Dataset,
    targets: Sequence[BenchmarkTarget],
    trained: TrainedModel,
    *,
    sectors: Sequence[int],
    selection: dict[str, int],
    resolved: Sequence[BenchmarkTarget],
    n_without_curve: int = 0,
    centroids: Sequence[dict[str, Any] | None] | None = None,
    stitched: bool = False,
    joined: str | None = None,
    archive: Mapping[str, Sequence[tuple[str, float]]] | None = None,
    archive_name: str | None = None,
) -> OpenRanking:
    """Score the open TOI hosts in ``dataset`` with ``trained`` and rank them.

    ``dataset`` rows are matched to ``targets`` by target ID, as in
    :func:`~transitml.benchmark.benchmark`; its labels are never read.
    ``resolved`` are the hosts the model was trained on, for the comparison.
    With ``centroids`` (one :func:`~transitml.benchmark.centroid_tests` entry
    per dataset row, ``None`` where the test was not run), each row carries
    the test's status and whether it placed the dip off target.  With
    ``archive`` (:func:`read_archive_planets`), each row names the archive
    planet at its TOI's period, if there is one; ``archive_name`` labels the
    table in the report.
    """
    by_id = {t.target_id: t for t in targets}
    ids = dataset.meta["target_id"].astype(str).tolist()
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} curve(s) match no open target, e.g. {missing[0]}")
    if centroids is not None and len(centroids) != len(ids):
        raise ValueError(f"{len(centroids)} centroid results for {len(ids)} stars")
    rows = [by_id[i] for i in ids]

    inputs = dataset.inputs(tuple(trained.feature_names))
    scores = trained.score(inputs)
    probability = trained.probability(inputs)
    kept = scores >= trained.threshold
    bls_period = 10.0 ** dataset.features["log_period"].to_numpy(dtype=float)
    toi_period = np.array([t.reference.period for t in rows], dtype=float)
    recovered = period_recovered(bls_period, toi_period)
    feats = dataset.features
    authors = dataset.meta["author"].tolist() if "author" in dataset.meta else [None] * len(ids)

    out: list[dict[str, Any]] = []
    for i, target in enumerate(rows):
        ref = target.reference
        test = centroids[i] if centroids is not None else None
        placed = test is not None and test["status"] == "ok"
        out.append(
            {
                "target_id": target.target_id,
                "toi": ref.toi,
                "open_tois": " ".join(
                    t.toi for t in target.tois if t.disposition in OPEN_DISPOSITIONS
                ),
                "disposition": ref.disposition,
                "sector": target.sector,
                "author": authors[i] if isinstance(authors[i], str) else None,
                "tess_mag": ref.tess_mag,
                "toi_period_days": ref.period,
                "toi_depth_ppm": ref.depth_ppm,
                "toi_snr": ref.snr,
                "bls_period_days": float(bls_period[i]),
                "period_recovered": bool(recovered[i]),
                "model_score": float(scores[i]),
                "probability": float(probability[i]),
                "kept": bool(kept[i]),
                "odd_even_sigma": float(feats["odd_even_sigma"].iloc[i]),
                "secondary_sigma": float(feats["secondary_sigma"].iloc[i]),
                "centroid_status": None if test is None else str(test["status"]),
                "centroid_offset_pixels": (
                    float(test["offset_distance_pixels"]) if placed else float("nan")
                ),
                "centroid_offset_sigma": float(test["offset_sigma"]) if placed else float("nan"),
                "centroid_flagged": None if test is None else bool(test["significant"]),
                "archive_planet": (
                    None
                    if archive is None
                    else archive_match(archive.get(target.target_id, ()), ref.period)
                ),
            }
        )
    out.sort(key=lambda r: -r["model_score"])
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    out = [{name: row[name] for name in OPEN_COLUMNS} for row in out]

    resolved_rows = {
        "tess_mag": [t.reference.tess_mag for t in resolved],
        "toi_depth_ppm": [t.reference.depth_ppm for t in resolved],
        "toi_snr": [t.reference.snr for t in resolved],
        "toi_period_days": [t.reference.period for t in resolved],
    }
    comparison = {
        name: {"open": _median([r[name] for r in out]), "resolved": _median(resolved_rows[name])}
        for name, _ in COMPARED
    }
    tested = [c for c in centroids or () if c is not None]
    return OpenRanking(
        sectors=list(sectors),
        selection=dict(selection),
        n_without_curve=int(n_without_curve),
        threshold=float(trained.threshold),
        threshold_rule=trained.threshold_rule,
        threshold_probability=float(trained.threshold_probability),
        training_planet_rate=float(trained.calibration.train_positive_rate),
        rows=out,
        comparison=comparison,
        n_resolved=len(resolved),
        n_centroids=len(tested),
        stitched=stitched,
        joined=joined,
        centroid_status=dict(Counter(str(c["status"]) for c in tested)),
        archive=None if archive is None else (archive_name or "archive planet table"),
    )


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not np.isfinite(value):
        return "-"
    return format(value, spec)


def _sector_text(sectors: Sequence[int]) -> str:
    return f"{min(sectors)} to {max(sectors)}" if len(sectors) > 1 else str(sectors[0])


def format_open_report(result: OpenRanking, top: int = 25) -> str:
    """The ranking as text: who was scored, how the probabilities spread, the top of the list."""
    lines: list[str] = []
    add = lines.append
    rule = "=" * 72
    add(rule)
    add(f"OPEN TOIs OF SECTORS {_sector_text(result.sectors).upper()}, RANKED")
    add(rule)
    sel = result.selection
    add(
        f"stars in TOI table: {sel.get('stars_in_table', 0)}   with a resolved planet or no open "
        f"TOI: {sel.get('not_open', 0)}   not observed in these sectors: "
        f"{sel.get('not_in_sectors', 0)}"
    )
    add(
        f"open TOI hosts: {sel.get('selected', 0)}   no light curve at MAST: "
        f"{result.n_without_curve}   scored: {result.n_stars}"
    )
    if result.stitched:
        add("  each searched on every one of these sectors it was observed in, joined")
    if result.joined:
        add(f"  with every sector of {result.joined} it was observed in added to its curve")
    add(
        f"model: trained on the {result.n_resolved} resolved hosts of the same sectors "
        f"(planet rate {result.training_planet_rate:.1%})"
    )
    add("")
    add("No average precision: nobody knows yet which of these are planets.")
    add("P(planet) is calibrated on resolved TOIs, at their planet rate.")
    add("")
    add("Open hosts against the resolved hosts the model learned from (medians)")
    add("-" * 72)
    for name, label in COMPARED:
        c = result.comparison[name]
        add(f"  {label:<22s} open {_fmt(c['open'], '.4g'):>9s}   resolved {_fmt(c['resolved'], '.4g'):>9s}")
    add("")
    recovered = sum(1 for r in result.rows if r["period_recovered"])
    with_period = sum(1 for r in result.rows if np.isfinite(r["toi_period_days"]))
    add(
        f"search found the TOI's period: {recovered} of {with_period} with a catalogued period "
        f"({recovered / with_period:.0%})" if with_period else "no catalogued periods"
    )
    add(
        f"above the operating threshold ({result.threshold_rule}): "
        f"{result.n_kept} of {result.n_stars} ({result.n_kept / max(result.n_stars, 1):.0%})"
    )
    for disposition, d in result.by_disposition().items():
        add(
            f"  {disposition:<4s} {d['n']:>5d} stars   median P(planet) "
            f"{d['median_probability']:.2f}   above threshold {d['kept_fraction']:.0%}"
        )
    add("")
    add("P(planet)        stars")
    for b in result.probability_bins():
        add(f"  {b['low']:.1f} to {b['high']:.1f}      {b['n']:>5d}")
    if result.n_centroids:
        flagged = sum(1 for r in result.rows if r["centroid_flagged"])
        status = ", ".join(f"{k} {v}" for k, v in sorted(result.centroid_status.items()))
        add("")
        add(
            f"centroid test on the top {result.n_centroids} with a pixel file: {flagged} dips "
            f"placed off target ({status})"
        )
    check = result.archive_check()
    if check is not None:
        add("")
        add(f"Already confirmed in the literature ({check['archive']})")
        add("-" * 72)
        add(
            f"  {check['n_confirmed']} of the {result.n_stars} are a confirmed planet at the TOI's "
            f"period (within {ARCHIVE_PERIOD_TOLERANCE:.0%})"
        )
        if "auc" in check:
            add(
                f"  above the threshold: {check['kept_confirmed']:.0%} of them, "
                f"{check['kept_rest']:.0%} of the rest; median P(planet) "
                f"{check['median_probability_confirmed']:.2f} against "
                f"{check['median_probability_rest']:.2f}"
            )
            add("  ROC-AUC against the rest (the rest are not all false positives):")
            for name, label in ARCHIVE_ORDERINGS:
                add(f"    {label:<20s} {check['auc'][name]:.3f}")
            add(
                f"  model ahead of the TOI SNR in {check['model_ahead_of_snr']:.0%} of "
                f"{check['n_resamples']} paired resamples"
            )
    add("")
    add(f"Top {min(top, result.n_stars)}")
    add("-" * 72)
    add(
        f"{'rank':>4s} {'TOI':>8s} {'target':>15s} {'Tmag':>5s} {'P_toi':>8s} {'P_bls':>8s} "
        f"{'depth':>7s} {'P(pl)':>6s} {'centroid':>9s}  archive"
    )
    for r in result.rows[:top]:
        centroid = (
            "-"
            if r["centroid_status"] is None
            else "off" if r["centroid_flagged"] else "on" if r["centroid_status"] == "ok" else "n/a"
        )
        add(
            f"{r['rank']:>4d} {r['toi']:>8s} {r['target_id']:>15s} {_fmt(r['tess_mag'], '.1f'):>5s} "
            f"{_fmt(r['toi_period_days'], '.3f'):>8s} {_fmt(r['bls_period_days'], '.3f'):>8s} "
            f"{_fmt(r['toi_depth_ppm'], '.0f'):>7s} {r['probability']:>6.2f} {centroid:>9s}"
            f"  {r['archive_planet'] or ''}".rstrip()
        )
    return "\n".join(lines)


def write_open_ranking(result: OpenRanking, results_dir: Path) -> list[Path]:
    """``open_tois.csv`` (the ranked table), ``open_tois.json`` and ``open_tois.txt``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    table = results_dir / "open_tois.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OPEN_COLUMNS))
        writer.writeheader()
        for row in result.rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    payload = results_dir / "open_tois.json"
    payload.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    report = results_dir / "open_tois.txt"
    report.write_text(format_open_report(result) + "\n")
    return [table, payload, report]
