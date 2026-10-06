"""Benchmark a trained model against real labels: TOI follow-up dispositions.

Everything else in this project scores the model on labels known by
construction, either fully synthetic or injected into real photometry.  This
module asks the question a user of the vetting tool actually cares about: on
real TESS signals that follow-up observers have since resolved, does the model
keep the planets and reject the false positives?

The model is never retrained here.  It is the one :func:`~transitml.model.train`
fitted on synthetic or injected data, with its operating threshold frozen on
that training split, applied unchanged to the TOI hosts.  Stars the model was
trained on are removed before scoring (:func:`~transitml.data.toi.select_benchmark_targets`).

Read the numbers with the caveats in :mod:`transitml.data.toi` in mind: every
star here is a TOI, so this measures vetting rather than detection, and the
positive rate is set by the catalogue rather than the sky.  For that reason
the report leads with two numbers that do not depend on the class mix:
**recall on confirmed planets** and **the fraction of known false positives
rejected**, both at the frozen threshold.  Average precision is reported too,
against its own chance level.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import BLSConfig, PreprocessConfig
from .data.base import LightCurve, LightCurveSource
from .data.injection import load_curves, save_curves
from .data.loader import Dataset, build_dataset
from .data.toi import BenchmarkTarget
from .evaluate import (
    N_BOOTSTRAP,
    CurveScores,
    bootstrap_indices,
    bootstrap_win_rate,
    period_recovered,
    precision_at_k,
    score_curve,
)
from .model import BASELINES, TrainedModel

#: Bins of the catalogue's TOI SNR.  That SNR comes from every sector the
#: discovery pipeline had, so a single sector here sees less than it says.
TOI_SNR_EDGES: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0, 1e9)


class CurveListSource(LightCurveSource):
    """A fixed list of already-downloaded light curves."""

    def __init__(self, curves: Sequence[LightCurve], name: str = "TOI hosts") -> None:
        self.curves = list(curves)
        self._name = name

    def __len__(self) -> int:
        return len(self.curves)

    def __iter__(self) -> Iterator[LightCurve]:
        return iter(self.curves)

    @property
    def name(self) -> str:
        return self._name


def fetch_benchmark_curves(
    targets: Sequence[BenchmarkTarget],
    *,
    author: str = "TESS-SPOC",
    exposure_time: int | None = 1800,
    n_workers: int = 8,
) -> list[LightCurve]:
    """Download one light curve per target, in its assigned sector.

    Targets MAST has no matching curve for are skipped (the report counts
    them).  Each curve carries its star's label.
    """
    from .data.mast import MASTLightCurveSource

    by_sector: dict[int, list[BenchmarkTarget]] = {}
    for target in targets:
        by_sector.setdefault(target.sector, []).append(target)

    curves: dict[str, LightCurve] = {}
    for sector in sorted(by_sector):
        source = MASTLightCurveSource(
            [(t.target_id, t.label) for t in by_sector[sector]],
            mission="TESS",
            author=author,
            exposure_time=exposure_time,
            sector=sector,
            n_workers=n_workers,
        )
        for lc in source:
            curves.setdefault(lc.target_id, lc)
    return [curves[t.target_id] for t in targets if t.target_id in curves]


def load_or_fetch_curves(
    targets: Sequence[BenchmarkTarget],
    cache: str | Path,
    *,
    author: str = "TESS-SPOC",
    exposure_time: int | None = 1800,
    n_workers: int = 8,
) -> list[LightCurve]:
    """One curve per target, from ``cache`` where possible and MAST otherwise.

    Only targets the cache has never tried are downloaded, so a rerun with
    the same TOI table needs no network, and widening the sector list only
    fetches the new stars.  Targets MAST had nothing for are remembered in a
    ``.tried.json`` file beside the cache so they are not retried every run.
    Labels always come from ``targets``, so a disposition that changed since
    the download is honoured.
    """
    cache = Path(cache)
    tried_path = cache.with_suffix(".tried.json")
    cached = load_curves(cache) if cache.exists() else []
    tried = set(json.loads(tried_path.read_text())) if tried_path.exists() else set()

    def key(target_id: str, sector: Any) -> str:
        return f"{target_id}:{sector}"

    have = {key(lc.target_id, lc.meta.get("sector")): lc for lc in cached}
    tried |= set(have)
    missing = [t for t in targets if key(t.target_id, t.sector) not in tried]
    if missing:
        fetched = fetch_benchmark_curves(
            missing, author=author, exposure_time=exposure_time, n_workers=n_workers
        )
        # Keyed by the sector asked for: that is the sector MAST was searched in.
        sector_of = {t.target_id: t.sector for t in missing}
        for lc in fetched:
            have[key(lc.target_id, sector_of[lc.target_id])] = lc
        tried |= {key(t.target_id, t.sector) for t in missing}
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_curves(list(have.values()), cache)
        tried_path.write_text(json.dumps(sorted(tried)))

    out: list[LightCurve] = []
    for target in targets:
        lc = have.get(key(target.target_id, target.sector))
        if lc is not None:
            out.append(replace(lc, label=target.label))
    return out


@dataclass
class BenchmarkResult:
    """Everything the TOI benchmark reports."""

    sectors: list[int]
    selection: dict[str, int]
    n_without_curve: int
    n_stars: int
    n_planets: int
    chance_average_precision: float
    model: CurveScores
    baselines: list[CurveScores]
    win_rate_vs_best_baseline: float
    threshold: float
    threshold_rule: str
    confusion: dict[str, int]
    planet_recall: float
    false_positive_rejection: float
    precision_at_catalogue_mix: float
    precision_at_k: dict[str, float]
    rejection_by_disposition: dict[str, dict[str, float]]
    search_recovery: dict[str, float]
    planet_recall_given_search: dict[str, float]
    recall_by_toi_snr: list[dict[str, Any]]
    missed_planets: list[dict[str, Any]] = field(default_factory=list)
    accepted_false_positives: list[dict[str, Any]] = field(default_factory=list)
    labels: NDArray[np.int_] | None = None

    @property
    def positive_rate(self) -> float:
        return self.n_planets / self.n_stars if self.n_stars else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sectors": self.sectors,
            "selection": self.selection,
            "n_without_curve": self.n_without_curve,
            "n_stars": self.n_stars,
            "n_planets": self.n_planets,
            "n_false_positives": self.n_stars - self.n_planets,
            "positive_rate": self.positive_rate,
            "chance_average_precision": self.chance_average_precision,
            "model": self.model.to_dict(),
            "baselines": [b.to_dict() for b in self.baselines],
            "bootstrap_win_rate_vs_best_baseline": self.win_rate_vs_best_baseline,
            "operating_point": {
                "threshold": self.threshold,
                "rule": self.threshold_rule,
                "planet_recall": self.planet_recall,
                "false_positive_rejection": self.false_positive_rejection,
                "precision_at_catalogue_mix": self.precision_at_catalogue_mix,
                "confusion_matrix": self.confusion,
            },
            "precision_at_k": self.precision_at_k,
            "rejection_by_disposition": self.rejection_by_disposition,
            "search_recovery": self.search_recovery,
            "planet_recall_given_search": self.planet_recall_given_search,
            "recall_by_toi_snr": self.recall_by_toi_snr,
            "missed_planets": self.missed_planets,
            "accepted_false_positives": self.accepted_false_positives,
        }


def _rate(mask: NDArray[np.bool_], within: NDArray[np.bool_]) -> float:
    n = int(within.sum())
    return float((mask & within).sum() / n) if n else float("nan")


def _row(
    target: BenchmarkTarget, score: float, bls_period: float, recovered: bool
) -> dict[str, Any]:
    ref = target.reference
    return {
        "target_id": target.target_id,
        "toi": ref.toi,
        "disposition": ref.disposition,
        "sector": target.sector,
        "toi_period_days": ref.period,
        "toi_depth_ppm": ref.depth_ppm,
        "toi_snr": ref.snr,
        "tess_mag": ref.tess_mag,
        "bls_period_days": bls_period,
        "period_recovered": recovered,
        "model_score": score,
    }


def benchmark(
    dataset: Dataset,
    targets: Sequence[BenchmarkTarget],
    trained: TrainedModel,
    *,
    sectors: Sequence[int],
    selection: dict[str, int],
    n_without_curve: int = 0,
    top_k: int = 20,
    seed: int = 42,
    n_bootstrap: int = N_BOOTSTRAP,
) -> BenchmarkResult:
    """Score the TOI hosts in ``dataset`` with ``trained`` at its frozen threshold.

    ``dataset`` rows are matched to ``targets`` by target ID, so the dataset
    may hold fewer stars than ``targets`` (those MAST had no curve for).
    """
    by_id = {t.target_id: t for t in targets}
    ids = dataset.meta["target_id"].astype(str).tolist()
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} curve(s) match no benchmark target, e.g. {missing[0]}")
    rows = [by_id[i] for i in ids]
    y = np.array([t.label for t in rows], dtype=int)
    if not np.array_equal(y, dataset.y):
        raise ValueError("dataset labels disagree with the TOI dispositions")
    if y.min() == y.max():
        raise ValueError("the benchmark needs both planets and false positives")

    X = dataset.X
    scores = trained.score(X)
    resamples = bootstrap_indices(len(y), n_bootstrap, seed) if n_bootstrap > 0 else None
    model_curve = score_curve(
        "gradient_boosting", "the trained model, unchanged", y, scores, resamples
    )
    baseline_curves = [
        score_curve(b.name, b.description, y, b.score(X), resamples) for b in BASELINES
    ]
    strongest = max(baseline_curves, key=lambda c: c.average_precision)
    win_rate = (
        bootstrap_win_rate(y, model_curve.scores, strongest.scores, resamples)
        if resamples is not None
        else float("nan")
    )

    kept = scores >= trained.threshold
    planets = y == 1
    negatives = ~planets
    tp = int((kept & planets).sum())
    fp = int((kept & negatives).sum())
    fn = int((~kept & planets).sum())
    tn = int((~kept & negatives).sum())

    bls_period = 10.0 ** dataset.features["log_period"].to_numpy(dtype=float)
    toi_period = np.array([t.reference.period for t in rows], dtype=float)
    recovered = period_recovered(bls_period, toi_period)
    has_period = np.isfinite(toi_period)

    dispositions = np.array([t.reference.disposition for t in rows])
    rejection = {
        str(d): {
            "n": int((dispositions == d).sum()),
            "rejected": _rate(~kept, (dispositions == d)),
        }
        for d in sorted(set(dispositions[negatives]))
    }

    toi_snr = np.array([t.reference.snr for t in rows], dtype=float)
    snr_rows: list[dict[str, Any]] = []
    for lo, hi in zip(TOI_SNR_EDGES[:-1], TOI_SNR_EDGES[1:]):
        sel = planets & (toi_snr >= lo) & (toi_snr < hi)
        snr_rows.append(
            {
                "snr_low": lo,
                "snr_high": hi,
                "n_planets": int(sel.sum()),
                "recall": _rate(kept, sel),
                "search_recovery": _rate(recovered, sel & has_period),
            }
        )

    missed = [
        _row(rows[i], float(scores[i]), float(bls_period[i]), bool(recovered[i]))
        for i in np.flatnonzero(planets & ~kept)
    ]
    missed.sort(key=lambda r: -np.nan_to_num(r["toi_snr"], nan=-1.0))
    accepted = [
        _row(rows[i], float(scores[i]), float(bls_period[i]), bool(recovered[i]))
        for i in np.flatnonzero(negatives & kept)
    ]
    accepted.sort(key=lambda r: -r["model_score"])

    return BenchmarkResult(
        sectors=list(sectors),
        selection=dict(selection),
        n_without_curve=int(n_without_curve),
        n_stars=len(y),
        n_planets=int(planets.sum()),
        chance_average_precision=float(y.mean()),
        model=model_curve,
        baselines=baseline_curves,
        win_rate_vs_best_baseline=win_rate,
        threshold=trained.threshold,
        threshold_rule=trained.threshold_rule,
        confusion={
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        planet_recall=_rate(kept, planets),
        false_positive_rejection=_rate(~kept, negatives),
        precision_at_catalogue_mix=float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        precision_at_k={
            f"model_top{top_k}": precision_at_k(y, scores, top_k),
            **{f"{b.name}_top{top_k}": precision_at_k(y, b.score(X), top_k) for b in BASELINES},
        },
        rejection_by_disposition=rejection,
        search_recovery={
            "planets": _rate(recovered, planets & has_period),
            "false_positives": _rate(recovered, negatives & has_period),
        },
        planet_recall_given_search={
            "period_recovered": _rate(kept, planets & has_period & recovered),
            "period_not_recovered": _rate(kept, planets & has_period & ~recovered),
        },
        recall_by_toi_snr=snr_rows,
        missed_planets=missed,
        accepted_false_positives=accepted,
        labels=y,
    )


def build_benchmark_dataset(
    curves: Sequence[LightCurve],
    *,
    preprocess: PreprocessConfig,
    bls: BLSConfig,
    n_jobs: int = -1,
) -> Dataset:
    """Detrend, search and featurise the TOI host curves exactly as for training."""
    return build_dataset(CurveListSource(curves), preprocess=preprocess, bls=bls, n_jobs=n_jobs)


def _fmt(value: float, spec: str = ".3f") -> str:
    return format(value, spec) if np.isfinite(value) else "n/a"


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Human-readable summary of the TOI benchmark."""
    lines: list[str] = []
    add = lines.append
    sel = result.selection
    add("=" * 72)
    add("REAL-LABEL BENCHMARK: TOI HOSTS WITH FOLLOW-UP DISPOSITIONS")
    add("=" * 72)
    add(f"sectors: {', '.join(str(s) for s in result.sectors)} (one sector per star)")
    add(
        f"stars in TOI table: {sel.get('stars_in_table', 0)}   unlabelled (PC/APC): "
        f"{sel.get('unlabelled', 0)}   not observed in these sectors: "
        f"{sel.get('not_in_sectors', 0)}   in the training set: {sel.get('in_training_set', 0)}"
    )
    add(
        f"selected: {sel.get('selected', 0)}   no light curve at MAST: {result.n_without_curve}   "
        f"scored: {result.n_stars}"
    )
    add(
        f"scored stars: {result.n_planets} planets (CP/KP), "
        f"{result.n_stars - result.n_planets} false positives (FP/FA), "
        f"positive rate {result.positive_rate:.1%}"
    )
    add("")
    add("At the frozen operating threshold (chosen on the training split only)")
    add("-" * 72)
    add(f"  threshold: {result.threshold:.4f}")
    add(f"  confirmed planets kept (recall):       {_fmt(result.planet_recall)}")
    add(f"  known false positives rejected:        {_fmt(result.false_positive_rejection)}")
    for disposition, row in result.rejection_by_disposition.items():
        add(f"    {disposition:<4s} n = {row['n']:<5d} rejected {_fmt(row['rejected'])}")
    add(
        f"  precision at this catalogue's mix:     {_fmt(result.precision_at_catalogue_mix)}"
        "   (not a survey precision; see below)"
    )
    c = result.confusion
    add("")
    add("  Confusion matrix (rows = disposition, cols = prediction)")
    add("                          pred: no planet    pred: planet")
    add(f"    FP / FA                  {c['true_negative']:>10d}      {c['false_positive']:>10d}")
    add(f"    CP / KP                  {c['false_negative']:>10d}      {c['true_positive']:>10d}")
    add("")
    add("Ranking (average precision; chance = positive rate)")
    add("-" * 72)
    add(f"  {'random ranking (chance)':<34s} AP = {result.chance_average_precision:.3f}")
    for b in result.baselines:
        add(
            f"  {'baseline: ' + b.name:<34s} AP = {b.average_precision:.3f}"
            f"  [{_fmt(b.ap_low)}, {_fmt(b.ap_high)}]   (ROC-AUC {b.roc_auc:.3f})"
        )
    m = result.model
    add(
        f"  {'model: gradient boosting':<34s} AP = {m.average_precision:.3f}"
        f"  [{_fmt(m.ap_low)}, {_fmt(m.ap_high)}]   (ROC-AUC {m.roc_auc:.3f})"
    )
    add("  (brackets are 68% bootstrap intervals)")
    add(
        f"  the model beats the best baseline in {_fmt(result.win_rate_vs_best_baseline, '.1%')}"
        " of paired bootstrap resamples"
    )
    for key, value in result.precision_at_k.items():
        add(f"  precision of {key:<30s} {_fmt(value)}")
    add("")
    add("Did the search find the catalogued signal?")
    add("-" * 72)
    add(
        "  BLS period matches the TOI period (or a low-order alias): planets "
        f"{_fmt(result.search_recovery['planets'], '.2f')}, false positives "
        f"{_fmt(result.search_recovery['false_positives'], '.2f')}"
    )
    given = result.planet_recall_given_search
    add(
        f"  planet recall when the period was found: {_fmt(given['period_recovered'], '.2f')}"
        f"   when it was not: {_fmt(given['period_not_recovered'], '.2f')}"
    )
    add("")
    add("  recall by catalogue TOI SNR (all sectors; one sector is scored here)")
    add(f"  {'SNR bin':<16s}{'n':>5s}{'recall':>9s}{'search':>9s}")
    for row in result.recall_by_toi_snr:
        hi = "inf" if row["snr_high"] > 1e8 else f"{row['snr_high']:.0f}"
        label = f"{row['snr_low']:.0f} - {hi}"
        add(
            f"  {label:<16s}{row['n_planets']:>5d}"
            f"{_fmt(row['recall'], '9.2f')}{_fmt(row['search_recovery'], '9.2f')}"
        )
    add("")
    add("Highest-SNR confirmed planets the model rejected")
    add("-" * 72)
    _rows(add, result.missed_planets)
    add("")
    add("Highest-scoring false positives the model kept")
    add("-" * 72)
    _rows(add, result.accepted_false_positives)
    add("")
    add("Reading these numbers")
    add("-" * 72)
    add("  Every star here is a TOI: both classes already passed a TESS pipeline's")
    add("  detection and vetting, so the false positives are the hard ones. The")
    add("  catalogue sets the positive rate, so precision here is not survey")
    add("  precision; recall and false-positive rejection are the transferable numbers.")
    add("=" * 72)
    return "\n".join(lines)


def _rows(add, rows: list[dict[str, Any]], limit: int = 10) -> None:
    if not rows:
        add("  none")
        return
    for row in rows[:limit]:
        add(
            f"  {row['target_id']:<15s} TOI {row['toi']:<8s} {row['disposition']:<3s} "
            f"S{row['sector']:<3d} SNR {_fmt(row['toi_snr'], '6.1f')}  "
            f"P {_fmt(row['toi_period_days'], '7.2f')} d  BLS {_fmt(row['bls_period_days'], '7.2f')} d"
            f"  score {row['model_score']:.3f}"
        )
    if len(rows) > limit:
        add(f"  ... and {len(rows) - limit} more")


def plot_benchmark(result: BenchmarkResult, path: Path) -> Path:
    """Precision-recall on the TOI hosts, and where each class's scores fall."""
    import matplotlib.pyplot as plt

    from .plots import INK_SOFT, NEUTRAL, SERIES, _save, _style

    _style()
    fig, (ax_pr, ax_hist) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    entries = [(result.model, SERIES[0], "gradient boosting")]
    entries += [(b, colour, f"baseline: {b.name}") for b, colour in zip(result.baselines, SERIES[1:])]
    for curve, colour, label in entries:
        ax_pr.plot(
            curve.recall, curve.precision, lw=2.0, color=colour,
            label=f"{label}  (AP = {curve.average_precision:.3f})",
        )
    chance = result.chance_average_precision
    ax_pr.axhline(chance, lw=1.4, ls="--", color=NEUTRAL)
    ax_pr.text(
        0.015, chance - 0.02, f"random ranking (AP = {chance:.3f})",
        ha="left", va="top", color=INK_SOFT, fontsize=8.5,
    )
    ax_pr.set_xlim(0, 1.02)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.set_xlabel("recall (fraction of confirmed planets kept)")
    ax_pr.set_ylabel("precision at this catalogue's mix")
    ax_pr.set_title(
        f"{result.n_stars} TOI hosts ({result.n_planets} CP/KP, "
        f"{result.n_stars - result.n_planets} FP/FA)",
        loc="left",
    )
    ax_pr.legend(loc="best")

    scores, labels = result.model.scores, result.labels
    if scores is not None and labels is not None:
        bins = np.linspace(0.0, 1.0, 26)
        for label, colour, name in ((1, SERIES[0], "CP / KP"), (0, SERIES[1], "FP / FA")):
            ax_hist.hist(
                scores[labels == label], bins=bins, color=colour,
                histtype="step", lw=2.0, label=name,
            )
        ax_hist.axvline(result.threshold, color=NEUTRAL, lw=1.4, ls="--")
        ax_hist.text(
            result.threshold, ax_hist.get_ylim()[1] * 0.97,
            f" threshold {result.threshold:.2f}",
            ha="left", va="top", color=INK_SOFT, fontsize=8.5,
        )
        ax_hist.set_xlabel("model score")
        ax_hist.set_ylabel("stars")
        ax_hist.set_title(
            f"kept {result.planet_recall:.0%} of planets, "
            f"rejected {result.false_positive_rejection:.0%} of false positives",
            loc="left",
        )
        ax_hist.legend(loc="upper center")
    fig.tight_layout()
    return _save(fig, path)
