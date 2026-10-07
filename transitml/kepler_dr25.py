"""Build the Kepler DR25 training set and score a first model on it.

    python -m transitml.kepler_dr25                       # 1000 TCEs per class
    python -m transitml.kepler_dr25 --per-class 300       # a quicker sample
    python -m transitml.kepler_dr25 --all --n-jobs 16     # all 34,032 TCEs

Three steps:

1. **Labels.**  The DR25 TCE and KOI tables from the NASA Exoplanet Archive,
   joined and labelled PC / AFP / NTP (:mod:`transitml.data.kepler`).  Cached
   as ``data/kepler_dr25/dr25_tce_labels.csv``, which is committed.
2. **Views.**  For each star, every long-cadence quarter from MAST, stitched,
   detrended with all of its TCEs masked, and folded into five views per TCE
   (:mod:`transitml.views`).  Saved as one ``training_set.npz``; the
   downloaded curves are cached per star, so a rerun is offline.
3. **A first model.**  Gradient boosting on the binned views, split by star
   so no star is in both halves, against the Kepler pipeline's own detection
   statistic (MES) as the baseline.  This is the number a CNN on the same
   views has to beat.

The sample is balanced by class (``--per-class``), which is what makes a few
thousand TCEs informative about the rare classes, but it also means precision
on the sample is not precision on the catalogue.  The report gives both: the
measured precision, and the precision the same per-class pass rates would give
at the catalogue's own class mix (4,034 PC : 3,025 AFP : 26,973 NTP).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from .data.kepler import (
    CLASSES,
    KeplerTCE,
    class_counts,
    load_or_download_star,
    load_or_fetch_catalogue,
    stratified_sample,
)
from .evaluate import fast_average_precision
from .views import (
    VIEW_NAMES,
    Ephemeris,
    ViewConfig,
    detrend_masked,
    in_transit_mask,
    knot_spacing_for,
    make_views,
)

DEFAULT_CATALOGUE = Path("data/kepler_dr25/dr25_tce_labels.csv")
DEFAULT_RESULTS = Path("results/kepler_dr25")
DEFAULT_FIGURES = Path("figures/kepler_dr25")

#: Per-TCE scalars stored beside the views.  ``mes`` and ``depth_ppm`` come
#: from the Kepler pipeline; the rest are measured here.
SCALAR_NAMES: tuple[str, ...] = (
    "period_days",
    "duration_hours",
    "depth_ppm",
    "mes",
    "depth_scale",
    "scatter",
    "local_empty_fraction",
    "n_cadences",
)

#: Target recall on planet candidates for the operating threshold, chosen on
#: the training split's out-of-fold scores and then frozen.
TARGET_RECALL = 0.9


# --------------------------------------------------------------------------
# The training set
# --------------------------------------------------------------------------
@dataclass
class TrainingSet:
    """Views, labels and scalars for a list of TCEs, row-aligned."""

    tce_ids: NDArray[np.str_]
    kepids: NDArray[np.int64]
    classes: NDArray[np.str_]
    labels: NDArray[np.int64]
    views: dict[str, NDArray[np.float32]]
    scalars: dict[str, NDArray[np.float64]]
    failures: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.labels.size)

    def subset(self, index: NDArray[np.int_]) -> "TrainingSet":
        return TrainingSet(
            tce_ids=self.tce_ids[index],
            kepids=self.kepids[index],
            classes=self.classes[index],
            labels=self.labels[index],
            views={k: v[index] for k, v in self.views.items()},
            scalars={k: v[index] for k, v in self.scalars.items()},
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            tce_ids=self.tce_ids,
            kepids=self.kepids,
            classes=self.classes,
            labels=self.labels,
            failures=np.array(json.dumps(self.failures)),
            **{f"view_{k}": v for k, v in self.views.items()},
            **{f"scalar_{k}": v for k, v in self.scalars.items()},
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainingSet":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                tce_ids=data["tce_ids"],
                kepids=data["kepids"],
                classes=data["classes"],
                labels=data["labels"],
                views={k: data[f"view_{k}"] for k in VIEW_NAMES},
                scalars={k: data[f"scalar_{k}"] for k in SCALAR_NAMES},
                failures=json.loads(str(data["failures"])),
            )


def star_records(
    kepid: int,
    tces: Sequence[KeplerTCE],
    cache_dir: str | Path,
    config: ViewConfig,
    all_tces: Sequence[KeplerTCE] = (),
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Views for the requested TCEs of one star.  Returns ``(records, failures)``.

    The trend is fitted with **every** DR25 TCE on the star masked
    (``all_tces``), not only the requested ones, so an unsampled planet does
    not bend the spline either.  Failures (no light curve, no cadence near the
    transit) are returned by TCE id rather than raised, so one bad star does
    not stop a 17,000-star run.
    """
    try:
        lc = load_or_download_star(kepid, cache_dir)
    except Exception as exc:  # network or FITS trouble: record and move on
        return [], {t.tce_id: f"download failed: {exc}" for t in tces}
    if lc is None:
        return [], {t.tce_id: "no long-cadence light curve at MAST" for t in tces}

    def ephemeris(t: KeplerTCE) -> Ephemeris:
        return Ephemeris(t.period_days, t.epoch_bkjd, t.duration_days)

    masked = [ephemeris(t) for t in (all_tces or tces)]
    mask = in_transit_mask(lc.time, masked, config.mask_half_width)
    flux = detrend_masked(
        lc.time,
        lc.flux,
        mask,
        knot_spacing_for(masked, config),
        config.gap_threshold_days,
    )

    records, failures = [], {}
    for tce in tces:
        try:
            views = make_views(lc.time, flux, ephemeris(tce), config)
        except ValueError as exc:
            failures[tce.tce_id] = str(exc)
            continue
        records.append(
            {
                "tce_id": tce.tce_id,
                "kepid": tce.kepid,
                "class": tce.tce_class,
                "label": tce.label,
                "views": views.as_dict(),
                "scalars": {
                    "period_days": tce.period_days,
                    "duration_hours": tce.duration_hours,
                    "depth_ppm": tce.depth_ppm,
                    "mes": tce.mes,
                    "depth_scale": views.depth_scale,
                    "scatter": views.scatter,
                    "local_empty_fraction": views.local_empty_fraction,
                    "n_cadences": float(lc.n_cadences),
                },
            }
        )
    return records, failures


def build_training_set(
    tces: Sequence[KeplerTCE],
    cache_dir: str | Path,
    *,
    config: ViewConfig | None = None,
    catalogue: Sequence[KeplerTCE] = (),
    n_jobs: int = 8,
    verbose: int = 0,
) -> TrainingSet:
    """Download, detrend and fold every TCE in ``tces``, one star per task.

    ``catalogue`` (the full DR25 list) supplies the other TCEs on each star
    for the trend mask; without it only the requested TCEs are masked.
    """
    config = config or ViewConfig()
    by_star: dict[int, list[KeplerTCE]] = {}
    for tce in tces:
        by_star.setdefault(tce.kepid, []).append(tce)
    on_star: dict[int, list[KeplerTCE]] = {}
    for tce in catalogue:
        if tce.kepid in by_star:
            on_star.setdefault(tce.kepid, []).append(tce)

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(star_records)(kepid, star, cache_dir, config, on_star.get(kepid, ()))
        for kepid, star in sorted(by_star.items())
    )
    records = [r for recs, _ in results for r in recs]
    failures = {k: v for _, fails in results for k, v in fails.items()}
    bins = {"global": config.global_bins}
    return TrainingSet(
        tce_ids=np.array([r["tce_id"] for r in records], dtype=str),
        kepids=np.array([r["kepid"] for r in records], dtype=np.int64),
        classes=np.array([r["class"] for r in records], dtype=str),
        labels=np.array([r["label"] for r in records], dtype=np.int64),
        views={
            name: (
                np.stack([r["views"][name] for r in records])
                if records
                else np.empty((0, bins.get(name, config.local_bins)), dtype=np.float32)
            )
            for name in VIEW_NAMES
        },
        scalars={
            name: np.array([r["scalars"][name] for r in records], dtype=np.float64)
            for name in SCALAR_NAMES
        },
        failures=failures,
    )


# --------------------------------------------------------------------------
# A first model
# --------------------------------------------------------------------------
def view_features(ts: TrainingSet, global_bins: int = 201) -> NDArray[np.float64]:
    """Flat model input: the four local views, the global view block-averaged
    to ``global_bins`` bins, and log period and duration.

    Nothing from the Kepler pipeline's detection statistics goes in, so the
    model has to work from the light curve's shape alone.
    """
    glob = ts.views["global"].astype(np.float64)
    edges = np.linspace(0, glob.shape[1], global_bins + 1).astype(int)
    coarse = np.stack(
        [glob[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1
    )
    parts = [ts.views[name].astype(np.float64) for name in ("local", "odd", "even", "secondary")]
    parts.append(coarse)
    parts.append(np.log10(ts.scalars["period_days"])[:, None])
    parts.append(np.log10(ts.scalars["duration_hours"])[:, None])
    return np.hstack(parts)


def group_split(
    kepids: NDArray[np.int64], test_size: float, seed: int
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Train/test row indices with every star's TCEs on one side only."""
    stars = np.unique(kepids)
    rng = np.random.default_rng(seed)
    test_stars = rng.choice(stars, size=max(1, int(round(test_size * stars.size))), replace=False)
    is_test = np.isin(kepids, test_stars)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def _build_classifier(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=seed,
    )


def _out_of_fold(X, y, groups, seed: int, n_folds: int = 5) -> NDArray[np.float64]:
    from sklearn.model_selection import GroupKFold

    scores = np.zeros(y.size)
    for train, test in GroupKFold(n_splits=n_folds).split(X, y, groups):
        model = _build_classifier(seed).fit(X[train], y[train])
        scores[test] = model.predict_proba(X[test])[:, 1]
    return scores


def threshold_for_recall(y: NDArray[np.int_], scores: NDArray[np.float64], recall: float) -> float:
    """Highest threshold that keeps at least ``recall`` of the positives."""
    positive = np.sort(scores[y == 1])[::-1]
    if positive.size == 0:
        return 0.5
    k = int(np.ceil(recall * positive.size)) - 1
    return float(positive[min(max(k, 0), positive.size - 1)])


def _pass_rates(classes, flagged) -> dict[str, float]:
    return {
        name: float(flagged[classes == name].mean()) if (classes == name).any() else float("nan")
        for name in CLASSES
    }


def catalogue_precision(pass_rates: dict[str, float], counts: dict[str, int]) -> float:
    """Precision the per-class pass rates would give at the catalogue's class mix."""
    kept = {name: pass_rates[name] * counts[name] for name in CLASSES}
    total = sum(kept.values())
    return float(kept["PC"] / total) if total > 0 else float("nan")


def _bootstrap_ap(y, scores, seed: int, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, y.size, y.size)
        if y[idx].any():
            values.append(fast_average_precision(y[idx], scores[idx]))
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass
class DR25Result:
    n_tces: int
    n_stars: int
    n_failed: int
    class_counts_sample: dict[str, int]
    class_counts_catalogue: dict[str, int]
    n_train: int
    n_test: int
    test_class_counts: dict[str, int]
    average_precision: float
    average_precision_ci: tuple[float, float]
    mes_average_precision: float
    chance_average_precision: float
    roc_auc: float
    threshold: float
    target_recall: float
    test_pass_rates: dict[str, float]
    test_precision: float
    catalogue_precision: float
    mes_pass_rates: dict[str, float]
    mes_threshold: float
    curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    test_scores: list[float] = field(default_factory=list)
    test_labels: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k not in ("curves", "test_scores", "test_labels")}
        out["average_precision_ci"] = list(self.average_precision_ci)
        return out


def evaluate_training_set(
    ts: TrainingSet,
    catalogue_counts: dict[str, int],
    *,
    seed: int = 0,
    test_size: float = 0.3,
    target_recall: float = TARGET_RECALL,
) -> DR25Result:
    """Fit on a star-grouped training split, score the held-out stars."""
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    X, y = view_features(ts), ts.labels
    train, test = group_split(ts.kepids, test_size, seed)
    oof = _out_of_fold(X[train], y[train], ts.kepids[train], seed)
    threshold = threshold_for_recall(y[train], oof, target_recall)
    model = _build_classifier(seed).fit(X[train], y[train])
    scores = model.predict_proba(X[test])[:, 1]

    mes = np.nan_to_num(ts.scalars["mes"], nan=0.0)
    mes_threshold = threshold_for_recall(y[train], mes[train], target_recall)

    yt, ct = y[test], ts.classes[test]
    flagged = scores >= threshold
    pass_rates = _pass_rates(ct, flagged)
    mes_rates = _pass_rates(ct, mes[test] >= mes_threshold)

    curves = {}
    for name, s in (("model", scores), ("mes", mes[test])):
        p, r, _ = precision_recall_curve(yt, s)
        curves[name] = {"precision": p.tolist(), "recall": r.tolist()}

    return DR25Result(
        n_tces=len(ts),
        n_stars=int(np.unique(ts.kepids).size),
        n_failed=len(ts.failures),
        class_counts_sample={c: int((ts.classes == c).sum()) for c in CLASSES},
        class_counts_catalogue=dict(catalogue_counts),
        n_train=int(train.size),
        n_test=int(test.size),
        test_class_counts={c: int((ct == c).sum()) for c in CLASSES},
        average_precision=fast_average_precision(yt, scores),
        average_precision_ci=_bootstrap_ap(yt, scores, seed),
        mes_average_precision=fast_average_precision(yt, mes[test]),
        chance_average_precision=float(yt.mean()),
        roc_auc=float(roc_auc_score(yt, scores)),
        threshold=threshold,
        target_recall=target_recall,
        test_pass_rates=pass_rates,
        test_precision=float(yt[flagged].mean()) if flagged.any() else float("nan"),
        catalogue_precision=catalogue_precision(pass_rates, catalogue_counts),
        mes_pass_rates=mes_rates,
        mes_threshold=mes_threshold,
        curves=curves,
        test_scores=scores.tolist(),
        test_labels=yt.tolist(),
    )


def format_report(result: DR25Result) -> str:
    def pct(x: float) -> str:
        return f"{x:.1%}" if np.isfinite(x) else "n/a"

    s, c = result.class_counts_sample, result.class_counts_catalogue
    t = result.test_class_counts
    lines = [
        "Kepler DR25 training set: first model on the folded views",
        "=" * 58,
        "",
        f"Catalogue: {sum(c.values()):,} TCEs ({c['PC']:,} PC, {c['AFP']:,} AFP, {c['NTP']:,} NTP).",
        f"Sample:    {result.n_tces:,} TCEs on {result.n_stars:,} stars "
        f"({s['PC']:,} PC, {s['AFP']:,} AFP, {s['NTP']:,} NTP); "
        f"{result.n_failed} more could not be built.",
        f"Split by star: {result.n_train:,} train, {result.n_test:,} test "
        f"({t['PC']} PC, {t['AFP']} AFP, {t['NTP']} NTP).",
        "",
        "Ranking (test split)",
        f"  average precision, views model   {result.average_precision:.3f} "
        f"(95% bootstrap {result.average_precision_ci[0]:.3f} to {result.average_precision_ci[1]:.3f})",
        f"  average precision, MES alone     {result.mes_average_precision:.3f}",
        f"  chance level (PC fraction)       {result.chance_average_precision:.3f}",
        f"  ROC AUC, views model             {result.roc_auc:.3f}",
        "",
        f"At a threshold frozen on training folds for {result.target_recall:.0%} PC recall",
        "                      views model   MES alone",
        f"  PC kept             {pct(result.test_pass_rates['PC']):>10}  {pct(result.mes_pass_rates['PC']):>10}",
        f"  AFP rejected        {pct(1 - result.test_pass_rates['AFP']):>10}  {pct(1 - result.mes_pass_rates['AFP']):>10}",
        f"  NTP rejected        {pct(1 - result.test_pass_rates['NTP']):>10}  {pct(1 - result.mes_pass_rates['NTP']):>10}",
        f"  precision, sample   {pct(result.test_precision):>10}",
        f"  precision at the catalogue's class mix: {pct(result.catalogue_precision)}",
        "",
        "Caveats",
        "  The labels are the DR25 Robovetter's calls, so this measures agreement",
        "  with the Robovetter, not with the truth.  The sample is balanced by class;",
        "  the catalogue-mix precision re-weights the measured per-class pass rates",
        "  to 4,034 : 3,025 : 26,973.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_examples(ts: TrainingSet, path: Path, seed: int = 0) -> Path:
    """One high-MES example of each class, all five views side by side."""
    from .plots import INK_SOFT, NEUTRAL, SERIES, _save, _style, plt

    _style()
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.4), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    names = {"PC": "planet candidate", "AFP": "astrophysical false positive", "NTP": "not transit-like"}
    rng = np.random.default_rng(seed)
    mes = np.nan_to_num(ts.scalars["mes"])
    for row, cls in enumerate(CLASSES):
        members = np.flatnonzero((ts.classes == cls) & (mes >= np.nanpercentile(mes, 50)))
        if members.size == 0:
            members = np.flatnonzero(ts.classes == cls)
        if members.size == 0:
            for ax in axes[row]:
                ax.set_visible(False)
            continue
        i = int(rng.choice(members))
        g = ts.views["global"][i]
        axes[row, 0].plot(np.linspace(-0.5, 0.5, g.size), g, lw=0.9, color=SERIES[0])
        axes[row, 0].set_title(f"{names[cls]}: KIC {ts.kepids[i]}, TCE {ts.tce_ids[i][-2:]}", loc="left")
        x = np.linspace(-4, 4, ts.views["local"].shape[1])
        axes[row, 1].plot(x, ts.views["odd"][i], lw=1.2, color=SERIES[0], label="odd")
        axes[row, 1].plot(x, ts.views["even"][i], lw=1.2, color=SERIES[1], label="even")
        axes[row, 2].plot(x, ts.views["local"][i], lw=1.2, color=SERIES[0], label="transit")
        axes[row, 2].plot(x, ts.views["secondary"][i], lw=1.2, color=SERIES[1], label="half an orbit later")
        for ax in axes[row]:
            ax.axhline(0, lw=0.8, color=NEUTRAL)
        if row == 0:
            axes[row, 1].legend(loc="lower left")
            axes[row, 2].legend(loc="lower left")
    for ax, label in zip(axes[-1], ("orbital phase", "transit durations from mid-transit", "transit durations from mid-transit")):
        ax.set_xlabel(label)
    for ax in axes[:, 0]:
        ax.set_ylabel("flux / local depth", color=INK_SOFT)
    fig.suptitle("DR25 views: global, odd vs even, transit vs secondary", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_pr(result: DR25Result, path: Path) -> Path:
    from .plots import NEUTRAL, SERIES, _save, _style, plt

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for key, colour, label, ap in (
        ("model", SERIES[0], "gradient boosting on views", result.average_precision),
        ("mes", SERIES[1], "baseline: Kepler MES", result.mes_average_precision),
    ):
        curve = result.curves[key]
        ax.plot(curve["recall"], curve["precision"], lw=2.0, color=colour, label=f"{label}  (AP = {ap:.3f})")
    ax.axhline(result.chance_average_precision, lw=1.4, ls="--", color=NEUTRAL)
    ax.text(0.015, result.chance_average_precision + 0.02, f"random ranking (AP = {result.chance_average_precision:.3f})", fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall on planet candidates")
    ax.set_ylabel("precision")
    ax.set_title(f"Held-out DR25 stars: {result.n_test} TCEs, class-balanced sample", loc="left")
    ax.legend(loc="center left")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.kepler_dr25",
        description="Build the Kepler DR25 TCE training set and score a first model on it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE, help="labelled TCE list; downloaded if absent")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--per-class", type=int, default=1000, help="TCEs drawn per class (PC, AFP, NTP)")
    size.add_argument("--all", action="store_true", help="every TCE in the catalogue (about 17,000 stars to download)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=8, help="stars processed at once")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--cache-dir", type=Path, default=None, help="per-star light curves (default: RESULTS_DIR/curves)")
    parser.add_argument("--no-train", action="store_true", help="build the training set only")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir or args.results_dir / "curves"
    catalogue = load_or_fetch_catalogue(args.catalogue)
    counts = class_counts(catalogue)
    print(f"catalogue: {len(catalogue):,} TCEs {counts}")
    chosen = list(catalogue) if args.all else stratified_sample(catalogue, args.per_class, args.seed)
    n_stars = len({t.kepid for t in chosen})
    print(f"building views for {len(chosen):,} TCEs on {n_stars:,} stars")

    ts = build_training_set(chosen, cache_dir, catalogue=catalogue, n_jobs=args.n_jobs, verbose=5)
    path = ts.save(args.results_dir / "training_set.npz")
    print(f"wrote {path}: {len(ts):,} TCEs, {len(ts.failures)} failed")
    if args.no_train:
        return 0

    result = evaluate_training_set(ts, counts, seed=args.seed)
    report = format_report(result)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "report.txt").write_text(report)
    (args.results_dir / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(report)
    if not args.no_figures:
        plot_examples(ts, args.figures_dir / "01_views.png", seed=args.seed)
        plot_pr(result, args.figures_dir / "02_precision_recall.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
