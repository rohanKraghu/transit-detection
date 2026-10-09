"""Train on real labels: the TOI hosts of one set of sectors.

Every other model in this project learns from labels known by construction,
synthetic or injected, and the TOI benchmark (:mod:`transitml.benchmark`)
only scores it.  This module trains the same classifier on TOI follow-up
dispositions instead: the labelled hosts of one set of sectors, searched and
featurised exactly as the benchmark's are.  The benchmark then scores the
model, unchanged, on the hosts of other sectors, after removing every star it
was trained on.

Two things set this model apart from the others, and the report says both:

* **The labels carry the follow-up programme's selection.**  A planet is
  easier to confirm when it is deep and its star is bright, and a false
  positive is easier to catch when a neighbour can be resolved.  A model
  trained on resolved TOIs learns that selection along with the
  astrophysics, so it ranks TOIs that resemble the resolved ones.  On the
  candidates still open it may do worse, and nothing here measures that.
* **About half the stars are planets**, which is the catalogue's mix, not the
  sky's.  At that rate the survey rule for the threshold, a floor on
  precision (:func:`~transitml.model.select_threshold`), is met by keeping
  every star or nearly, so the threshold is instead where the calibrated
  probability of a planet reaches one half
  (:func:`~transitml.model.probability_threshold`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logit
from sklearn.metrics import average_precision_score

from .calibration import PlattScaling, brier_score, reliability_table
from .centroid import combined_pixel_files, sector_combination
from .data.loader import Dataset
from .data.toi import POSITIVE_DISPOSITIONS
from .evaluate import N_BOOTSTRAP, CurveScores, bootstrap_indices, score_curve
from .features import FEATURE_NAMES
from .model import BASELINES, Split, TrainedModel, build_model, select_threshold, train

#: Where the threshold of a model trained on TOI hosts goes: keep a star when it
#: is at least as likely a planet as not, at the training set's own mix.
TOI_OPERATING_PROBABILITY: float = 0.5

#: Probability bins for the calibration check on the scored stars.  At about
#: half planets equal widths suit, unlike the survey's log-spaced bins.
TOI_RELIABILITY_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def train_on_hosts(
    dataset: Dataset,
    *,
    feature_names: Sequence[str],
    n_folds: int,
    seed: int,
    target_precision: float,
    precision_lcb_z: float,
    operating_probability: float = TOI_OPERATING_PROBABILITY,
) -> tuple[TrainedModel, Split]:
    """Train on every star in ``dataset``; threshold and calibration come from cross-validation.

    Nothing is held out: the test set is the benchmark's, on other sectors.
    The returned :class:`~transitml.model.Split` has an empty test half.
    """
    X = dataset.inputs(feature_names)
    y = dataset.y
    index = np.arange(len(y))
    split = Split(
        X_train=X,
        y_train=y,
        X_test=X[:0],
        y_test=y[:0],
        train_index=index,
        test_index=index[:0],
    )
    trained = train(
        split,
        n_folds=n_folds,
        seed=seed,
        target_precision=target_precision,
        precision_lcb_z=precision_lcb_z,
        operating_probability=operating_probability,
        feature_names=feature_names,
    )
    return trained, split


def learning_curve(
    train_set: Dataset,
    test_set: Dataset,
    *,
    feature_names: Sequence[str],
    seed: int,
    repeats: int = 10,
    start: int = 100,
) -> list[dict[str, float]]:
    """Average precision on ``test_set`` against the number of training hosts.

    The classifier is trained on random subsets of ``train_set``, each with its
    planet rate, at sizes doubling from ``start``, ``repeats`` subsets to a
    size, and scored on ``test_set``.  The last row is all of ``train_set``,
    the model itself.  A curve still rising at the end says more labels would
    help; one gone flat says they would not.
    """
    X, y = train_set.inputs(feature_names), train_set.y
    X_test, y_test = test_set.inputs(feature_names), test_set.y
    sizes, size = [], start
    while size < len(y):
        sizes.append(size)
        size *= 2
    sizes.append(len(y))
    rng = np.random.default_rng(seed)
    planets, others = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows: list[dict[str, float]] = []
    for size in sizes:
        scores = []
        for draw in range(1 if size == len(y) else repeats):
            if size == len(y):
                rows_used = np.arange(len(y))
            else:
                k = round(size * len(planets) / len(y))
                rows_used = np.concatenate(
                    [
                        rng.choice(planets, k, replace=False),
                        rng.choice(others, size - k, replace=False),
                    ]
                )
            model = build_model(seed + draw).fit(X[rows_used], y[rows_used])
            scores.append(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))
        rows.append(
            {
                "n_training": int(size),
                "average_precision": float(np.mean(scores)),
                "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "n_draws": len(scores),
            }
        )
    return rows


@dataclass
class TrainingSummary:
    """What cross-validation on the training hosts says, before any benchmark star is read."""

    sectors: list[int]
    selection: dict[str, int]
    n_without_curve: int
    n_stars: int
    n_planets: int
    feature_names: list[str]
    n_folds: int
    #: Out-of-fold scores of the model, and the baselines on the same stars.
    model: CurveScores
    baselines: list[CurveScores]
    threshold: float
    threshold_rule: str
    threshold_probability: float
    #: Out of fold, at the threshold.
    planet_recall: float
    false_positive_rejection: float
    precision: float
    #: What the survey rule would have done with the same scores.
    precision_rule: dict[str, Any]
    calibration: dict[str, float]
    #: Stars with a pixel file, when the model reads the centroid test.
    n_with_pixels: int | None = None
    #: Pixel files tested, when each star's sectors were tested and combined.
    n_pixel_files: int | None = None
    #: How they were combined (``"stouffer"`` or ``"sky"``), when they were.
    combination: str | None = None
    #: Each star searched on every one of ``sectors`` it was observed in, joined.
    stitched: bool = False
    #: Further sectors added to each star's curve (``--join-sectors``), if any.
    joined: str | None = None

    @property
    def positive_rate(self) -> float:
        return self.n_planets / self.n_stars if self.n_stars else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sectors": self.sectors,
            "stitched": self.stitched,
            **({"joined": self.joined} if self.joined else {}),
            "selection": self.selection,
            "n_without_curve": self.n_without_curve,
            "n_stars": self.n_stars,
            "n_planets": self.n_planets,
            "n_false_positives": self.n_stars - self.n_planets,
            "positive_rate": self.positive_rate,
            "feature_names": self.feature_names,
            **({"n_with_pixels": self.n_with_pixels} if self.n_with_pixels is not None else {}),
            **({"n_pixel_files": self.n_pixel_files} if self.n_pixel_files is not None else {}),
            **({"combination": self.combination} if self.combination is not None else {}),
            "cross_validation": {
                "n_folds": self.n_folds,
                "chance_average_precision": self.positive_rate,
                "model": self.model.to_dict(),
                "baselines": [b.to_dict() for b in self.baselines],
            },
            "operating_point": {
                "threshold": self.threshold,
                "rule": self.threshold_rule,
                "threshold_as_probability": self.threshold_probability,
                "cv_planet_recall": self.planet_recall,
                "cv_false_positive_rejection": self.false_positive_rejection,
                "cv_precision": self.precision,
                "precision_rule": self.precision_rule,
            },
            "calibration": self.calibration,
        }


def _rate(mask: np.ndarray, within: np.ndarray) -> float:
    n = int(within.sum())
    return float((mask & within).sum() / n) if n else float("nan")


def summarise_training(
    dataset: Dataset,
    split: Split,
    trained: TrainedModel,
    *,
    sectors: Sequence[int],
    selection: dict[str, int],
    n_without_curve: int,
    n_folds: int,
    target_precision: float,
    precision_lcb_z: float,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
    centroids: Sequence[dict[str, Any] | None] | None = None,
    stitched: bool = False,
    joined: str | None = None,
) -> TrainingSummary:
    """Out-of-fold ranking and operating point of ``trained`` on its own training hosts."""
    y = split.y_train
    oof = trained.oof_scores
    resamples = bootstrap_indices(len(y), n_bootstrap, seed) if n_bootstrap > 0 else None
    model = score_curve(
        "gradient_boosting", "out-of-fold scores on the training hosts", y, oof, resamples
    )
    baselines = [
        score_curve(b.name, b.description, y, b.score(dataset.X), resamples) for b in BASELINES
    ]
    planets, negatives = y == 1, y == 0
    kept = oof >= trained.threshold
    survey_threshold, survey_rule, survey_precision, _ = select_threshold(
        y, oof, target_precision, precision_lcb_z
    )
    survey_kept = oof >= survey_threshold
    return TrainingSummary(
        stitched=stitched,
        joined=joined,
        sectors=list(sectors),
        selection=dict(selection),
        n_without_curve=int(n_without_curve),
        n_stars=len(y),
        n_planets=int(planets.sum()),
        feature_names=list(trained.feature_names),
        n_folds=n_folds,
        model=model,
        baselines=baselines,
        threshold=trained.threshold,
        threshold_rule=trained.threshold_rule,
        threshold_probability=trained.threshold_probability,
        planet_recall=_rate(kept, planets),
        false_positive_rejection=_rate(~kept, negatives),
        precision=trained.achieved_cv_precision,
        precision_rule={
            "rule": survey_rule,
            "threshold": survey_threshold,
            "cv_planet_recall": _rate(survey_kept, planets),
            "cv_false_positive_rejection": _rate(~survey_kept, negatives),
            "cv_precision": survey_precision,
        },
        calibration=trained.calibration.to_dict(),
        n_with_pixels=(
            sum(t is not None for t in centroids) if centroids is not None else None
        ),
        n_pixel_files=combined_pixel_files(centroids) if centroids is not None else None,
        combination=sector_combination(centroids) if centroids is not None else None,
    )


def scored_calibration(
    stars: Sequence[dict[str, Any]],
    calibration: PlattScaling,
    edges: tuple[float, ...] = TOI_RELIABILITY_EDGES,
) -> dict[str, Any]:
    """How well the calibrated ``P(planet)`` holds on the stars a benchmark scored.

    ``stars`` are the benchmark's rows
    (:attr:`~transitml.benchmark.BenchmarkResult.stars`): each carries the
    classifier's score, whose log-odds the calibration maps, and its reference
    TOI's disposition, which agrees with the star's label.  The probability is
    at the training planet rate, like the threshold; the benchmark's own rate
    is not used.
    """
    score = np.clip(np.array([s["model_score"] for s in stars], dtype=float), 1e-12, 1 - 1e-12)
    y = np.array([s["disposition"] in POSITIVE_DISPOSITIONS for s in stars], dtype=int)
    p = calibration.probability(logit(score))
    return {
        "brier": brier_score(y, p),
        "brier_training_rate": brier_score(y, np.full(y.size, calibration.train_positive_rate)),
        "mean_probability": float(p.mean()),
        "planet_rate": float(y.mean()),
        "reliability": reliability_table(y, p, edges),
    }


def _sector_text(sectors: Sequence[int]) -> str:
    """``[1, ..., 13]`` -> ``"1 to 13"``, ``[1, ..., 13, 27, ..., 96]`` -> ``"1 to 13 and 27 to 96"``.

    Runs of three or more are written as ranges, anything else is listed.
    """
    runs: list[list[int]] = []
    for sector in sectors:
        if runs and sector == runs[-1][-1] + 1:
            runs[-1].append(sector)
        else:
            runs.append([sector])
    parts: list[str] = []
    for run in runs:
        if len(run) > 2:
            parts.append(f"{run[0]} to {run[-1]}")
        else:
            parts.extend(str(s) for s in run)
    if len(parts) > 1 and any(len(run) > 2 for run in runs):
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    return ", ".join(parts)


def _fmt(value: float, spec: str = ".3f") -> str:
    return format(value, spec) if np.isfinite(value) else "n/a"


def _ap_line(label: str, curve: CurveScores) -> str:
    return (
        f"  {label:<34s} AP = {curve.average_precision:.3f}"
        f"  [{_fmt(curve.ap_low)}, {_fmt(curve.ap_high)}]   (ROC-AUC {curve.roc_auc:.3f})"
    )


def format_training_report(
    summary: TrainingSummary,
    benchmark: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    learning: Sequence[dict[str, float]] | None = None,
) -> str:
    """Human-readable summary of the training run; with ``benchmark``, its headline too.

    ``benchmark`` is :meth:`~transitml.benchmark.BenchmarkResult.to_dict` of the
    model scored on other sectors, ``calibration`` its
    :func:`scored_calibration` and ``learning`` its :func:`learning_curve`.
    """
    lines: list[str] = []
    add = lines.append
    sel = summary.selection
    add("=" * 72)
    add(f"TRAINED ON REAL LABELS: TOI HOSTS OF SECTORS {_sector_text(summary.sectors).upper()}")
    add("=" * 72)
    add(
        f"stars in TOI table: {sel.get('stars_in_table', 0)}   unlabelled (PC/APC): "
        f"{sel.get('unlabelled', 0)}   not observed in these sectors: "
        f"{sel.get('not_in_sectors', 0)}"
    )
    if sel.get("in_benchmark_sectors"):
        add(f"observed in the benchmark's sectors, so left out: {sel['in_benchmark_sectors']}")
    add(
        f"selected: {sel.get('selected', 0)}   no light curve at MAST: "
        f"{summary.n_without_curve}   trained on: {summary.n_stars}"
    )
    if summary.stitched:
        add("  each searched on every one of these sectors it was observed in, joined")
    if summary.joined:
        add(f"  with every sector of {summary.joined} it was observed in added to its curve")
    if sel.get("from_a_later_sector"):
        add(
            f"  of which {sel['from_a_later_sector']} from a later sector than their first, "
            "which MAST had no light curve for"
        )
    add(
        f"training stars: {summary.n_planets} planets (CP/KP), "
        f"{summary.n_stars - summary.n_planets} false positives (FP/FA), "
        f"planet rate {summary.positive_rate:.1%}"
    )
    extra = [name for name in summary.feature_names if name not in FEATURE_NAMES]
    add(
        f"model inputs: the {len(FEATURE_NAMES)} light-curve features"
        + (f" and {', '.join(extra)}" if extra else "")
    )
    if summary.n_with_pixels is not None:
        add(f"stars with a target pixel file: {summary.n_with_pixels} of {summary.n_stars}")
    if summary.n_pixel_files is not None:
        add(
            f"  each tested in every sector it was joined from ({summary.n_pixel_files} "
            "pixel files), the tests combined"
            + (" as offsets on the sky" if summary.combination == "sky" else "")
        )
    add("")
    add(f"{summary.n_folds}-fold cross-validation on the training stars (out-of-fold scores)")
    add("-" * 72)
    add(f"  {'random ranking (chance)':<34s} AP = {summary.positive_rate:.3f}")
    for b in summary.baselines:
        add(_ap_line("baseline: " + b.name, b))
    add(_ap_line("model: gradient boosting", summary.model))
    add("  (brackets are 68% bootstrap intervals)")
    add("")
    add("Operating threshold")
    add("-" * 72)
    add(f"  score >= {summary.threshold:.4f}: {summary.threshold_rule}")
    add(
        f"  out of fold: planets kept {_fmt(summary.planet_recall, '.2f')}, false positives "
        f"rejected {_fmt(summary.false_positive_rejection, '.2f')}, precision "
        f"{_fmt(summary.precision, '.2f')}"
    )
    rule = summary.precision_rule
    add(
        f"  the survey rule would keep {_fmt(rule['cv_planet_recall'], '.2f')} of the planets "
        f"and reject {_fmt(rule['cv_false_positive_rejection'], '.2f')} of the false positives:"
    )
    add(f"    {rule['rule']}")
    cal = summary.calibration
    add(
        f"  calibration: P(planet) = expit({cal['slope']:.3f} * log-odds "
        f"{'+' if cal['intercept'] >= 0 else '-'} {abs(cal['intercept']):.3f})"
    )
    if benchmark is not None:
        add("")
        add(
            "Scored, unchanged, on the TOI hosts of sectors "
            f"{_sector_text(benchmark['sectors'])} (toi_benchmark.txt)"
        )
        add("-" * 72)
        n_fp = benchmark["n_stars"] - benchmark["n_planets"]
        chance = benchmark["chance_average_precision"]
        add(
            f"  {benchmark['n_stars']} stars, {benchmark['n_planets']} planets and {n_fp} false "
            f"positives, none of them trained on; chance AP {chance:.3f}"
        )
        model = benchmark["model"]
        low, high = model["average_precision_ci68"]
        add(
            f"  {'model':<34s} AP = {model['average_precision']:.3f}  [{_fmt(low)}, {_fmt(high)}]"
            f"   (ROC-AUC {model['roc_auc']:.3f})"
        )
        veto = benchmark.get("centroid_veto")
        if veto is not None:
            low, high = veto["model"]["average_precision_ci68"]
            add(
                f"  {'model with the centroid veto':<34s} AP = "
                f"{veto['model']['average_precision']:.3f}  [{_fmt(low)}, {_fmt(high)}]"
            )
        point = benchmark["operating_point"]
        add(
            f"  at the threshold: planets kept {_fmt(point['planet_recall'], '.2f')}, "
            f"false positives rejected {_fmt(point['false_positive_rejection'], '.2f')}"
        )
    if calibration is not None:
        add(
            f"  calibrated P(planet): Brier score {calibration['brier']:.3f}, against "
            f"{calibration['brier_training_rate']:.3f} for the training planet rate alone"
        )
        add(f"    {'P(planet)':<14s}{'stars':>6s}{'mean P':>9s}{'planets':>9s}")
        for row in calibration["reliability"]:
            label = f"{row['p_low']:.1f} - {row['p_high']:.1f}"
            add(
                f"    {label:<14s}{row['n']:>6d}{_fmt(row['mean_predicted'], '9.3f')}"
                f"{_fmt(row['observed_rate'], '9.3f')}"
            )
    if learning:
        add("")
        add("Learning curve: trained on random subsets of the training stars, each with")
        add("their planet rate, and scored on the stars above (the last row is the model)")
        add("-" * 72)
        add(f"  {'training stars':>14s}{'AP':>8s}{'sd':>8s}{'draws':>7s}")
        for row in learning:
            sd = f"{row['sd']:8.3f}" if row["n_draws"] > 1 else f"{'-':>8s}"
            add(
                f"  {row['n_training']:>14d}{row['average_precision']:8.3f}{sd}"
                f"{row['n_draws']:>7d}"
            )
    add("=" * 72)
    return "\n".join(lines)
