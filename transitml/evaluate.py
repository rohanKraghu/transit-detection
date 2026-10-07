"""Evaluation for a severely imbalanced detection problem.

**Accuracy is not reported anywhere in this project, on purpose.**  At a 4%
positive rate, the classifier that returns "no planet" for every star scores 96%
accuracy while finding nothing.  Any metric a constant function can win is not
measuring the thing we care about.

**Plain ROC-AUC is reported but not used as the headline.**  The false-positive
rate on the ROC axis has ~2300 negatives in its denominator, so moving from 5
false positives to 50 -- the difference between a usable and an unusable
candidate list -- changes FPR by 2 percentage points and barely moves the curve.
Precision has the number of *flagged* objects in its denominator, which is the
quantity that maps onto telescope time, so precision-recall is the right plane
and **average precision** is the headline number.

The reference point for average precision is the positive rate itself: a random
ranking achieves AP = P(positive).  An AP of 0.6 at a 4% positive rate is a 15x
lift; an AP of 0.6 at a 50% positive rate would be worse than a coin flip.  Both
numbers are always reported together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from .calibration import calibration_summary
from .data.loader import Dataset
from .features import FEATURE_NAMES
from .model import BASELINES, Split, TrainedModel
from .treeshap import top_reasons

#: SHAP reasons listed per star in the report's false-positive and missed-planet rows.
N_SHAP_REASONS: int = 3


@dataclass
class CurveScores:
    """Precision-recall curve plus its summary statistic, for one scorer."""

    name: str
    description: str
    average_precision: float
    roc_auc: float
    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    ap_low: float = float("nan")
    ap_high: float = float("nan")
    scores: NDArray[np.float64] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "average_precision": self.average_precision,
            "average_precision_ci68": [self.ap_low, self.ap_high],
            "roc_auc": self.roc_auc,
        }


@dataclass
class EvaluationResult:
    """Everything the pipeline reports, computed on held-out data only."""

    n_test: int
    n_test_positive: int
    positive_rate: float
    chance_average_precision: float
    model: CurveScores
    baselines: list[CurveScores]
    threshold: float
    threshold_rule: str
    cv_precision: float
    cv_recall: float
    confusion: dict[str, int]
    test_precision: float
    test_recall: float
    test_f1: float
    precision_at_k: dict[str, float]
    false_positive_breakdown: dict[str, int]
    missed_breakdown: list[dict[str, Any]]
    recall_by_snr: list[dict[str, Any]]
    search_recovery: float = float("nan")
    bootstrap_win_rate_vs_best_baseline: float = float("nan")
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    shap_base_log_odds: float = float("nan")
    shap_importance: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_test": self.n_test,
            "n_test_positive": self.n_test_positive,
            "test_positive_rate": self.positive_rate,
            "chance_average_precision": self.chance_average_precision,
            "model": self.model.to_dict(),
            "baselines": [b.to_dict() for b in self.baselines],
            "lift_over_chance": self.model.average_precision / self.chance_average_precision
            if self.chance_average_precision > 0
            else float("nan"),
            "operating_point": {
                "threshold": self.threshold,
                "rule": self.threshold_rule,
                "cv_precision_on_train": self.cv_precision,
                "cv_recall_on_train": self.cv_recall,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
                "test_f1": self.test_f1,
                "confusion_matrix": self.confusion,
            },
            "precision_at_k": self.precision_at_k,
            "false_positive_breakdown": self.false_positive_breakdown,
            "missed_planets": self.missed_breakdown,
            "recall_by_true_snr": self.recall_by_snr,
            "search_recovery_rate": self.search_recovery,
            "bootstrap_win_rate_vs_best_baseline": self.bootstrap_win_rate_vs_best_baseline,
            "feature_importance": self.feature_importance,
            "calibration": self.calibration,
            "shap": {
                "units": "calibrated log-odds; base + sum over features = the star's log-odds",
                "base_log_odds": self.shap_base_log_odds,
                "mean_abs_by_feature": self.shap_importance,
            },
            "false_positives": self.false_positives,
        }


#: Bootstrap resamples used for the interval on average precision.
N_BOOTSTRAP: int = 2000


def fast_average_precision(
    y_true: NDArray[np.int_], scores: NDArray[np.float64]
) -> float:
    """Average precision, identical to sklearn's but much cheaper in a loop.

    ``sum over thresholds of (R_n - R_{n-1}) * P_n``, with tied scores collapsed
    into one threshold -- which matters here, because bootstrap resampling
    duplicates rows and therefore creates ties by construction.  Used only
    inside the bootstrap, where it is called tens of thousands of times;
    ``test_fast_average_precision_matches_sklearn`` pins it to the reference.
    """
    order = np.argsort(-scores, kind="mergesort")
    y = np.asarray(y_true)[order]
    s = np.asarray(scores)[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), s.size - 1]
    tp, fp = tp[last], fp[last]
    n_pos = float(tp[-1])
    if n_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def bootstrap_indices(
    n: int, n_resamples: int, seed: int
) -> NDArray[np.int_]:
    """Resample indices once, shared by every scorer.

    Sharing the resamples is what makes the *difference* between two scorers
    comparable: the model and the baseline are then scored on the same
    bootstrap worlds, and their correlation is preserved.
    """
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def score_curve(
    name: str,
    description: str,
    y_true: NDArray[np.int_],
    scores: NDArray[np.float64],
    resamples: NDArray[np.int_] | None = None,
) -> CurveScores:
    """Average precision, ROC-AUC, the PR curve, and a bootstrap interval.

    With ~34 positives in the test set, average precision has a genuine
    sampling uncertainty of several hundredths.  Quoting it to three decimals
    without an interval invites over-reading the third one, so the 68% interval
    is computed and reported alongside.
    """
    finite = np.where(np.isfinite(scores), scores, np.nanmin(scores[np.isfinite(scores)]) - 1.0)
    precision, recall, _ = precision_recall_curve(y_true, finite)
    low = high = float("nan")
    if resamples is not None:
        draws = [
            fast_average_precision(y_true[idx], finite[idx])
            for idx in resamples
            if y_true[idx].sum() >= 2
        ]
        if draws:
            low, high = (float(v) for v in np.percentile(draws, [16, 84]))
    return CurveScores(
        name=name,
        description=description,
        average_precision=float(average_precision_score(y_true, finite)),
        roc_auc=float(roc_auc_score(y_true, finite)),
        precision=precision,
        recall=recall,
        ap_low=low,
        ap_high=high,
        scores=finite,
    )


def bootstrap_win_rate(
    y_true: NDArray[np.int_],
    model_scores: NDArray[np.float64],
    baseline_scores: NDArray[np.float64],
    resamples: NDArray[np.int_],
) -> float:
    """Fraction of bootstrap resamples in which the model out-scores the baseline.

    The paired form of the question that matters: not "are the intervals
    disjoint" but "how often does the model win on the same data".
    """
    wins = 0
    total = 0
    for idx in resamples:
        if y_true[idx].sum() < 2:
            continue
        total += 1
        wins += fast_average_precision(y_true[idx], model_scores[idx]) > (
            fast_average_precision(y_true[idx], baseline_scores[idx])
        )
    return float(wins / total) if total else float("nan")


def precision_at_k(
    y_true: NDArray[np.int_], scores: NDArray[np.float64], k: int
) -> float:
    """Fraction of the top-k ranked candidates that are real planets.

    This is the number an observing proposal actually lives on: "we have k
    nights, here are our k best targets, how many are real?"
    """
    k = min(k, len(y_true))
    if k == 0:
        return float("nan")
    top = np.argsort(-scores)[:k]
    return float(y_true[top].mean())


#: Low-order aliases a periodic search is allowed to land on and still count as
#: a recovery: half, double and triple the true period all phase-fold the
#: transits on top of each other (or on top of every other one).
_PERIOD_ALIASES: tuple[float, ...] = (1.0, 0.5, 2.0, 3.0, 1.0 / 3.0)


def period_recovered(
    found: NDArray[np.float64], injected: NDArray[np.float64], tolerance: float = 0.03
) -> NDArray[np.bool_]:
    """Whether the search landed on the injected period or a low-order alias."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.asarray(found, dtype=float) / np.asarray(injected, dtype=float)
    close = np.zeros(ratio.shape, dtype=bool)
    for alias in _PERIOD_ALIASES:
        close |= np.abs(ratio - alias) < tolerance * alias
    return close & np.isfinite(ratio)


def recall_by_snr(
    y_true: NDArray[np.int_],
    predicted: NDArray[np.bool_],
    true_snr: NDArray[np.float64],
    edges: NDArray[np.float64],
    recovered: NDArray[np.bool_] | None = None,
) -> list[dict[str, Any]]:
    """Completeness as a function of injected transit SNR.

    This is the pipeline-characterisation table every transit survey publishes:
    recall is meaningless as a single number because it is set almost entirely
    by where the injected signals sit relative to the noise.  The interesting
    quantity is the SNR at which recall crosses 50%.

    Recall is also decomposed.  A planet is missed either because the **search**
    never found its period -- no classifier can rescue that -- or because the
    search found it and the **classifier** rejected it.  Those are different
    problems with different fixes, and reporting only their product hides which
    one is binding.
    """
    rows: list[dict[str, Any]] = []
    positives = y_true == 1
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = positives & (true_snr >= lo) & (true_snr < hi)
        n = int(sel.sum())
        row: dict[str, Any] = {
            "snr_low": float(lo),
            "snr_high": float(hi),
            "n_planets": n,
            "recall": float(predicted[sel].mean()) if n else float("nan"),
            "search_recovery": float("nan"),
            "classifier_recall_given_search": float("nan"),
        }
        if n and recovered is not None:
            row["search_recovery"] = float(recovered[sel].mean())
            found = sel & recovered
            if found.any():
                row["classifier_recall_given_search"] = float(predicted[found].mean())
        rows.append(row)
    return rows


def evaluate(
    dataset: Dataset,
    split: Split,
    trained: TrainedModel,
    *,
    top_k: int,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
    snr_edges: tuple[float, ...] = (0.0, 7.0, 12.0, 20.0, 40.0, 1e9),
) -> EvaluationResult:
    """Score the model and the baselines on the held-out test set.

    Everything here reads ``split.X_test`` / ``split.y_test`` and the test rows
    of ``dataset.meta``.  The threshold comes in pre-frozen from ``trained``;
    nothing in this function is allowed to choose it.
    """
    y_test = split.y_test
    X_test = split.X_test
    scores = trained.score(X_test)

    resamples = (
        bootstrap_indices(len(y_test), n_bootstrap, seed) if n_bootstrap > 0 else None
    )
    model_curve = score_curve(
        "gradient_boosting",
        "HistGradientBoostingClassifier on BLS + vetting features",
        y_test,
        scores,
        resamples,
    )
    baseline_curves = [
        score_curve(b.name, b.description, y_test, b.score(X_test), resamples)
        for b in BASELINES
    ]
    strongest = max(baseline_curves, key=lambda c: c.average_precision)
    win_rate = (
        bootstrap_win_rate(y_test, model_curve.scores, strongest.scores, resamples)
        if resamples is not None
        else float("nan")
    )

    predicted = scores >= trained.threshold
    probability = trained.probability(X_test)
    calibration = calibration_summary(
        y_test,
        scores,
        probability,
        trained.log_odds(X_test),
        predicted,
        trained.calibration,
        trained.threshold_probability,
    )
    shap_values = trained.explain(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )

    meta_test = dataset.meta.iloc[split.test_index].reset_index(drop=True)

    # What kind of object is producing each false positive?  "How many false
    # positives" is much less useful than "are they eclipsing binaries the
    # vetting features should have caught, or noise peaks?"
    fp_mask = predicted & (y_test == 0)
    kinds = meta_test["kind"].astype(str).to_numpy()
    fp_breakdown = {
        str(kind): int(np.sum(fp_mask & (kinds == kind))) for kind in np.unique(kinds)
    }
    fp_breakdown = {k: v for k, v in fp_breakdown.items() if k != "planet"}
    target_ids = meta_test["target_id"].astype(str).to_numpy()
    false_positives = [
        {
            "target_id": str(target_ids[i]),
            "kind": str(kinds[i]),
            "model_score": float(scores[i]),
            "probability": float(probability[i]),
            "pushed_up_by": top_reasons(
                shap_values[i], X_test[i], FEATURE_NAMES, N_SHAP_REASONS, "up"
            ),
        }
        for i in np.flatnonzero(fp_mask)
    ]
    false_positives.sort(key=lambda r: -r["model_score"])

    # And which planets were missed, with the parameters that explain why.
    missed = predicted == False  # noqa: E712 - explicit for the mask below
    missed &= y_test == 1
    features_test = dataset.features.iloc[split.test_index].reset_index(drop=True)
    bls_period = 10.0 ** features_test["log_period"].to_numpy(dtype=float)
    injected_period = pd.to_numeric(meta_test["period"], errors="coerce").to_numpy(float)
    recovered = period_recovered(bls_period, injected_period)

    missed_rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(missed):
        row = meta_test.iloc[i]
        feature_row = features_test.iloc[i]
        missed_rows.append(
            {
                "target_id": str(row["target_id"]),
                "true_snr": _as_float(row["true_snr"]),
                "depth_ppm": _as_float(row["depth"]) * 1e6
                if np.isfinite(_as_float(row["depth"]))
                else float("nan"),
                "period_days": _as_float(row["period"]),
                "duration_hours": _as_float(row["duration_t14"]) * 24.0,
                "impact_parameter": _as_float(row["impact_parameter"]),
                "n_transits_in_window": _as_float(row["n_transits_in_window"]),
                "model_score": float(scores[i]),
                "probability": float(probability[i]),
                "pushed_down_by": top_reasons(
                    shap_values[i], X_test[i], FEATURE_NAMES, N_SHAP_REASONS, "down"
                ),
                "period_recovered": bool(recovered[i]),
                "odd_even_sigma": _as_float(feature_row["odd_even_sigma"]),
                "secondary_sigma": _as_float(feature_row["secondary_sigma"]),
                "red_noise_beta": _as_float(feature_row["red_noise_beta"]),
                "reason": (
                    "search did not recover the period"
                    if not recovered[i]
                    else "period recovered; classifier rejected it"
                ),
            }
        )
    missed_rows.sort(key=lambda r: (np.nan_to_num(r["true_snr"], nan=0.0)))

    true_snr = pd.to_numeric(meta_test["true_snr"], errors="coerce").to_numpy(dtype=float)
    importance = _permutation_importance(trained, X_test, y_test, seed=seed)

    return EvaluationResult(
        n_test=len(y_test),
        n_test_positive=int(y_test.sum()),
        positive_rate=float(y_test.mean()),
        chance_average_precision=float(y_test.mean()),
        model=model_curve,
        baselines=baseline_curves,
        threshold=trained.threshold,
        threshold_rule=trained.threshold_rule,
        cv_precision=trained.achieved_cv_precision,
        cv_recall=trained.achieved_cv_recall,
        confusion={
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        test_precision=precision,
        test_recall=recall,
        test_f1=f1,
        precision_at_k={
            f"model_top{top_k}": precision_at_k(y_test, scores, top_k),
            **{
                f"{b.name}_top{top_k}": precision_at_k(y_test, b.score(X_test), top_k)
                for b in BASELINES
            },
        },
        false_positive_breakdown=fp_breakdown,
        missed_breakdown=missed_rows,
        recall_by_snr=recall_by_snr(
            y_test, predicted, true_snr, np.array(snr_edges), recovered
        ),
        search_recovery=float(recovered[y_test == 1].mean()) if (y_test == 1).any() else float("nan"),
        bootstrap_win_rate_vs_best_baseline=win_rate,
        feature_importance=importance,
        calibration=calibration,
        shap_base_log_odds=trained.base_log_odds,
        shap_importance=shap_importance(shap_values),
        false_positives=false_positives,
    )


def shap_importance(shap_values: NDArray[np.float64]) -> list[dict[str, Any]]:
    """Mean absolute SHAP value per feature, largest first.

    The global view that complements permutation importance: how far, on
    average, knowing this feature moves a star's log-odds, rather than how
    much average precision is lost without it.  A feature can move many
    scores a long way and still cost little AP when shuffled, if another
    feature carries the same information.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    rows = [
        {"feature": name, "mean_abs_shap": float(v)} for name, v in zip(FEATURE_NAMES, mean_abs)
    ]
    rows.sort(key=lambda r: -r["mean_abs_shap"])
    return rows


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _permutation_importance(
    trained: TrainedModel,
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int_],
    *,
    seed: int,
    n_repeats: int = 10,
) -> list[dict[str, Any]]:
    """Permutation importance measured in average precision, not accuracy.

    Scored with ``average_precision`` for the same reason the headline metric
    is: permuting a feature barely moves accuracy at a 4% positive rate, so an
    accuracy-scored importance plot would be flat and meaningless.
    """
    result = permutation_importance(
        trained.estimator,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    rows = [
        {
            "feature": name,
            "importance": float(mean),
            "std": float(sd),
        }
        for name, mean, sd in zip(
            FEATURE_NAMES, result.importances_mean, result.importances_std
        )
    ]
    rows.sort(key=lambda r: -r["importance"])
    return rows


def format_report(result: EvaluationResult) -> str:
    """Human-readable summary, printed by ``run_pipeline.py``."""
    lines: list[str] = []
    add = lines.append
    add("=" * 72)
    add("HELD-OUT TEST SET RESULTS")
    add("=" * 72)
    add(
        f"test curves: {result.n_test}   planets: {result.n_test_positive}   "
        f"positive rate: {result.positive_rate:.3%}"
    )
    add("")
    add("Average precision (headline metric; chance = positive rate)")
    add("-" * 72)
    add(f"  {'random ranking (chance)':<34s} AP = {result.chance_average_precision:.4f}")
    for b in result.baselines:
        add(
            f"  {'baseline: ' + b.name:<34s} AP = {b.average_precision:.3f}"
            f"  [{b.ap_low:.3f}, {b.ap_high:.3f}]   (ROC-AUC {b.roc_auc:.3f})"
        )
    add(
        f"  {'model: gradient boosting':<34s} AP = {result.model.average_precision:.3f}"
        f"  [{result.model.ap_low:.3f}, {result.model.ap_high:.3f}]"
        f"   (ROC-AUC {result.model.roc_auc:.3f})"
    )
    add(f"  (brackets are 68% bootstrap intervals over {N_BOOTSTRAP} test-set resamples)")
    best_baseline = max(result.baselines, key=lambda b: b.average_precision)
    add("")
    add(
        f"  lift over chance:          {result.model.average_precision / result.chance_average_precision:6.1f}x"
    )
    add(
        f"  lift over best baseline:   {result.model.average_precision / best_baseline.average_precision:6.2f}x"
        f"  (best baseline: {best_baseline.name})"
    )
    add(
        f"  the model beats that baseline in {result.bootstrap_win_rate_vs_best_baseline:.1%}"
        " of paired bootstrap resamples"
    )
    add("")
    add("Operating point (threshold chosen by CV on the TRAINING split only)")
    add("-" * 72)
    add(f"  rule:      {result.threshold_rule}")
    add(f"  threshold: {result.threshold:.4f}")
    add(
        f"  CV on train:  precision = {result.cv_precision:.3f}   recall = {result.cv_recall:.3f}"
    )
    add(
        f"  HELD-OUT:     precision = {result.test_precision:.3f}   recall = {result.test_recall:.3f}"
        f"   F1 = {result.test_f1:.3f}"
    )
    add("")
    c = result.confusion
    add("  Confusion matrix (rows = truth, cols = prediction)")
    add("                    pred: no planet    pred: planet")
    add(f"    truth: no planet   {c['true_negative']:>10d}      {c['false_positive']:>10d}")
    add(f"    truth: planet      {c['false_negative']:>10d}      {c['true_positive']:>10d}")
    if result.false_positive_breakdown:
        detail = ", ".join(
            f"{k}: {v}" for k, v in sorted(result.false_positive_breakdown.items())
        )
        add(f"    false positives by true object type -> {detail}")
    add("")
    add("Precision@k (what a k-night follow-up programme would actually get)")
    add("-" * 72)
    for key, value in result.precision_at_k.items():
        add(f"  {key:<34s} {value:.3f}")
    add("")
    if result.calibration:
        lines.extend(format_calibration(result.calibration))
        add("")
    add("Completeness vs injected transit SNR (held-out planets)")
    add("-" * 72)
    add("  recall = P(search finds the period) x P(classifier keeps it | found)")
    add(f"  {'SNR bin':<16s}{'n':>4s}{'recall':>9s}{'search':>9s}{'classifier':>12s}")
    for row in result.recall_by_snr:
        hi = "inf" if row["snr_high"] > 1e8 else f"{row['snr_high']:.0f}"
        label = f"{row['snr_low']:.0f} - {hi}"
        if row["n_planets"] == 0:
            add(f"  {label:<16s}{0:>4d}{'n/a':>9s}{'n/a':>9s}{'n/a':>12s}")
            continue
        add(
            f"  {label:<16s}{row['n_planets']:>4d}{row['recall']:>9.2f}"
            f"{row['search_recovery']:>9.2f}{row['classifier_recall_given_search']:>12.2f}"
        )
    add(f"  overall search recovery on held-out planets: {result.search_recovery:.2f}")
    add("")
    add("Top features (permutation importance in average precision, on test)")
    add("-" * 72)
    for row in result.feature_importance[:8]:
        add(f"  {row['feature']:<28s} {row['importance']:+.4f} +/- {row['std']:.4f}")
    add("")
    if result.shap_importance:
        add("What moves a star's score (mean |SHAP| on test, in calibrated log-odds)")
        add("-" * 72)
        base = result.shap_base_log_odds
        add(
            f"  every star starts at log-odds {base:+.2f} (P = {1.0 / (1.0 + np.exp(-base)):.3f}); "
            "its SHAP values add up to its own"
        )
        for row in result.shap_importance[:8]:
            add(f"  {row['feature']:<28s} {row['mean_abs_shap']:.3f}")
        add("")
    if result.false_positives:
        add("False positives at the operating threshold, and what pushed each one up (SHAP)")
        add("-" * 72)
        for row in result.false_positives[:12]:
            add(
                f"  {row['target_id']:<12s} {row['kind']:<18s} score {row['model_score']:.3f}"
                f"   P(planet) {row['probability']:.3f}"
            )
            add(f"               -> {_format_reasons(row['pushed_up_by'])}")
        if len(result.false_positives) > 12:
            add(f"  ... and {len(result.false_positives) - 12} more")
        add("")
    add("Missed planets (false negatives at the operating threshold)")
    add("-" * 72)
    if not result.missed_breakdown:
        add("  none")
    for row in result.missed_breakdown[:12]:
        add(
            f"  {row['target_id']:<12s} SNR {row['true_snr']:6.1f}  depth {row['depth_ppm']:7.0f} ppm  "
            f"P {row['period_days']:6.2f} d  T14 {row['duration_hours']:5.2f} h  "
            f"b {row['impact_parameter']:.2f}  n_tr {row['n_transits_in_window']:.0f}"
        )
        add(
            f"               -> {row['reason']}"
            + (
                f"  (odd/even {row['odd_even_sigma']:.1f} sigma, "
                f"secondary {row['secondary_sigma']:.1f} sigma, "
                f"red-noise beta {row['red_noise_beta']:.2f})"
                if row["period_recovered"]
                else ""
            )
        )
        if row["period_recovered"] and row.get("pushed_down_by"):
            add(f"               pushed down by {_format_reasons(row['pushed_down_by'])}")
    if len(result.missed_breakdown) > 12:
        add(f"  ... and {len(result.missed_breakdown) - 12} more")
    add("=" * 72)
    return "\n".join(lines)


def _format_reasons(rows: list[dict[str, Any]]) -> str:
    """``feature = value (+0.84), ...`` for SHAP reason rows."""
    if not rows:
        return "nothing (no feature pushed this way)"
    return ", ".join(
        f"{r['feature']} = {r['value']:.3g} ({r['shap']:+.2f})" for r in rows
    )


def format_calibration(calibration: dict[str, Any]) -> list[str]:
    """The report's calibration block: the map, proper scores, counts and a reliability table."""
    lines: list[str] = []
    add = lines.append
    scaling = calibration["scaling"]
    add("Calibrated probability (Platt scaling fitted on out-of-fold training scores)")
    add("-" * 72)
    sign = "-" if scaling["intercept"] < 0 else "+"
    add(
        f"  P(planet) = 1 / (1 + exp(-({scaling['slope']:.3f} s {sign} "
        f"{abs(scaling['intercept']):.3f}))), s = the trees' log-odds"
    )
    add("  monotone in the score, so ranking, average precision and verdicts are unchanged")
    add(
        f"  the operating threshold is P(planet) = {calibration['threshold_as_probability']:.3f}"
        f" at the training planet rate ({scaling['train_positive_rate']:.2%})"
    )
    add("")
    add(f"  {'held-out probabilities':<30s}{'Brier':>9s}{'log loss':>11s}{'ECE':>9s}")
    labels = {
        "training_rate": "constant training planet rate",
        "uncalibrated": "score read as a probability",
        "calibrated": "calibrated",
    }
    for key, label in labels.items():
        row = calibration["scores"][key]
        add(f"  {label:<30s}{row['brier']:>9.4f}{row['log_loss']:>11.4f}{row['ece']:>9.4f}")
    expected = calibration["expected_planets"]
    add("")
    add(
        f"  planets the calibrated probabilities expect: {expected['all']:.1f} in the test set"
        f" ({expected['all_observed']} there),"
    )
    add(
        f"  {expected['flagged']:.1f} among the {expected['n_flagged']} flagged"
        f" ({expected['flagged_observed']} of them planets)"
    )
    add(
        f"  calibration slope on test {calibration['test_calibration_slope']:.2f}, intercept "
        f"{calibration['test_calibration_intercept']:+.2f}"
    )
    add("  (1 and 0 are ideal; a slope below 1 means the probabilities are too extreme)")
    add("")
    add(f"  {'calibrated P(planet)':<22s}{'stars':>7s}{'planets':>9s}{'mean P':>9s}{'observed':>10s}")
    for row in calibration["reliability"]["calibrated"]:
        label = f"{row['p_low']:.3g} - {row['p_high']:.3g}"
        if row["n"] == 0:
            add(f"  {label:<22s}{0:>7d}{0:>9d}{'':>9s}{'':>10s}")
            continue
        add(
            f"  {label:<22s}{row['n']:>7d}{row['n_planets']:>9d}"
            f"{row['mean_predicted']:>9.3f}{row['observed_rate']:>10.3f}"
        )
    return lines
