"""Evaluation hygiene: the test set is read once, and nothing is chosen on it.

Two things can silently inflate a result in a problem like this, and both are
invisible in the output if you do not test for them:

1. the operating threshold being tuned on the data it is then scored on;
2. ground-truth columns leaking into the feature matrix.

Both are asserted here directly rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from transitml.data.loader import Dataset
from transitml.evaluate import (
    evaluate,
    period_recovered,
    precision_at_k,
    recall_by_snr,
    score_curve,
)
from transitml.features import FEATURE_NAMES
from transitml.model import (
    BASELINES,
    make_split,
    select_threshold,
    train,
    wilson_lower_bound,
)


@pytest.fixture(scope="module")
def toy_dataset() -> Dataset:
    """A separable-but-noisy toy problem at a 4% positive rate.

    Deliberately synthetic at the *feature* level: these tests are about the
    evaluation protocol, not about astrophysics, and running the real BLS here
    would make them slow without testing anything extra.
    """
    rng = np.random.default_rng(0)
    n, n_pos = 800, 32
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_pos, replace=False)] = 1

    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    # Make three features carry real signal, with heavy overlap.
    X[labels == 1, 0] += 1.6
    X[labels == 1, 1] += 1.2
    X[labels == 1, 2] += 0.9

    features = pd.DataFrame(X, columns=list(FEATURE_NAMES))
    meta = pd.DataFrame(
        {
            "target_id": [f"T{i:04d}" for i in range(n)],
            "kind": np.where(labels == 1, "planet", "noise"),
            "label": labels,
            "true_snr": np.where(labels == 1, rng.uniform(5, 60, n), 0.0),
            "period": rng.uniform(1, 12, n),
            "epoch": rng.uniform(0, 5, n),
            "depth": rng.uniform(1e-4, 1e-2, n),
            "duration_t14": rng.uniform(0.03, 0.2, n),
            "impact_parameter": rng.uniform(0, 0.9, n),
            "n_transits_in_window": rng.integers(2, 12, n),
        }
    )
    return Dataset(features=features, labels=labels, meta=meta)


@pytest.fixture(scope="module")
def trained_pair(toy_dataset):
    split = make_split(toy_dataset, test_size=0.35, seed=0)
    model = train(split, n_folds=5, seed=0, target_precision=0.5)
    return split, model


def test_split_is_stratified_and_disjoint(toy_dataset):
    split = make_split(toy_dataset, test_size=0.35, seed=0)
    train_set, test_set = set(split.train_index), set(split.test_index)

    assert train_set.isdisjoint(test_set)
    assert len(train_set | test_set) == len(toy_dataset)
    assert split.y_train.mean() == pytest.approx(toy_dataset.positive_rate, abs=0.005)
    assert split.y_test.mean() == pytest.approx(toy_dataset.positive_rate, abs=0.005)


def test_threshold_is_chosen_without_the_test_set(toy_dataset, trained_pair):
    """Scrambling the held-out labels must not move the operating threshold.

    If the threshold depended on the test set at all, this would change it.
    """
    split, model = trained_pair
    scrambled = np.array(split.y_test, copy=True)
    np.random.default_rng(1).shuffle(scrambled)
    tampered = type(split)(
        X_train=split.X_train,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=scrambled,
        train_index=split.train_index,
        test_index=split.test_index,
    )
    retrained = train(tampered, n_folds=5, seed=0, target_precision=0.5)

    assert retrained.threshold == model.threshold
    assert retrained.threshold_rule == model.threshold_rule
    assert retrained.achieved_cv_precision == model.achieved_cv_precision


def _same_rows(a: list[dict], b: list[dict]) -> bool:
    """Compare rows that may contain NaN, which never equals itself."""
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if set(left) != set(right):
            return False
        for key in left:
            x, y = left[key], right[key]
            if isinstance(x, float) and isinstance(y, float):
                if not (x == y or (np.isnan(x) and np.isnan(y))):
                    return False
            elif x != y:
                return False
    return True


def _with_rows_replaced(dataset: Dataset, rows, value: float) -> Dataset:
    """A copy of ``dataset`` with the given rows overwritten, features and meta."""
    matrix = dataset.features.to_numpy(dtype=float, copy=True)
    matrix[rows] = value
    features = pd.DataFrame(matrix, columns=list(dataset.features.columns))
    meta = dataset.meta.copy()
    meta.loc[rows, "true_snr"] = -999.0
    return Dataset(features=features, labels=dataset.labels, meta=meta)


def test_metrics_use_test_rows_only(toy_dataset, trained_pair):
    """Corrupting the training rows must not change a single reported number.

    The split is rebuilt from the corrupted dataset, so this covers the whole
    path -- if any training row could reach the evaluation, through the split,
    the feature matrix or the ground-truth table, it would show up here.
    """
    split, model = trained_pair
    baseline_result = evaluate(toy_dataset, split, model, top_k=20, seed=0, n_bootstrap=0)

    tampered_dataset = _with_rows_replaced(toy_dataset, split.train_index, np.nan)
    tampered_split = make_split(tampered_dataset, test_size=0.35, seed=0)

    np.testing.assert_array_equal(tampered_split.test_index, split.test_index)
    np.testing.assert_array_equal(tampered_split.X_test, split.X_test)

    tampered_result = evaluate(tampered_dataset, tampered_split, model, top_k=20, seed=0, n_bootstrap=0)

    assert tampered_result.model.average_precision == baseline_result.model.average_precision
    assert tampered_result.confusion == baseline_result.confusion
    assert _same_rows(tampered_result.recall_by_snr, baseline_result.recall_by_snr)
    assert tampered_result.precision_at_k == baseline_result.precision_at_k
    assert tampered_result.n_test == len(split.y_test)
    assert tampered_result.n_test + len(split.y_train) == len(toy_dataset)


def test_corrupting_the_test_rows_does_change_the_metrics(toy_dataset, trained_pair):
    """The complement of the previous test: it is not vacuously passing."""
    split, model = trained_pair
    baseline_result = evaluate(toy_dataset, split, model, top_k=20, seed=0, n_bootstrap=0)

    tampered_dataset = _with_rows_replaced(toy_dataset, split.test_index, 0.0)
    tampered_split = make_split(tampered_dataset, test_size=0.35, seed=0)
    tampered_result = evaluate(tampered_dataset, tampered_split, model, top_k=20, seed=0, n_bootstrap=0)

    assert tampered_result.model.average_precision != baseline_result.model.average_precision


def test_ground_truth_cannot_reach_the_feature_matrix(toy_dataset):
    """A Dataset whose features overlap the ground-truth columns is rejected."""
    leaky = toy_dataset.features.copy()
    leaky["true_snr"] = toy_dataset.meta["true_snr"].to_numpy()
    with pytest.raises(ValueError, match="leaked"):
        Dataset(features=leaky, labels=toy_dataset.labels, meta=toy_dataset.meta)

    # And the model input is restricted to the declared feature list.
    assert toy_dataset.X.shape == (len(toy_dataset), len(FEATURE_NAMES))


def test_average_precision_beats_chance_and_the_baselines(toy_dataset, trained_pair):
    split, model = trained_pair
    result = evaluate(toy_dataset, split, model, top_k=20, seed=0, n_bootstrap=0)

    assert result.chance_average_precision == pytest.approx(float(split.y_test.mean()))
    assert result.model.average_precision > 3 * result.chance_average_precision
    assert len(result.baselines) == len(BASELINES)
    for baseline in result.baselines:
        assert 0.0 <= baseline.average_precision <= 1.0


def test_bootstrap_interval_brackets_the_point_estimate(toy_dataset, trained_pair):
    """The headline metric is quoted with an interval, and it has to contain it."""
    split, model = trained_pair
    result = evaluate(toy_dataset, split, model, top_k=20, seed=0, n_bootstrap=300)

    assert result.model.ap_low < result.model.average_precision < result.model.ap_high
    assert result.model.ap_high - result.model.ap_low > 0.0
    assert 0.0 <= result.bootstrap_win_rate_vs_best_baseline <= 1.0


def test_fast_average_precision_matches_sklearn():
    """The bootstrap's cheap estimator must agree with the reference exactly."""
    from transitml.evaluate import fast_average_precision

    rng = np.random.default_rng(0)
    for _ in range(20):
        y = (rng.random(400) < 0.05).astype(int)
        if y.sum() < 2:
            continue
        # Deliberately include ties: bootstrap resampling duplicates rows.
        scores = np.round(rng.normal(size=400) + 1.5 * y, 2)
        assert fast_average_precision(y, scores) == pytest.approx(
            average_precision_score(y, scores), rel=1e-12
        )


def test_bootstrap_resamples_are_shared_between_scorers():
    """Paired resampling: the same bootstrap worlds score every scorer."""
    from transitml.evaluate import bootstrap_indices

    first = bootstrap_indices(50, 8, seed=3)
    second = bootstrap_indices(50, 8, seed=3)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 50)
    assert first.min() >= 0 and first.max() < 50


def test_average_precision_endpoints():
    """Sanity-check the metric itself against its two known values."""
    y = np.zeros(1000, dtype=int)
    y[:40] = 1
    rng = np.random.default_rng(0)

    perfect = np.where(y == 1, 1.0, 0.0) + rng.normal(0, 1e-6, y.size)
    assert score_curve("p", "", y, perfect).average_precision == pytest.approx(1.0, abs=1e-3)

    random_scores = rng.normal(size=y.size)
    chance = score_curve("r", "", y, random_scores).average_precision
    assert chance == pytest.approx(y.mean(), abs=0.03)

    # The point of the whole exercise: a constant "no planet" classifier is 96%
    # accurate and has zero value, which average precision reflects and
    # accuracy does not.
    constant_accuracy = float((y == 0).mean())
    assert constant_accuracy > 0.95
    assert average_precision_score(y, np.zeros_like(y, dtype=float)) == pytest.approx(
        y.mean(), abs=1e-9
    )


def test_select_threshold_respects_the_precision_floor():
    y = np.array([1] * 20 + [0] * 480)
    rng = np.random.default_rng(3)
    scores = np.where(y == 1, rng.uniform(0.6, 1.0, y.size), rng.uniform(0.0, 0.7, y.size))

    threshold, rule, precision, recall = select_threshold(y, scores, target_precision=0.5)
    assert precision >= 0.5
    assert "max recall" in rule
    predicted = scores >= threshold
    realised = float(y[predicted].mean())
    assert realised >= 0.5
    assert recall > 0


def test_select_threshold_falls_back_loudly_when_unreachable():
    """An unreachable target must be reported, not silently ignored."""
    y = np.array([1] * 5 + [0] * 495)
    scores = np.random.default_rng(0).uniform(size=y.size)
    _, rule, _, _ = select_threshold(y, scores, target_precision=0.99)
    assert "unreachable" in rule


def test_wilson_lower_bound_is_below_the_point_estimate_and_tightens_with_n():
    tp = np.array([1.0, 5.0, 50.0, 10.0, 0.0])
    n = np.array([1.0, 10.0, 100.0, 10.0, 4.0])
    point = tp / n
    lower = wilson_lower_bound(tp, n, z=1.0)
    assert np.all(lower <= point + 1e-12)
    assert np.all(lower >= 0.0)
    # Same 0.5 precision: the bound is tighter at 100 candidates than at 10.
    assert lower[2] > lower[1]
    assert wilson_lower_bound(tp, n, z=0.0) == pytest.approx(point)
    # 1-sigma Wilson bound for 5/10, by hand: (0.55 - sqrt(0.025 + 0.0025)) / 1.1
    assert lower[1] == pytest.approx((0.55 - np.sqrt(0.0275)) / 1.1, rel=1e-12)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("target", [0.3, 0.5, 0.7])
def test_lcb_rule_never_picks_a_lower_threshold_than_the_point_rule(seed, target):
    """Requiring the lower bound to clear the floor can only move the threshold up."""
    rng = np.random.default_rng(seed)
    n_pos = int(rng.integers(10, 60))
    y = np.array([1] * n_pos + [0] * (1500 - n_pos))
    separation = rng.uniform(0.5, 3.0)
    scores = rng.normal(size=y.size) + separation * y

    point_thr, point_rule, _, _ = select_threshold(y, scores, target, precision_lcb_z=0.0)
    for z in (0.5, 1.0, 2.0):
        lcb_thr, lcb_rule, _, _ = select_threshold(y, scores, target, precision_lcb_z=z)
        assert lcb_thr >= point_thr, (z, lcb_rule, point_rule)


def test_lcb_rule_on_the_trained_oof_scores(trained_pair):
    """The same guarantee on real out-of-fold scores, and the rule says what it did."""
    split, model = trained_pair
    point_thr, point_rule, _, _ = select_threshold(
        split.y_train, model.oof_scores, 0.5, precision_lcb_z=0.0
    )
    assert "lower bound" not in point_rule
    assert model.threshold >= point_thr
    assert "Wilson" in model.threshold_rule


def test_precision_at_k_and_recall_by_snr():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert precision_at_k(y, scores, 2) == pytest.approx(1.0)
    assert precision_at_k(y, scores, 4) == pytest.approx(0.5)
    assert precision_at_k(y, scores, 100) == pytest.approx(y.mean())

    predicted = scores >= 0.55
    snr = np.array([50.0, 30.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    rows = recall_by_snr(y, predicted, snr, np.array([0.0, 10.0, 100.0]))
    assert rows[0]["n_planets"] == 1 and rows[0]["recall"] == pytest.approx(0.0)
    assert rows[1]["n_planets"] == 2 and rows[1]["recall"] == pytest.approx(1.0)


def test_period_recovery_accepts_low_order_aliases():
    """Half, double and triple the true period all fold the transits together."""
    injected = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    found = np.array([4.0, 2.0, 8.0, 12.0, 4.4, 3.1])
    got = period_recovered(found, injected)
    assert got.tolist() == [True, True, True, True, False, False]
    # A NaN period (no search result) is never a recovery.
    assert not period_recovered(np.array([np.nan]), np.array([4.0]))[0]


def test_recall_is_decomposed_into_search_and_classifier(toy_dataset, trained_pair):
    """recall = P(search finds it) x P(classifier keeps it | found)."""
    split, model = trained_pair
    result = evaluate(toy_dataset, split, model, top_k=20, seed=0, n_bootstrap=0)
    assert 0.0 <= result.search_recovery <= 1.0
    for row in result.recall_by_snr:
        if row["n_planets"] == 0:
            continue
        product = row["search_recovery"] * row["classifier_recall_given_search"]
        if np.isfinite(product):
            assert row["recall"] == pytest.approx(product, abs=1e-9)


def test_permutation_importance_is_reported(toy_dataset, trained_pair):
    split, model = trained_pair
    result = evaluate(toy_dataset, split, model, top_k=20, seed=0, n_bootstrap=0)
    importance = result.feature_importance

    assert len(importance) == len(FEATURE_NAMES)
    assert importance == sorted(importance, key=lambda r: -r["importance"])
    # The three features that actually carry signal should lead.
    top = {row["feature"] for row in importance[:5]}
    assert top & set(FEATURE_NAMES[:3])


def test_results_serialise(toy_dataset, trained_pair):
    import json

    split, model = trained_pair
    payload = evaluate(toy_dataset, split, model, top_k=20, seed=0, n_bootstrap=0).to_dict()
    json.loads(json.dumps(payload, default=str))
    assert payload["operating_point"]["confusion_matrix"]["true_positive"] >= 0
    assert set(payload["operating_point"]) >= {
        "threshold",
        "rule",
        "test_precision",
        "test_recall",
        "confusion_matrix",
    }
