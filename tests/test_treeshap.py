"""SHAP values for the boosted trees: exact, additive, and the same as the reference."""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from transitml.features import FEATURE_NAMES
from transitml.treeshap import TreeExplainer, top_reasons


@pytest.fixture(scope="module")
def small_model():
    """Five features, missing values in training, shallow trees: brute force stays cheap."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5))
    y = ((X[:, 0] + 0.8 * X[:, 1] * X[:, 2] + 0.3 * rng.normal(size=600)) > 0.6).astype(int)
    X[rng.random(X.shape) < 0.12] = np.nan
    model = HistGradientBoostingClassifier(
        max_iter=15, max_depth=3, max_leaf_nodes=8, learning_rate=0.3, random_state=0
    ).fit(X, y)
    probe = rng.normal(size=(12, 5))
    probe[rng.random(probe.shape) < 0.25] = np.nan
    probe[0] = np.nan  # a star with nothing measured
    return model, X, probe


def _expected_output(nodes, x, known: frozenset[int]) -> float:
    """E[tree(x) | features in ``known``]: follow x on known splits, weight by cover otherwise."""

    def walk(index: int) -> float:
        node = nodes[index]
        if node["is_leaf"]:
            return float(node["value"])
        feature = int(node["feature_idx"])
        left, right = int(node["left"]), int(node["right"])
        if feature in known:
            value = x[feature]
            goes_left = bool(node["missing_go_to_left"]) if np.isnan(value) else value <= node["num_threshold"]
            return walk(left if goes_left else right)
        total = float(node["count"])
        return (
            float(nodes[left]["count"]) * walk(left) + float(nodes[right]["count"]) * walk(right)
        ) / total

    return walk(0)


def _brute_force_shap(model, x) -> np.ndarray:
    """Shapley values by enumerating every subset of the features."""
    n = x.size
    trees = [p[0].nodes for p in model._predictors]

    def v(subset) -> float:
        known = frozenset(subset)
        return sum(_expected_output(nodes, x, known) for nodes in trees)

    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                phi[i] += weight * (v(subset + (i,)) - v(subset))
    return phi


def test_matches_brute_force_enumeration(small_model):
    model, _, probe = small_model
    explainer = TreeExplainer(model)
    values = explainer.shap_values(probe)
    for row in range(probe.shape[0]):
        np.testing.assert_allclose(values[row], _brute_force_shap(model, probe[row]), atol=1e-12)


def test_values_add_up_to_the_raw_score(small_model):
    model, X, probe = small_model
    explainer = TreeExplainer(model)
    for data in (X, probe):
        values = explainer.shap_values(data)
        np.testing.assert_allclose(
            explainer.expected_value + values.sum(axis=1), model.decision_function(data), atol=1e-12
        )


def test_base_value_is_the_mean_training_output(small_model):
    """With every feature unknown, the cover-weighted walk averages over the training rows."""
    model, X, _ = small_model
    explainer = TreeExplainer(model)
    assert explainer.expected_value == pytest.approx(model.decision_function(X).mean(), abs=1e-10)


def test_matches_the_shap_package(small_model):
    shap = pytest.importorskip("shap")
    model, X, probe = small_model
    reference = shap.TreeExplainer(model)
    for data in (X[:200], probe):
        np.testing.assert_allclose(
            TreeExplainer(model).shap_values(data), np.asarray(reference.shap_values(data)), atol=1e-10
        )
    assert TreeExplainer(model).expected_value == pytest.approx(
        float(np.ravel(reference.expected_value)[0]), abs=1e-10
    )


def test_a_feature_no_tree_uses_gets_exactly_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    X[:, 2] = 1.0  # constant: never split on
    y = (X[:, 0] > 0.3).astype(int)
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=2, random_state=0).fit(X, y)
    values = TreeExplainer(model).shap_values(X)
    assert np.all(values[:, 2] == 0.0)


def test_refuses_multiclass_and_unfitted_models():
    with pytest.raises(ValueError, match="not fitted"):
        TreeExplainer(HistGradientBoostingClassifier())
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 2))
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    model = HistGradientBoostingClassifier(max_iter=3).fit(X, y)
    with pytest.raises(ValueError, match="binary"):
        TreeExplainer(model)


def test_the_pipeline_model_explains_its_calibrated_log_odds(tiny_trained):
    split, trained = tiny_trained
    X = split.X_test.copy()
    X[::3, FEATURE_NAMES.index("odd_even_sigma")] = np.nan  # too few transits to test
    X[1::4, :] = np.nan  # nothing measured at all
    values = trained.explain(X)
    assert values.shape == X.shape
    np.testing.assert_allclose(
        trained.base_log_odds + values.sum(axis=1), trained.log_odds(X), atol=1e-9
    )


def test_top_reasons_pick_by_direction():
    values = np.array([0.5, -1.0, 0.0, 2.0, -0.2])
    x = np.arange(5.0)
    names = tuple(f"f{i}" for i in range(5))
    assert [r["feature"] for r in top_reasons(values, x, names, 3, "up")] == ["f3", "f0"]
    assert [r["feature"] for r in top_reasons(values, x, names, 3, "down")] == ["f1", "f4"]
    assert [r["feature"] for r in top_reasons(values, x, names, 3)] == ["f3", "f1", "f0"]
    assert top_reasons(values, x, names, 1, "up")[0] == {"feature": "f3", "value": 3.0, "shap": 2.0}
    with pytest.raises(ValueError, match="direction"):
        top_reasons(values, x, names, 1, "sideways")


def test_feature_names_line_up_with_the_model(tiny_trained):
    _, trained = tiny_trained
    assert trained.explainer.n_features == len(FEATURE_NAMES)
