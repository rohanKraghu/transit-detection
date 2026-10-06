"""Exact SHAP values for the boosted trees, computed here from the fitted nodes.

A SHAP value (Lundberg & Lee 2017) splits one star's score into a part per
feature: ``base + sum_j phi_j = log-odds of this star``, with ``base`` the
average log-odds over the training stars.  ``phi_j`` is the Shapley value of
feature ``j`` in the game "how much does knowing this feature's value move the
expected score", averaged over every order in which the features could be
revealed.  Unlike the median-swap reasons ``vet`` used to give, the parts add
up exactly and do not depend on which baseline a feature is swapped to.

The expected score given a subset ``S`` of known features is the
path-dependent one TreeSHAP uses (Lundberg et al. 2020): walk each tree,
follow the star at a split on a known feature, and at a split on an unknown
feature take both branches weighted by how many training stars went each
way.  Written out, every leaf contributes

    value * prod over the path's splits of (follows(x) if feature known else cover fraction)

which, grouped by feature, is a product of one factor per distinct feature on
the path.  The Shapley value of a product has a closed form (coefficients of
``prod_k (c_k + t * o_k)``), so each leaf is handled in a few vectorised
lines over all stars at once.  With trees of depth 3 there are at most three
features per path.  This is the same quantity TreeSHAP's polynomial-time
recursion computes; ``tests/test_treeshap.py`` checks it against brute-force
enumeration over every feature subset and, when it is installed, against the
``shap`` package.

Done here rather than with ``shap`` because the package is a large dependency
for about eighty lines of arithmetic, and because reading the nodes directly
pins the semantics: NaN goes the way the tree learned (``missing_go_to_left``),
everything else goes left when ``x <= threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass(frozen=True)
class _Split:
    slot: int  # index into the leaf's distinct features
    feature: int
    threshold: float
    missing_go_to_left: bool
    goes_left: bool


@dataclass(frozen=True)
class _Leaf:
    value: float
    features: tuple[int, ...]  # distinct features on the root-to-leaf path
    cover: NDArray[np.float64]  # per distinct feature: product of its cover fractions
    splits: tuple[_Split, ...]


def _shapley_weights(m: int) -> NDArray[np.float64]:
    """``s! (m - s - 1)! / m!`` for coalitions of size ``s = 0 .. m-1``."""
    return np.array([factorial(s) * factorial(m - s - 1) / factorial(m) for s in range(m)])


class TreeExplainer:
    """SHAP values, in the trees' raw log-odds, for a fitted binary HGB classifier.

    ``expected_value`` is the base value: the classifier's average raw output
    over its training set under the trees' own cover weights.  For every row,
    ``expected_value + shap_values(X).sum(axis=1)`` equals
    ``estimator.decision_function(X)`` to rounding.
    """

    def __init__(self, estimator: HistGradientBoostingClassifier) -> None:
        if not hasattr(estimator, "_predictors"):
            raise ValueError("the estimator is not fitted")
        if getattr(estimator, "n_trees_per_iteration_", 1) != 1:
            raise ValueError("only binary classifiers (one tree per iteration) are supported")
        self.n_features = int(estimator.n_features_in_)
        self._leaves: list[_Leaf] = []
        expected = float(np.ravel(estimator._baseline_prediction)[0])
        for (predictor,) in estimator._predictors:
            nodes = predictor.nodes
            if np.any(nodes["is_categorical"][nodes["is_leaf"] == 0]):
                raise ValueError("categorical splits are not supported")
            self._collect(nodes, 0, [])
        for leaf in self._leaves:
            expected += leaf.value * float(np.prod(leaf.cover))
        self.expected_value = expected

    def _collect(
        self, nodes: NDArray, index: int, path: list[tuple[int, float, bool, bool, float]]
    ) -> None:
        """Walk one tree depth-first, recording every leaf with the splits above it."""
        node = nodes[index]
        if node["is_leaf"]:
            features: list[int] = []
            cover: list[float] = []
            splits: list[_Split] = []
            for feature, threshold, missing_left, goes_left, fraction in path:
                if feature not in features:
                    features.append(feature)
                    cover.append(1.0)
                slot = features.index(feature)
                cover[slot] *= fraction
                splits.append(_Split(slot, feature, threshold, missing_left, goes_left))
            self._leaves.append(
                _Leaf(float(node["value"]), tuple(features), np.array(cover), tuple(splits))
            )
            return
        parent = float(node["count"])
        feature = int(node["feature_idx"])
        threshold = float(node["num_threshold"])
        missing_left = bool(node["missing_go_to_left"])
        for child, goes_left in ((int(node["left"]), True), (int(node["right"]), False)):
            fraction = float(nodes[child]["count"]) / parent
            self._collect(
                nodes, child, path + [(feature, threshold, missing_left, goes_left, fraction)]
            )

    def shap_values(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n_rows, n_features)`` SHAP values in raw log-odds."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        n = X.shape[0]
        missing = np.isnan(X)
        phi = np.zeros((n, self.n_features))
        weights: dict[int, NDArray[np.float64]] = {}
        for leaf in self._leaves:
            m = len(leaf.features)
            if m == 0:
                continue  # a single-leaf tree: a constant, already in the base value
            follows = np.ones((m, n), dtype=bool)
            for split in leaf.splits:
                column = X[:, split.feature]
                left = np.where(
                    missing[:, split.feature], split.missing_go_to_left, column <= split.threshold
                )
                follows[split.slot] &= left if split.goes_left else ~left
            o = follows.astype(float)
            c = leaf.cover
            w = weights.setdefault(m, _shapley_weights(m))
            for j in range(m):
                # Coefficients of prod_{k != j} (c_k + t * o_k), lowest power first.
                poly = np.zeros((m, n))
                poly[0] = 1.0
                degree = 0
                for k in range(m):
                    if k == j:
                        continue
                    poly[1 : degree + 2] = poly[1 : degree + 2] * c[k] + poly[: degree + 1] * o[k]
                    poly[0] *= c[k]
                    degree += 1
                phi[:, leaf.features[j]] += leaf.value * (o[j] - c[j]) * (w @ poly)
        return phi


def top_reasons(
    values: NDArray[np.float64],
    x: NDArray[np.float64],
    feature_names: tuple[str, ...],
    n: int,
    direction: str = "either",
) -> list[dict[str, float | str]]:
    """The ``n`` largest SHAP values of one star, as ``{feature, value, shap}`` rows.

    ``direction`` picks which: ``"up"`` the features that pushed the score up
    most, ``"down"`` those that pushed it down most, ``"either"`` the largest
    in size.  Features whose SHAP value is zero, or points the other way, are
    never listed, so a row can come back shorter than ``n``.
    """
    values = np.asarray(values, dtype=float)
    if direction == "up":
        order = np.argsort(-values, kind="stable")
        keep = values[order] > 0
    elif direction == "down":
        order = np.argsort(values, kind="stable")
        keep = values[order] < 0
    elif direction == "either":
        order = np.argsort(-np.abs(values), kind="stable")
        keep = values[order] != 0
    else:
        raise ValueError(f"direction must be 'up', 'down' or 'either', not {direction!r}")
    return [
        {"feature": feature_names[j], "value": float(x[j]), "shap": float(values[j])}
        for j in order[keep][:n]
    ]
