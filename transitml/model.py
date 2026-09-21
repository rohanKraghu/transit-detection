"""The classifier, the baselines it has to beat, and the train/test protocol.

Why gradient boosting on hand-built features rather than a 1D CNN on folded
light curves:

* **Sample size.**  The demo has ~96 positives.  AstroNet (Shallue & Vanderburg
  2018) trained on ~15,000 labelled Kepler TCEs; ExoMiner used ~35,000.  A CNN
  with 10^5-10^6 parameters trained on 96 positives memorises them.  Gradient
  boosting on 23 features is in the right regime for this much data, and the
  same argument applies to any real single-sector, single-campaign study.
* **The hard work is already done by BLS.**  The SNR gain in transit detection
  comes from phase-folding N transits together.  BLS does that optimally for a
  box model.  A CNN on unfolded curves has to rediscover folding from data; a
  CNN on *folded* curves needs BLS first anyway, at which point the expensive
  step is shared and the only question is what to do with the fold.
* **Auditability.**  Every feature maps onto a test a human vetter or the
  Kepler Robovetter applies (odd/even, secondary, shape, duration-density
  consistency).  When the model says "no", the permutation importances say why.

The honest counterpoint is in the README: with 10^5 real light curves a CNN on
global+local folded views does beat this, because it picks up transit shape
detail that a handful of scalars throws away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data.loader import Dataset
from .features import FEATURE_NAMES


@dataclass
class Split:
    """A stratified train/test split of a :class:`Dataset`.

    The test set is touched exactly once, at the very end, by
    :func:`transitml.evaluate.evaluate`.  Model selection and the operating
    threshold both come from cross-validation *inside* ``train``.
    """

    X_train: NDArray[np.float64]
    y_train: NDArray[np.int_]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int_]
    train_index: NDArray[np.int_]
    test_index: NDArray[np.int_]

    @property
    def n_train_positive(self) -> int:
        return int(self.y_train.sum())

    @property
    def n_test_positive(self) -> int:
        return int(self.y_test.sum())


def make_split(dataset: Dataset, *, test_size: float, seed: int) -> Split:
    """Stratified split.

    Stratification is not optional here: at a 4% positive rate an unstratified
    split can easily hand the test set 20% more or fewer positives than the
    train set purely by chance, which moves average precision around more than
    any modelling decision does.
    """
    index = np.arange(len(dataset))
    train_index, test_index = train_test_split(
        index, test_size=test_size, random_state=seed, stratify=dataset.y
    )
    X, y = dataset.X, dataset.y
    return Split(
        X_train=X[train_index],
        y_train=y[train_index],
        X_test=X[test_index],
        y_test=y[test_index],
        train_index=train_index,
        test_index=test_index,
    )


def build_model(seed: int) -> HistGradientBoostingClassifier:
    """The classifier.

    ``class_weight="balanced"`` mainly affects calibration rather than ranking,
    but it stops the trees from spending their capacity on the 96% majority and
    it makes the predicted probabilities land in a range where a precision
    target is actually reachable.

    ``HistGradientBoostingClassifier`` handles NaN natively by learning a
    default direction at each split, which is what we want: a missing
    odd/even test means "too few transits to run the test", and that is
    information, not something to impute away.

    Hyperparameters are deliberately conservative (shallow trees, strong L2,
    large leaves) because with ~60 training positives anything else overfits.
    """
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_iter=300,
        max_depth=3,
        max_leaf_nodes=8,
        min_samples_leaf=15,
        l2_regularization=1.0,
        max_features=0.8,
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )


class ScoreFunction(Protocol):
    """Anything that turns a feature matrix into a ranking score."""

    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass
class Baseline:
    """A single-feature heuristic used as a ranking score.

    These are the honest points of comparison.  A transit search *is* a
    thresholding problem at heart, and any ML result that does not beat "rank
    by BLS signal-to-noise" is not worth the dependency.
    """

    name: str
    feature: str
    description: str
    sign: float = 1.0

    @property
    def column(self) -> int:
        return FEATURE_NAMES.index(self.feature)

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ranking score; NaN is mapped to -inf so untestable curves rank last."""
        values = self.sign * X[:, self.column].astype(float)
        return np.where(np.isfinite(values), values, -np.inf)


#: The two baselines.  ``depth_threshold`` is the naive heuristic the brief asks
#: for; ``bls_snr`` is the much stronger thing a working astronomer would
#: actually do, and is the number the model has to beat to justify itself.
BASELINES: tuple[Baseline, ...] = (
    Baseline(
        name="depth_threshold",
        feature="log_depth",
        description="Rank by the depth of the best BLS box. The naive heuristic: "
        "deeper dip = more likely a planet.",
    ),
    Baseline(
        name="bls_snr",
        feature="bls_depth_snr",
        description="Rank by BLS depth signal-to-noise. The classical "
        "single-statistic transit search, and a genuinely strong baseline.",
    ),
)


def cross_val_scores(
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    *,
    n_folds: int,
    seed: int,
) -> NDArray[np.float64]:
    """Out-of-fold positive-class probabilities over the *training* set.

    These are what the operating threshold is chosen from.  Choosing a
    threshold on in-sample predictions would pick a threshold that is optimistic
    by construction; choosing it on the test set would leak the test set into
    the reported confusion matrix.  Out-of-fold predictions on train are the
    only option that is neither.
    """
    oof = np.zeros(len(y), dtype=float)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in folds.split(X, y):
        model = build_model(seed)
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return oof


@dataclass
class TrainedModel:
    """A fitted classifier plus everything chosen on training data alone."""

    estimator: HistGradientBoostingClassifier
    oof_scores: NDArray[np.float64]
    threshold: float
    threshold_rule: str
    achieved_cv_precision: float
    achieved_cv_recall: float

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.estimator.predict_proba(X)[:, 1]


def select_threshold(
    y: NDArray[np.int_], scores: NDArray[np.float64], target_precision: float
) -> tuple[float, str, float, float]:
    """Pick the operating point: highest recall subject to precision >= target.

    Returns ``(threshold, rule, precision, recall)``.

    Why a precision floor rather than "maximise F1" or "0.5"?  Because the cost
    structure is asymmetric and known.  Each candidate above the threshold buys
    a follow-up campaign -- ground-based seeing-limited photometry to confirm
    the event is on-target, then high-resolution imaging, then radial
    velocities: nights of telescope time per candidate.  A missed planet costs
    nothing today; it is still in the archive and the next sector's data will
    find it.  So the sensible policy is to cap the waste rate and take whatever
    recall that buys.  ``target_precision = 0.5`` means at most one wasted
    campaign per confirmed planet.

    A default 0.5 probability threshold has no such justification, and under
    ``class_weight="balanced"`` it does not even correspond to a 50% posterior.
    """
    order = np.argsort(-scores)
    y_sorted = y[order]
    s_sorted = scores[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)
    precision = tp / k
    n_pos = int(y.sum())
    recall = tp / n_pos if n_pos else np.zeros_like(tp, dtype=float)

    ok = precision >= target_precision
    if ok.any():
        # Deepest point down the ranked list that still satisfies the floor.
        best = int(np.flatnonzero(ok)[-1])
        rule = f"max recall subject to CV precision >= {target_precision:.2f}"
    else:
        # Unreachable target: fall back to the most precise point that still
        # returns a usable number of candidates, and say so loudly.
        usable = k >= 5
        best = int(np.argmax(np.where(usable, precision, -np.inf)))
        rule = (
            f"target precision {target_precision:.2f} unreachable in CV; "
            "fell back to the most precise point with >= 5 candidates"
        )
    threshold = float(s_sorted[best])
    return threshold, rule, float(precision[best]), float(recall[best])


def train(
    split: Split, *, n_folds: int, seed: int, target_precision: float
) -> TrainedModel:
    """Cross-validate on train, choose the threshold, then refit on all of train.

    The test set is not read anywhere in this function.
    """
    oof = cross_val_scores(split.X_train, split.y_train, n_folds=n_folds, seed=seed)
    threshold, rule, precision, recall = select_threshold(
        split.y_train, oof, target_precision
    )
    estimator = build_model(seed)
    estimator.fit(split.X_train, split.y_train)
    return TrainedModel(
        estimator=estimator,
        oof_scores=oof,
        threshold=threshold,
        threshold_rule=rule,
        achieved_cv_precision=precision,
        achieved_cv_recall=recall,
    )
