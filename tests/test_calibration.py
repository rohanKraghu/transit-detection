"""Calibrated probabilities: Platt scaling on out-of-fold scores, and how it is scored."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.metrics import log_loss as sklearn_log_loss

from transitml.calibration import (
    PlattScaling,
    brier_score,
    calibration_summary,
    expected_calibration_error,
    fit_platt,
    log_loss,
    reliability_table,
)
from transitml.model import cross_val_raw_scores, cross_val_scores


def test_platt_recovers_a_known_logistic_map():
    rng = np.random.default_rng(0)
    raw = rng.normal(-3.0, 2.0, size=200_000)
    y = (rng.random(raw.size) < expit(0.6 * raw - 1.1)).astype(int)
    fit = fit_platt(raw, y)
    assert fit.slope == pytest.approx(0.6, abs=0.01)
    assert fit.intercept == pytest.approx(-1.1, abs=0.02)
    assert fit.train_positive_rate == pytest.approx(y.mean())


def test_platt_matches_scikit_learns_sigmoid_calibration():
    """Same smoothed targets, same optimum; sklearn writes P = 1 / (1 + exp(a s + b))."""
    sigmoid = pytest.importorskip("sklearn.calibration")._sigmoid_calibration
    rng = np.random.default_rng(1)
    raw = rng.normal(-2.0, 2.5, size=900)
    y = (rng.random(raw.size) < expit(0.8 * raw + 0.3)).astype(int)
    a, b = sigmoid(raw, y)
    fit = fit_platt(raw, y)
    assert fit.slope == pytest.approx(-a, rel=1e-4)
    assert fit.intercept == pytest.approx(-b, rel=1e-4, abs=1e-6)


def test_platt_is_strictly_increasing_even_on_useless_scores():
    """Scores that anti-correlate with the labels would want a negative slope; it is floored."""
    raw = np.linspace(-3, 3, 60)
    y = (raw < 0).astype(int)
    fit = fit_platt(raw, y)
    assert fit.slope > 0
    p = fit.probability(raw)
    assert np.all(np.diff(p) > 0)


def test_platt_refuses_one_class_and_non_finite_scores():
    with pytest.raises(ValueError, match="both classes"):
        fit_platt(np.zeros(5), np.zeros(5, dtype=int))
    with pytest.raises(ValueError, match="finite"):
        fit_platt(np.array([0.0, np.inf]), np.array([0, 1]))


def test_planet_rate_shift_is_bayes_rule():
    scaling = PlattScaling(slope=0.7, intercept=-1.0, train_positive_rate=0.04)
    raw = np.array([-6.0, 0.0, 4.0])
    base = scaling.log_odds(raw)
    shifted = scaling.log_odds(raw, positive_rate=0.5)
    np.testing.assert_allclose(shifted - base, logit(0.5) - logit(0.04))
    np.testing.assert_allclose(scaling.log_odds(raw, positive_rate=0.04), base)
    with pytest.raises(ValueError, match="positive_rate"):
        scaling.probability(raw, positive_rate=1.0)


def test_cross_val_scores_are_the_logistic_of_the_raw_scores_bit_for_bit():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + rng.normal(size=300) > 1.2).astype(int)
    raw = cross_val_raw_scores(X, y, n_folds=3, seed=0)
    np.testing.assert_array_equal(cross_val_scores(X, y, n_folds=3, seed=0), expit(raw))


def test_trained_model_probabilities_keep_the_ranking(tiny_trained):
    split, trained = tiny_trained
    score = trained.score(split.X_test)
    probability = trained.probability(split.X_test)
    np.testing.assert_array_equal(np.argsort(score, kind="stable"), np.argsort(probability, kind="stable"))
    assert average_precision_score(split.y_test, probability) == average_precision_score(
        split.y_test, score
    )
    np.testing.assert_array_equal(score, trained.estimator.predict_proba(split.X_test)[:, 1])
    # The threshold maps to the probability of a star sitting exactly on it.
    on_threshold = trained.threshold_probability
    flagged = score >= trained.threshold
    assert np.all(probability[flagged] >= on_threshold - 1e-12)
    assert np.all(probability[~flagged] < on_threshold + 1e-12)


def test_calibration_is_fitted_on_out_of_fold_scores(tiny_trained):
    split, trained = tiny_trained
    refit = fit_platt(logit(np.clip(trained.oof_scores, 1e-300, 1 - 1e-16)), split.y_train)
    assert trained.calibration.slope == pytest.approx(refit.slope, rel=1e-6)
    assert trained.calibration.intercept == pytest.approx(refit.intercept, rel=1e-6, abs=1e-9)


def test_proper_scores_match_scikit_learn():
    rng = np.random.default_rng(3)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    assert brier_score(y, p) == pytest.approx(brier_score_loss(y, p))
    assert log_loss(y, p) == pytest.approx(sklearn_log_loss(y, p))


def test_reliability_table_counts_every_star_once():
    rng = np.random.default_rng(4)
    p = np.r_[rng.random(300) ** 4, 0.0, 1.0]
    y = (rng.random(p.size) < p).astype(int)
    rows = reliability_table(y, p)
    assert sum(r["n"] for r in rows) == p.size
    assert sum(r["n_planets"] for r in rows) == y.sum()


def test_ece_is_zero_when_every_bin_predicts_its_own_rate():
    y = np.array([0] * 98 + [1] * 2 + [0] * 6 + [1] * 4)
    p = np.array([0.02] * 100 + [0.4] * 10)
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)


def test_summary_counts_expected_planets_and_scores_three_forecasts():
    rng = np.random.default_rng(5)
    raw = rng.normal(-3, 2, 4000)
    scaling = PlattScaling(1.0, 0.0, train_positive_rate=0.05)
    p = scaling.probability(raw)
    y = (rng.random(raw.size) < p).astype(int)
    flagged = p > 0.5
    summary = calibration_summary(y, expit(raw + 2), p, scaling.log_odds(raw), flagged, scaling, 0.5)
    assert set(summary["scores"]) == {"training_rate", "uncalibrated", "calibrated"}
    assert summary["scores"]["calibrated"]["log_loss"] < summary["scores"]["uncalibrated"]["log_loss"]
    assert summary["expected_planets"]["all"] == pytest.approx(p.sum())
    assert summary["expected_planets"]["n_flagged"] == flagged.sum()
    # The outcomes were drawn from these very probabilities: the slope is close to 1.
    assert summary["test_calibration_slope"] == pytest.approx(1.0, abs=0.15)
