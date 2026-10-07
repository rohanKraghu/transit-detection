"""The trained classifier is saved with its threshold and feature order, and reloads exactly."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from transitml.features import FEATURE_NAMES
from transitml.model import feature_medians, load_model, save_model


def test_saved_model_round_trips(tiny_model_config, tiny_trained, tmp_path):
    split, trained = tiny_trained
    path = save_model(
        trained, split, tmp_path / "model.joblib",
        preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls,
    )
    loaded = load_model(path)

    assert loaded.threshold == trained.threshold
    assert loaded.threshold_rule == trained.threshold_rule
    assert loaded.feature_names == FEATURE_NAMES
    assert loaded.bls == tiny_model_config.bls
    assert loaded.preprocess == tiny_model_config.preprocess
    np.testing.assert_array_equal(loaded.score(split.X_test), trained.score(split.X_test))
    np.testing.assert_array_equal(loaded.train_medians, feature_medians(split.X_train))
    assert loaded.provenance["n_train"] == len(split.y_train)
    assert loaded.calibration == trained.calibration
    np.testing.assert_array_equal(loaded.probability(split.X_test), trained.probability(split.X_test))
    np.testing.assert_array_equal(loaded.explain(split.X_test), trained.explain(split.X_test))


def test_a_model_from_an_older_format_is_refused(tiny_model_config, tiny_trained, tmp_path):
    split, trained = tiny_trained
    path = save_model(
        trained, split, tmp_path / "model.joblib",
        preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls,
    )
    payload = joblib.load(path)
    payload["format_version"] = 1
    del payload["calibration"]
    joblib.dump(payload, path)
    with pytest.raises(ValueError, match="model format 1.*retrain"):
        load_model(path)


def test_a_model_with_a_different_feature_set_is_refused(tiny_model_config, tiny_trained, tmp_path):
    split, trained = tiny_trained
    path = save_model(
        trained, split, tmp_path / "model.joblib",
        preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls,
    )
    payload = joblib.load(path)
    payload["feature_names"] = payload["feature_names"][:-1]
    joblib.dump(payload, path)
    with pytest.raises(ValueError, match="different feature set"):
        load_model(path)


def test_feature_medians_ignore_nan_and_keep_empty_columns_nan():
    X = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    medians = feature_medians(X)
    assert medians[0] == 2.0 and np.isnan(medians[1])


def test_run_pipeline_writes_a_loadable_model(tmp_path):
    import run_pipeline

    status = run_pipeline.main(
        ["--n-curves", "250", "--n-jobs", "2", "--no-figures", "--results-dir", str(tmp_path)]
    )
    assert status == 0
    model = load_model(tmp_path / "model.joblib")
    assert 0.0 < model.threshold < 1.0
    assert (tmp_path / "report.txt").exists()
