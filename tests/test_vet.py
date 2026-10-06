"""``vet``: one light curve in, a score, its reasons and every candidate signal out."""

from __future__ import annotations

import json

import numpy as np
import pytest

from transitml.features import FEATURE_NAMES
from transitml.model import load_model, save_model
from transitml.vet import N_REASONS, feature_contributions, vet_light_curve, write_json

from .conftest import clean_transit_curve


@pytest.fixture(scope="module")
def model_path(tiny_model_config, tiny_trained, tmp_path_factory):
    split, trained = tiny_trained
    return save_model(
        trained, split, tmp_path_factory.mktemp("model") / "model.joblib",
        preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls,
    )


@pytest.fixture(scope="module")
def planet_curve():
    lc, truth = clean_transit_curve(period=3.0, depth=4e-3, sigma=4e-4, seed=7)
    return lc, truth


def test_vet_scores_the_primary_and_lists_candidates(model_path, planet_curve):
    lc, truth = planet_curve
    model = load_model(model_path)
    result, flat = vet_light_curve(lc, model)

    assert 0.0 <= result.score <= 1.0
    assert result.threshold == model.threshold
    assert result.primary["period"] == pytest.approx(truth["period"], rel=0.01)
    assert len(result.candidates) >= 1
    assert result.candidates[0].period == result.primary["period"]
    assert list(result.features) == list(FEATURE_NAMES)
    assert flat.time.size == result.n_cadences


def test_contributions_vanish_at_the_training_median(model_path):
    model = load_model(model_path)
    x = np.where(np.isfinite(model.train_medians), model.train_medians, 0.0)
    rows = feature_contributions(model, x)
    assert rows and all(row["delta_score"] == 0.0 for row in rows)


def test_contributions_are_sorted_by_size_and_measure_a_real_change(model_path, planet_curve):
    model = load_model(model_path)
    result, _ = vet_light_curve(planet_curve[0], model)
    deltas = [abs(row["delta_score"]) for row in result.contributions]
    assert deltas == sorted(deltas, reverse=True)
    top = result.contributions[0]
    x = np.array([result.features[n] for n in FEATURE_NAMES])
    x[FEATURE_NAMES.index(top["feature"])] = top["training_median"]
    assert result.score - model.score(x)[0] == pytest.approx(top["delta_score"])


def test_json_is_strict_and_carries_score_and_candidates(model_path, planet_curve, tmp_path):
    result, _ = vet_light_curve(planet_curve[0], load_model(model_path))
    path = write_json(result, tmp_path / "vet.json")
    payload = json.loads(path.read_text())  # strict: NaN would have raised on write

    assert payload["score"] == pytest.approx(result.score)
    assert payload["threshold"] == pytest.approx(result.threshold)
    assert isinstance(payload["candidates"], list) and payload["candidates"]
    assert payload["candidates"][0]["period"] == pytest.approx(3.0, rel=0.01)
    assert len(payload["top_reasons"]) == min(N_REASONS, len(result.contributions))
    assert "approximate" in payload["contribution_method"]
    assert payload["verdict"].startswith(
        "planet candidate" if result.above_threshold else "not a candidate"
    )
