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


def test_report_writes_a_png_and_matching_json(model_path, planet_curve, tmp_path):
    from transitml.vet import write_report

    lc = planet_curve[0]
    result, flat = vet_light_curve(lc, load_model(model_path))
    png, js = write_report(lc, flat, result, tmp_path)

    assert png.name == "vet_TEST_0001.png" and js.name == "vet_TEST_0001.json"
    assert png.stat().st_size > 50_000
    assert json.loads(js.read_text())["score"] == pytest.approx(result.score)


def write_planet_csv(lc, path):
    np.savetxt(
        path, np.column_stack([lc.time, lc.flux * 52_000.0, lc.flux_err * 52_000.0]),
        delimiter=",", header="time,flux,flux_err", comments="",
    )
    return path


def test_cli_end_to_end_from_a_csv(model_path, planet_curve, tmp_path, capsys):
    from transitml import vet

    csv_path = write_planet_csv(planet_curve[0], tmp_path / "planet.csv")
    out = tmp_path / "reports"
    assert vet.main([str(csv_path), "--model", str(model_path), "--out-dir", str(out)]) == 0

    png, js = out / "vet_planet.png", out / "vet_planet.json"
    assert png.exists() and png.stat().st_size > 50_000
    payload = json.loads(js.read_text())
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["candidates"][0]["period"] == pytest.approx(3.0, rel=0.01)
    assert "signal 1: P = 3.0" in capsys.readouterr().out


def test_cli_reads_one_star_from_an_npz_cache(model_path, planet_curve, tmp_path):
    from transitml import vet
    from transitml.data.base import LightCurve
    from transitml.data.injection import save_curves

    lc = planet_curve[0]
    other = LightCurve("TIC 9", lc.time, np.ones_like(lc.flux), lc.flux_err)
    planet = LightCurve("TIC 8", lc.time, lc.flux, lc.flux_err)
    save_curves([other, planet], tmp_path / "cache.npz")

    with pytest.raises(SystemExit, match="2 light curves"):
        vet.main([str(tmp_path / "cache.npz"), "--model", str(model_path)])
    vet.main([
        str(tmp_path / "cache.npz"), "--target-id", "TIC 8",
        "--model", str(model_path), "--out-dir", str(tmp_path),
    ])
    assert json.loads((tmp_path / "vet_TIC_8.json").read_text())["target_id"] == "TIC 8"


def test_cli_downloads_a_tic_through_the_mast_source(model_path, planet_curve, tmp_path, monkeypatch):
    """Offline: the MAST source is replaced; only the wiring is tested."""
    from transitml import vet
    from transitml.data.base import LightCurve

    calls = {}

    class FakeMAST:
        def __init__(self, targets, **kwargs):
            calls["targets"], calls["kwargs"] = targets, kwargs

        def __iter__(self):
            lc = planet_curve[0]
            target = calls["targets"][0][0]
            yield LightCurve(target, lc.time, lc.flux, lc.flux_err, meta={"sector": 14})

    monkeypatch.setattr(vet, "MASTLightCurveSource", FakeMAST)
    vet.main(["tic307210830", "--sector", "14", "--model", str(model_path), "--out-dir", str(tmp_path)])

    assert calls["targets"] == [("TIC 307210830", None)]
    assert calls["kwargs"]["sector"] == 14 and calls["kwargs"]["author"] == "TESS-SPOC"
    payload = json.loads((tmp_path / "vet_TIC_307210830.json").read_text())
    assert payload["target_id"] == "TIC 307210830" and payload["candidates"]


def test_cli_refuses_a_target_that_is_neither_file_nor_tic(model_path):
    from transitml import vet

    with pytest.raises(SystemExit, match="neither an existing file nor a TIC"):
        vet.main(["no-such-file.csv", "--model", str(model_path)])


def test_tic_id_normalisation():
    from transitml.vet import tic_id

    assert tic_id("TIC 123") == tic_id("tic123") == tic_id(" 123 ") == "TIC 123"
    assert tic_id("star.csv") is None
