"""``vet`` with target pixel files: the centroid test is reported beside the score."""

from __future__ import annotations

import json

import numpy as np
import pytest

from transitml.data.synthetic_tpf import blend_scenario
from transitml.data.tpf import save_tpf
from transitml.model import load_model, save_model
from transitml.vet import vet_light_curve


@pytest.fixture(scope="module")
def model_path(tiny_model_config, tiny_trained, tmp_path_factory):
    split, trained = tiny_trained
    return save_model(
        trained,
        split,
        tmp_path_factory.mktemp("model") / "model.joblib",
        preprocess=tiny_model_config.preprocess,
        bls=tiny_model_config.bls,
    )


@pytest.fixture(scope="module")
def scenes():
    """Light curve and pixels of one star, for an on-target transit and a blend."""
    out = {}
    for kind in ("on_target", "blend"):
        tpf, truth = blend_scenario(kind, seed=31)
        tpf.target_id = "TIC 42"
        out[kind] = (tpf.to_light_curve(), tpf, truth)
    return out


def write_csv(lc, path):
    np.savetxt(
        path,
        np.column_stack([lc.time, lc.flux, lc.flux_err]),
        delimiter=",",
        header="time,flux,flux_err",
        comments="",
    )
    return path


def test_centroid_test_is_beside_the_score_not_in_it(model_path, scenes):
    lc, tpf, truth = scenes["blend"]
    model = load_model(model_path)
    plain, _ = vet_light_curve(lc, model)
    with_pixels, _ = vet_light_curve(lc, model, tpfs=[tpf])

    assert plain.primary["period"] == pytest.approx(truth["period"], rel=0.01)
    assert with_pixels.score == plain.score
    assert with_pixels.features == plain.features
    assert plain.centroids is None and plain.centroid_offset is None
    assert with_pixels.centroid_offset is True
    assert "centroid" not in plain.to_dict()
    section = with_pixels.to_dict()["centroid"]
    assert section["offset_flag"] is True and "not use" in section["note"]
    assert section["tests"][0]["significant"] is True


@pytest.mark.parametrize("kind, flagged", [("on_target", False), ("blend", True)])
def test_cli_with_a_tpf_file(model_path, scenes, tmp_path, capsys, kind, flagged):
    from transitml import vet

    lc, tpf, _ = scenes[kind]
    csv_path = write_csv(lc, tmp_path / "star.csv")
    tpf_path = save_tpf(tpf, tmp_path / "star_tpf.npz")
    out = tmp_path / "reports"
    argv = [str(csv_path), "--tpf", str(tpf_path), "--model", str(model_path)]
    assert vet.main(argv + ["--out-dir", str(out)]) == 0

    payload = json.loads((out / "vet_star.json").read_text())
    centroid = payload["centroid"]
    assert centroid["offset_flag"] is flagged
    assert len(centroid["tests"]) == 1
    test = centroid["tests"][0]
    assert test["status"] == "ok" and test["reference"] == "target_position"
    assert test["pixel_scale_arcsec"] == 21.0
    printed = capsys.readouterr().out
    assert ("FLAG: significant centroid offset" in printed) is flagged
    assert "centroid (" in printed
    assert (out / "vet_star.png").stat().st_size > 50_000


def test_report_png_gains_a_centroid_row(model_path, scenes, tmp_path):
    from matplotlib.image import imread

    from transitml.vet import write_report

    lc, tpf, _ = scenes["blend"]
    model = load_model(model_path)
    plain, flat = vet_light_curve(lc, model)
    with_pixels, _ = vet_light_curve(lc, model, tpfs=[tpf])
    png_plain, _ = write_report(lc, flat, plain, tmp_path, stem="plain")
    png_pixels, _ = write_report(lc, flat, with_pixels, tmp_path, stem="pixels")
    assert imread(png_pixels).shape[0] > 1.15 * imread(png_plain).shape[0]


def test_tpf_from_another_span_does_not_crash(model_path, scenes, tmp_path):
    lc, tpf, _ = scenes["blend"]
    early = type(
        tpf
    )(  # ten cadences, all before the first transit
        tpf.target_id,
        tpf.time[:10],
        tpf.flux[:10],
        tpf.aperture,
        target_position=(5.0, 5.0),
    )
    result, flat = vet_light_curve(lc, load_model(model_path), tpfs=[early])
    assert result.centroids[0].status != "ok"
    assert result.centroid_offset is False
    from transitml.vet import write_report

    png, js = write_report(lc, flat, result, tmp_path)
    assert json.loads(js.read_text())["centroid"]["tests"][0]["significant"] is False
    assert png.exists()


def test_cli_downloads_pixels_for_a_tic(
    model_path, scenes, tmp_path, monkeypatch, capsys
):
    """Offline: MAST light curves and target pixel files are both stubbed."""
    from transitml import vet
    from transitml.data.base import LightCurve

    lc, tpf, _ = scenes["blend"]
    seen = {}

    class FakeMAST:
        def __init__(self, targets, **kwargs):
            pass

        def __iter__(self):
            yield LightCurve(
                "TIC 42", lc.time, lc.flux, lc.flux_err, meta={"sector": 14}
            )

    def fake_download(target, **kwargs):
        seen["target"], seen["kwargs"] = target, kwargs
        return [tpf]

    monkeypatch.setattr(vet, "MASTLightCurveSource", FakeMAST)
    monkeypatch.setattr(vet, "download_tpfs", fake_download)
    vet.main(
        ["TIC 42", "--sector", "14", "--centroids"]
        + ["--model", str(model_path), "--out-dir", str(tmp_path)]
    )
    assert seen["target"] == "TIC 42"
    assert seen["kwargs"] == {
        "author": "TESS-SPOC",
        "exposure_time": 1800,
        "sector": 14,
    }
    payload = json.loads((tmp_path / "vet_TIC_42.json").read_text())
    assert payload["centroid"]["offset_flag"] is True


def test_cli_reports_a_missing_tpf(model_path, scenes, tmp_path, monkeypatch, capsys):
    from transitml import vet
    from transitml.data.base import LightCurve

    lc = scenes["on_target"][0]

    class FakeMAST:
        def __init__(self, targets, **kwargs):
            pass

        def __iter__(self):
            yield LightCurve("TIC 42", lc.time, lc.flux, lc.flux_err)

    monkeypatch.setattr(vet, "MASTLightCurveSource", FakeMAST)
    monkeypatch.setattr(vet, "download_tpfs", lambda target, **kwargs: [])
    vet.main(
        [
            "TIC 42",
            "--centroids",
            "--model",
            str(model_path),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert "no target pixel file found" in capsys.readouterr().out
    centroid = json.loads((tmp_path / "vet_TIC_42.json").read_text())["centroid"]
    assert centroid["tests"] == [] and centroid["offset_flag"] is None
    assert centroid["note"].startswith("no target pixel file was available")


def test_cli_refuses_centroids_for_a_local_file(model_path, scenes, tmp_path):
    from transitml import vet

    csv_path = write_csv(scenes["on_target"][0], tmp_path / "star.csv")
    with pytest.raises(SystemExit, match="--tpf"):
        vet.main([str(csv_path), "--centroids", "--model", str(model_path)])
