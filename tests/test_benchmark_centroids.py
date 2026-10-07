"""The TOI benchmark's centroid veto, offline, on synthetic pixel scenes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import transitml.benchmark as bench
from transitml.benchmark import (
    benchmark,
    build_benchmark_dataset,
    centroid_tests,
    format_benchmark_report,
    load_or_fetch_tpfs,
    plot_benchmark,
    tpf_cache_path,
)
from transitml.data.synthetic_tpf import blend_scenario
from transitml.data.toi import TOI, BenchmarkTarget
from transitml.data.tpf import load_tpf, save_tpf

N_EACH = 4


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    """Planets on their targets (CP) and binaries on a neighbour (FP), pixels and curves.

    Each light curve is its own stamp's aperture sum, so the search finds the
    event the pixels hold, as it would on a real star.
    """
    cache = tmp_path_factory.mktemp("toi_tpfs")
    curves, targets = [], []
    for i in range(2 * N_EACH):
        kind = "on_target" if i < N_EACH else "blend"
        tpf, truth = blend_scenario(kind, seed=200 + i)
        tic = 7000 + i
        tpf.target_id = f"TIC {tic}"
        label = int(kind == "on_target")
        toi = TOI(
            tic=tic,
            toi=f"{800 + i}.01",
            disposition="CP" if label else "FP",
            period=truth["period"],
            sectors=(14,),
        )
        targets.append(BenchmarkTarget(tic=tic, label=label, sector=14, reference=toi, tois=(toi,)))
        save_tpf(tpf, tpf_cache_path(cache, f"TIC {tic}", 14))
        curves.append(replace(tpf.to_light_curve(), label=label, meta={"kind": "real", "sector": 14}))
    return curves, targets, cache


@pytest.fixture(scope="module")
def dataset(scenes, tiny_model_config):
    curves, _, _ = scenes
    return build_benchmark_dataset(
        curves, preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls, n_jobs=2
    )


@pytest.fixture(scope="module")
def tests_and_result(scenes, dataset, tiny_trained):
    _, targets, cache = scenes
    _, trained = tiny_trained
    paths = load_or_fetch_tpfs(targets, cache, n_workers=1)
    # One star has no pixel file: it is never flagged.
    paths.pop(targets[-1].target_id)
    tests = centroid_tests(dataset, paths, n_jobs=2)
    comments = {t.reference.toi: "retired as TFOP FP/NEB" for t in targets[N_EACH:]}
    result = benchmark(
        dataset,
        targets,
        trained,
        sectors=[14],
        selection={"selected": len(targets)},
        top_k=4,
        n_bootstrap=200,
        centroids=tests,
        comments=comments,
    )
    return tests, result


def test_search_ephemeris_is_kept_beside_the_features(dataset, scenes):
    _, targets, _ = scenes
    meta = dataset.meta
    period = meta["search_period"].to_numpy(dtype=float)
    np.testing.assert_allclose(period, 10 ** dataset.features["log_period"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(period, [t.reference.period for t in targets], rtol=0.01)
    assert np.all(np.isfinite(meta["search_epoch"].to_numpy(dtype=float)))
    assert np.all(meta["search_duration"].to_numpy(dtype=float) > 0)


def test_blends_are_flagged_and_planets_are_not(tests_and_result):
    tests, _ = tests_and_result
    planets, blends = tests[:N_EACH], tests[N_EACH:]
    assert all(t is not None and t["status"] == "ok" and not t["significant"] for t in planets)
    assert all(t is not None and t["significant"] for t in blends[:-1])
    assert blends[-1] is None
    assert all(t["offset_distance_pixels"] > 0.5 for t in blends[:-1])


def test_the_veto_only_moves_flagged_stars_down(tests_and_result):
    tests, result = tests_and_result
    veto = result.centroid
    assert veto is not None
    flagged = np.array([bool(t and t["significant"]) for t in tests])
    scores, vetoed = result.model.scores, veto.model.scores
    np.testing.assert_array_equal(vetoed[~flagged], scores[~flagged])
    assert np.all(vetoed[flagged] < scores[~flagged].min())
    assert veto.model.average_precision >= result.model.average_precision
    assert veto.ap_gain == pytest.approx(
        veto.model.average_precision - result.model.average_precision
    )
    assert veto.planet_recall == result.planet_recall
    assert veto.false_positive_rejection >= result.false_positive_rejection


def test_flag_counts_by_class_disposition_and_reason(tests_and_result):
    _, result = tests_and_result
    flagged = result.centroid.flagged
    assert flagged["planets (CP/KP)"] == {"n": N_EACH, "with_pixels": N_EACH, "flagged": 0}
    assert flagged["false positives (FP/FA)"] == {
        "n": N_EACH,
        "with_pixels": N_EACH - 1,
        "flagged": N_EACH - 1,
    }
    assert flagged["FP, off target"]["flagged"] == N_EACH - 1
    assert flagged["FP, binary on target"]["n"] == 0
    assert result.centroid.n_with_pixels == 2 * N_EACH - 1
    floors = {row["min_offset_pixels"]: row for row in result.centroid.floors}
    assert floors[0.5]["planets_flagged"] == 0.0
    assert floors[0.5]["false_positives_flagged"] == 1.0


def test_report_json_stars_and_figure(tests_and_result, tmp_path):
    _, result = tests_and_result
    report = format_benchmark_report(result)
    assert "Centroid veto" in report and "model with the centroid veto" in report
    payload = json.loads(json.dumps(result.to_dict(), default=str))
    assert payload["centroid_veto"]["n_with_pixels"] == 2 * N_EACH - 1
    star = payload["stars"][N_EACH]
    assert star["centroid"]["significant"] is True
    assert star["false_positive_reason"] == "off target"
    assert payload["stars"][-1]["centroid"] is None
    assert "false_positive_reason" not in payload["stars"][0]
    assert Path(plot_benchmark(result, tmp_path / "toi.png")).stat().st_size > 10_000


def test_without_pixels_the_result_is_unchanged(scenes, dataset, tiny_trained, tests_and_result):
    _, targets, _ = scenes
    _, trained = tiny_trained
    plain = benchmark(
        dataset, targets, trained, sectors=[14], selection={}, top_k=4, n_bootstrap=200
    )
    _, with_veto = tests_and_result
    assert plain.centroid is None and "centroid_veto" not in plain.to_dict()
    assert plain.model.average_precision == with_veto.model.average_precision
    assert "Centroid veto" not in format_benchmark_report(plain)


def test_one_result_per_star_is_required(scenes, dataset, tiny_trained):
    _, targets, _ = scenes
    _, trained = tiny_trained
    with pytest.raises(ValueError, match="centroid results"):
        benchmark(dataset, targets, trained, sectors=[14], selection={}, centroids=[None])


def test_pixel_cache_fetches_only_what_it_has_not_tried(scenes, tmp_path, monkeypatch):
    _, targets, cache = scenes
    asked: list[tuple[str, int]] = []

    def fake_download(target_id, *, author, exposure_time, sector):
        asked.append((target_id, sector))
        assert (author, exposure_time) == ("TESS-SPOC", 1800)
        if target_id == targets[2].target_id:
            return []  # MAST has nothing for this star
        return [load_tpf(tpf_cache_path(cache, target_id, sector))]

    monkeypatch.setattr(bench, "download_tpfs", fake_download)
    fresh = tmp_path / "toi_tpfs"
    first = load_or_fetch_tpfs(targets[:4], fresh, n_workers=1)
    assert sorted(asked) == sorted((t.target_id, 14) for t in targets[:4])
    assert set(first) == {t.target_id for t in targets[:4]} - {targets[2].target_id}
    loaded = load_tpf(first[targets[0].target_id])
    np.testing.assert_array_equal(
        loaded.flux, load_tpf(tpf_cache_path(cache, targets[0].target_id, 14)).flux
    )

    again = load_or_fetch_tpfs(targets[:4], fresh, n_workers=1)
    assert again == first and len(asked) == 4, "a rerun must not touch the network"
    load_or_fetch_tpfs(targets[:5], fresh, n_workers=1)
    assert asked[-1] == (targets[4].target_id, 14) and len(asked) == 5
