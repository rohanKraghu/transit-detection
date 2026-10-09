"""One centroid verdict from a star's pixel files in several sectors."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import transitml.benchmark as bench
from transitml.benchmark import (
    SECTOR_CENTROID_FIELDS,
    build_benchmark_dataset,
    centroid_tests,
    load_or_fetch_sector_tpfs,
    tpf_cache_path,
)
from transitml.centroid import (
    CentroidConfig,
    _gaussian_sigma_from_log_p,
    centroid_test,
    combine_sector_tests,
    combined_pixel_files,
)
from transitml.data.base import stitch_light_curves
from transitml.data.synthetic_tpf import blend_scenario
from transitml.data.tpf import load_tpf, save_tpf

#: The neighbour's offset as each sector's stamp sees it: the spacecraft turns
#: between sectors, so the same star on the sky sits elsewhere in the stamp.
TURNED = [(1.6, 0.8), (-0.8, 1.6), (-1.6, -0.8), (0.8, -1.6)]


def sector_tests(kind: str, seed: int, **scene) -> list[dict]:
    tests = []
    for k, offset in enumerate(TURNED):
        tpf, truth = blend_scenario(kind, neighbour_offset=offset, seed=seed + k, **scene)
        result = centroid_test(tpf, truth["period"], truth["epoch"], truth["duration"])
        tests.append({name: getattr(result, name) for name in SECTOR_CENTROID_FIELDS})
    return tests


def test_one_sector_gives_back_its_own_numbers():
    (test,) = sector_tests("blend", seed=3)[:1]
    combined = combine_sector_tests([test])
    assert combined["offset_sigma"] == pytest.approx(test["offset_sigma"], rel=1e-9)
    assert combined["offset_distance_pixels"] == pytest.approx(test["offset_distance_pixels"])
    assert combined["difference_snr"] == pytest.approx(test["difference_snr"])
    assert combined["significant"] == test["significant"]
    assert (combined["n_sectors"], combined["n_sectors_placed"]) == (1, 1)


def test_a_blend_too_faint_for_any_one_sector_is_flagged_by_all_of_them():
    tests = sector_tests("blend", seed=0, binary_depth=0.009)
    assert not any(t["significant"] for t in tests)
    combined = combine_sector_tests(tests)
    assert combined["significant"]
    assert combined["offset_sigma"] > max(t["offset_sigma"] for t in tests)
    # The neighbour is 1.8 pixels away; edge pull makes the measured length a lower bound.
    assert 1.2 < combined["offset_distance_pixels"] < 2.5
    snrs = np.array([t["difference_snr"] for t in tests])
    assert combined["difference_snr"] == pytest.approx(np.sqrt(np.sum(snrs**2)))
    assert combined["n_transits"] == sum(t["n_transits"] for t in tests)


@pytest.mark.parametrize("seed", [0, 10, 20])
def test_planets_on_their_target_stay_unflagged(seed):
    combined = combine_sector_tests(sector_tests("on_target", seed=seed))
    assert combined["status"] == "ok" and not combined["significant"]
    assert combined["offset_distance_pixels"] < CentroidConfig().min_offset_pixels


@pytest.mark.parametrize("k", [2, 5])
def test_noise_in_every_sector_stays_noise(k):
    """Under no offset at all, the combined sigma keeps its nominal false-alarm rate."""
    rng = np.random.default_rng(k)
    log_p = np.log(rng.uniform(size=(10000, k)))
    sigmas = [_gaussian_sigma_from_log_p(lp) for lp in log_p.ravel()]
    rows = np.array(sigmas).reshape(-1, k)
    combined = [
        combine_sector_tests(
            [
                {"status": "ok", "offset_sigma": s, "offset_distance_pixels": 1.0,
                 "offset_error_pixels": (0.3, 0.3), "difference_snr": 10.0, "n_transits": 5}
                for s in row
            ]
        )["offset_sigma"]
        for row in rows
    ]
    assert np.mean(np.array(combined) >= 2.0) == pytest.approx(0.0455, abs=0.007)


def test_one_wild_sector_among_quiet_ones_does_not_decide():
    quiet = {"status": "ok", "offset_sigma": 0.5, "offset_distance_pixels": 0.3,
             "offset_error_pixels": (0.4, 0.4), "difference_snr": 10.0, "n_transits": 5}
    wild = {**quiet, "offset_sigma": 6.0, "offset_distance_pixels": 2.5,
            "offset_error_pixels": (0.3, 0.3)}
    assert combine_sector_tests([wild])["significant"]
    combined = combine_sector_tests([wild] + [quiet] * 8)
    assert combined["offset_sigma"] < 3.0 and not combined["significant"]


def test_the_better_measured_sector_weighs_more():
    tests = [
        {"status": "ok", "offset_sigma": 2.0, "offset_distance_pixels": 1.0,
         "offset_error_pixels": (0.1, 0.1), "difference_snr": 30.0, "n_transits": 9},
        {"status": "ok", "offset_sigma": 1.0, "offset_distance_pixels": 3.0,
         "offset_error_pixels": (1.0, 1.0), "difference_snr": 5.0, "n_transits": 3},
    ]
    assert combine_sector_tests(tests)["offset_distance_pixels"] == pytest.approx(
        (100 * 1.0 + 1 * 3.0) / 101
    )


def test_sectors_that_do_not_place_the_dip_are_left_out_of_the_offset():
    placed = {"status": "ok", "offset_sigma": 4.0, "offset_distance_pixels": 1.5,
              "offset_error_pixels": (0.2, 0.3), "difference_snr": 12.0, "n_transits": 6}
    weak = {"status": "weak_difference_image", "offset_sigma": 9.0,
            "offset_distance_pixels": 5.0, "offset_error_pixels": (2.0, 2.0),
            "difference_snr": 2.0, "n_transits": 4}
    empty = {"status": "no_in_transit_cadences", "offset_sigma": float("nan"),
             "offset_distance_pixels": float("nan"), "offset_error_pixels": None,
             "difference_snr": float("nan"), "n_transits": 0}
    # A centroid 15 pixels out, from a window of 3, is noise of both signs, not a dip.
    outside = {**placed, "offset_sigma": 6.0, "offset_distance_pixels": 15.0,
               "difference_snr": 7.0}
    combined = combine_sector_tests([empty, weak, outside, placed])
    assert combined["status"] == "ok" and combined["significant"]
    assert combined["offset_sigma"] == pytest.approx(4.0)
    assert combined["offset_distance_pixels"] == pytest.approx(1.5)
    assert combined["difference_snr"] == pytest.approx(np.sqrt(12.0**2 + 2.0**2 + 7.0**2))
    assert (combined["n_sectors"], combined["n_sectors_placed"]) == (4, 1)
    assert combine_sector_tests([outside, weak])["status"] == "centroid_outside_window"
    # Without a sector that placed it, the dip is not placed, and the reason says why.
    alone = combine_sector_tests([empty, weak])
    assert alone["status"] == "weak_difference_image" and not alone["significant"]
    assert np.isnan(alone["offset_sigma"]) and alone["n_sectors_placed"] == 0
    assert combine_sector_tests([empty])["status"] == "no_in_transit_cadences"


def test_combined_pixel_files_counts_only_combined_tests():
    assert combined_pixel_files([None, {"status": "ok"}]) is None
    assert combined_pixel_files([None, {"n_sectors": 3}, {"n_sectors": 1}]) == 4


@pytest.fixture(scope="module")
def three_sectors(tmp_path_factory):
    """A faint blend seen in three sectors, each stamp turned, its light curves joined."""
    cache = tmp_path_factory.mktemp("toi_tpfs")
    curves = []
    for k, offset in enumerate(TURNED[:3]):
        tpf, truth = blend_scenario(
            "blend", neighbour_offset=offset, binary_depth=0.009, seed=k
        )
        # Sectors a whole number of periods apart keep the eclipses on the ephemeris.
        tpf.time = tpf.time + 9 * truth["period"] * k
        tpf.target_id = "TIC 4242"
        save_tpf(tpf, tpf_cache_path(cache, "TIC 4242", 20 + k))
        curves.append(replace(tpf.to_light_curve(), label=0, meta={"kind": "real", "sector": 20 + k}))
    return cache, stitch_light_curves(curves)


def test_centroid_tests_combine_a_list_of_pixel_files(
    three_sectors, tiny_model_config, monkeypatch
):
    cache, joined = three_sectors
    monkeypatch.setattr(bench, "download_tpfs", lambda *_, **__: [])  # nothing for sector 23
    dataset = build_benchmark_dataset(
        [joined], preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls, n_jobs=1
    )
    paths = load_or_fetch_sector_tpfs({"TIC 4242": [20, 21, 22, 23]}, cache, n_workers=1)
    assert paths == {"TIC 4242": [tpf_cache_path(cache, "TIC 4242", s) for s in (20, 21, 22)]}

    (every,) = centroid_tests(dataset, paths, n_jobs=1)
    (first,) = centroid_tests(dataset, {"TIC 4242": paths["TIC 4242"][0]}, n_jobs=1)
    assert set(first) == set(bench.CENTROID_FIELDS)
    # On the search's own ephemeris one sector's centroid can fall outside its window.
    assert every["n_sectors"] == 3 and every["n_sectors_placed"] >= 2
    assert every["significant"] and not first["significant"]
    assert every["difference_snr"] > first["difference_snr"]


def test_sector_pixel_files_are_fetched_once_each(three_sectors, tmp_path, monkeypatch):
    cache, _ = three_sectors
    asked: list[tuple[str, int]] = []

    def fake_download(target_id, *, author, exposure_time, sector):
        asked.append((target_id, sector))
        path = tpf_cache_path(cache, target_id, sector)
        return [load_tpf(path)] if path.exists() else []

    monkeypatch.setattr(bench, "download_tpfs", fake_download)
    fresh = tmp_path / "toi_tpfs"
    first = load_or_fetch_sector_tpfs({"TIC 4242": [22, 20, 23]}, fresh, n_workers=1)
    assert sorted(asked) == [("TIC 4242", 20), ("TIC 4242", 22), ("TIC 4242", 23)]
    # In the order of the star's sectors, without the one MAST has no file for.
    assert first == {"TIC 4242": [tpf_cache_path(fresh, "TIC 4242", s) for s in (22, 20)]}
    again = load_or_fetch_sector_tpfs({"TIC 4242": [20, 22, 23]}, fresh, n_workers=1)
    assert len(asked) == 3, "a rerun must not touch the network"
    assert again["TIC 4242"] == [tpf_cache_path(fresh, "TIC 4242", s) for s in (20, 22)]
