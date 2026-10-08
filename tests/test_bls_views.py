"""Views folded at the pipeline's own BLS ephemeris, offline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from transitml import bls_views as bv
from transitml.benchmark import (
    _centroid_one,
    build_benchmark_dataset,
    centroid_features,
    tpf_cache_path,
)
from transitml.data.base import LightCurve
from transitml.data.synthetic_tpf import blend_scenario
from transitml.data.toi import TOI, BenchmarkTarget
from transitml.data.tpf import save_tpf
from transitml.kepler_dr25 import SCALAR_NAMES, TrainingSet
from transitml.views import VIEW_NAMES, Ephemeris, ViewConfig

from .conftest import clean_transit_curve


def as_host(lc: LightCurve, tic: int, label: int) -> LightCurve:
    return LightCurve(f"TIC {tic}", lc.time, lc.flux, lc.flux_err, label, {"sector": 14})


def target(tic: int, label: int) -> BenchmarkTarget:
    toi = TOI(tic=tic, toi=f"{tic}.01", disposition="CP" if label else "FP", sectors=(14,))
    return BenchmarkTarget(tic=tic, label=label, sector=14, reference=toi, tois=(toi,))


def tiny_hosts(n=60, seed=0, n_features=5):
    """One row per star; planets (label 1) dip in the local view and lean on feature 0."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2
    views = {k: rng.normal(0, 0.1, (n, 201)).astype(np.float32) for k in VIEW_NAMES}
    views["local"][labels == 1, 95:106] -= 1.0
    views["global"] = rng.normal(0, 0.1, (n, 2001)).astype(np.float32)
    scalars = {k: np.full(n, 3.0) for k in SCALAR_NAMES}
    scalars["scatter"] = rng.uniform(1e-4, 2e-4, n)
    ts = TrainingSet(
        tce_ids=np.array([f"TIC {i}" for i in range(n)]),
        kepids=np.arange(n, dtype=np.int64),
        classes=np.array(["CP" if y else "FP" for y in labels]),
        labels=labels.astype(np.int64),
        views=views,
        scalars=scalars,
    )
    features = rng.normal(size=(n, n_features))
    features[:, 0] += 0.5 * labels
    pixels = rng.normal(size=(n, 3))
    pixels[::4] = np.nan  # no pixel file
    pixels[1::4, :2] = np.nan  # a file, but the dip was not placed
    return bv.Hosts(views=ts, features=features, pixels=pixels)


def test_views_are_folded_at_the_given_ephemeris():
    lc, truth = clean_transit_curve(period=3.0, depth=4e-3, duration=0.12, sigma=2e-4, seed=1)
    eph = Ephemeris(truth["period"], truth["epoch"], truth["duration"])
    views, scalars = bv.host_views(lc, eph, ViewConfig())
    assert set(views) == set(VIEW_NAMES) and set(scalars) == set(SCALAR_NAMES)
    assert scalars["period_days"] == 3.0 and scalars["duration_hours"] == pytest.approx(2.88)
    assert scalars["depth_scale"] == pytest.approx(4e-3, rel=0.15)
    assert abs(int(np.argmin(views["local"])) - 100) <= 10  # the dip is centred
    off = Ephemeris(truth["period"], truth["epoch"] + 1.5, truth["duration"])
    assert bv.host_views(lc, off, ViewConfig())[1]["depth_scale"] < 2e-3  # nothing there


def test_hosts_are_searched_featurised_and_tested_as_the_pipeline_does(config, tmp_path):
    planet, _ = clean_transit_curve(period=3.0, depth=4e-3, sigma=2e-4, seed=2)
    quiet, _ = clean_transit_curve(period=3.0, depth=0.0, sigma=2e-4, seed=3)
    curves = [as_host(planet, 5, 1), as_host(quiet, 6, 0)]
    targets = [target(5, 1), target(6, 0)]
    tpf, _ = blend_scenario("blend", seed=11)
    path = tpf_cache_path(tmp_path, "TIC 5", 14)
    save_tpf(tpf, path)  # star 6 has no pixel file

    hosts = bv.search_hosts(curves, targets, tmp_path, config=config, n_jobs=1)
    pipeline = build_benchmark_dataset(curves, preprocess=config.preprocess, bls=config.bls, n_jobs=1)
    np.testing.assert_array_equal(hosts.features, pipeline.features.to_numpy())
    assert hosts.views.kepids.tolist() == [5, 6] and hosts.views.labels.tolist() == [1, 0]
    ephemeris = pipeline.meta[["search_period", "search_epoch", "search_duration"]].to_numpy(float)
    np.testing.assert_array_equal(hosts.views.scalars["period_days"], ephemeris[:, 0])
    assert hosts.views.scalars["period_days"][0] == pytest.approx(3.0, rel=0.01)
    expected = centroid_features([_centroid_one(path, *ephemeris[0], None), None]).to_numpy()
    np.testing.assert_array_equal(hosts.pixels, expected)
    assert np.isnan(hosts.pixels[1]).all() and hosts.n_without_views == 0


def test_each_input_set_is_built_from_the_hosts():
    hosts = tiny_hosts(12)
    views = bv.model_inputs(hosts, "views")
    assert views.shape == (12, 4 * 201 + 201 + 2)
    np.testing.assert_array_equal(bv.model_inputs(hosts, "features"), hosts.features)
    np.testing.assert_array_equal(
        bv.model_inputs(hosts, "features_pixels"), np.hstack([hosts.features, hosts.pixels])
    )
    full = bv.model_inputs(hosts, "views_depth_pixels")
    np.testing.assert_array_equal(bv.model_inputs(hosts, "everything"), np.hstack([full, hosts.features]))
    np.testing.assert_array_equal(full[:, : views.shape[1]], views)
    with pytest.raises(ValueError, match="unknown input set"):
        bv.model_inputs(hosts, "pixels")


def test_every_set_is_compared_with_the_pipeline_and_a_run_can_be_read_back(tmp_path):
    rng = np.random.default_rng(4)
    train, test = tiny_hosts(240, seed=1), tiny_hosts(120, seed=2)
    rows = [
        {"target_id": f"TIC {tic}", "disposition": "CP" if y else "FP",
         "model_score": float(rng.random()), "period_recovered": tic % 3 != 0}
        for tic, y in zip(test.views.kepids.tolist(), test.views.labels.tolist())
    ]
    pipeline = {"pipeline": rows, "pipeline_pixels": rows}
    other = {"one_year": {k: {int(t): float(rng.random()) for t in test.views.kepids} for k in bv.INPUT_SETS}}
    out = bv.compare_hosts(train, test, pipeline, other_runs=other, n_fits=2)
    assert out["n_stars"] == 120
    assert out["average_precision"]["views"] > 0.9  # the planted dip is easy
    for pair in ("features - pipeline", "features_pixels - pipeline_pixels", "views - features",
                 "views_depth_pixels - features_pixels", "everything - features_pixels",
                 "views_depth - views", "everything - views_depth_pixels", "views - one_year_views"):
        assert pair in out["differences"], pair

    result = bv.HostViewsResult(
        train_sectors="1-13", test_sectors="14-26", n_train_hosts=len(train),
        n_train_planets=int(train.views.labels.sum()), n_train_from_later_sector=3,
        n_without_views={"train": 0, "test": 1}, centroid_placed={"train": 0.5, "test": 0.5},
        n_fits=2, seed=0, **out,
    )
    report = bv.format_host_views_report(result)
    assert "the pipeline's model + centroid test" in report and "views (one year)" in report
    assert "3 taken from a later sector" in report
    (tmp_path / "metrics.json").write_text(json.dumps(result.to_dict(), default=str))
    back = bv.read_run(tmp_path, bv.INPUT_SETS)
    assert back["everything"] == {row["tic"]: row["everything"] for row in out["stars"]}


def test_without_pixel_files_the_pixel_models_are_left_out():
    train, test = tiny_hosts(80, seed=5), tiny_hosts(40, seed=6)
    train.pixels[:] = np.nan
    test.pixels[:] = np.nan
    assert bv.usable_input_sets(train.pixels) == ("features", "views", "views_depth")
    rows = [{"target_id": f"TIC {t}", "disposition": "KP" if y else "FA", "model_score": 0.5,
             "period_recovered": True}
            for t, y in zip(test.views.kepids.tolist(), test.views.labels.tolist())]
    out = bv.compare_hosts(train, test, {"pipeline": rows}, n_fits=1)
    assert set(out["average_precision"]) == {"features", "views", "views_depth", "pipeline"}
    assert out["operating_point"] == {}


def test_command_line_defaults():
    args = bv.parse_args([])
    assert args.train_sectors == "1-13" and args.test_sectors == "14-26" and args.fits == 5
    assert args.compare is None and args.tpfs is None and args.stars is None
    assert str(args.results_dir) == "results/bls_views"
