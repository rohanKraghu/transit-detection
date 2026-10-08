"""Folded views with depth and the centroid test, offline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from transitml import toi_views as tv
from transitml.benchmark import centroid_features, fetch_with_fallback, tpf_cache_path
from transitml.data.base import LightCurve
from transitml.data.synthetic_tpf import blend_scenario
from transitml.data.toi import TOI, BenchmarkTarget
from transitml.data.tpf import save_tpf
from transitml.kepler_dr25 import SCALAR_NAMES, TrainingSet
from transitml.tess_finetune import fit_gbm, labelled
from transitml.views import VIEW_NAMES


def tiny_set(n=40, seed=0, global_bins=2001, local_bins=201):
    """Two TOIs a star; planets (label 1) dip in the local view and are deeper."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 3 - 1 for i in range(n)])  # -1, 0, 1: open, FP, planet
    views = {k: rng.normal(0, 0.1, (n, local_bins)).astype(np.float32) for k in VIEW_NAMES}
    views["local"][labels == 1, 95:106] -= 1.0
    views["global"] = rng.normal(0, 0.1, (n, global_bins)).astype(np.float32)
    scalars = {k: np.full(n, 3.0) for k in SCALAR_NAMES}
    scalars["depth_scale"] = np.where(labels == 1, 2e-3, 1e-3) * rng.uniform(0.8, 1.2, n)
    scalars["scatter"] = rng.uniform(1e-4, 2e-4, n)
    return TrainingSet(
        tce_ids=np.array([f"{i // 2}.0{i % 2 + 1}" for i in range(n)]),
        kepids=np.arange(n, dtype=np.int64) // 2,
        classes=np.array(["PC"] * n),
        labels=labels.astype(np.int64),
        views=views,
        scalars=scalars,
    )


def test_each_input_set_adds_to_the_last():
    ts = tiny_set(12)
    pixels = np.arange(36, dtype=float).reshape(12, 3)
    views, depth, everything = (tv.model_inputs(ts, pixels, kind) for kind in tv.INPUT_SETS)
    assert views.shape == (12, 4 * 201 + 201 + 2)
    np.testing.assert_array_equal(depth[:, : views.shape[1]], views)
    np.testing.assert_array_equal(depth[:, -2:], np.column_stack([ts.scalars["depth_scale"], ts.scalars["scatter"]]))
    np.testing.assert_array_equal(everything[:, : depth.shape[1]], depth)
    np.testing.assert_array_equal(everything[:, -3:], pixels)
    with pytest.raises(ValueError, match="unknown input set"):
        tv.model_inputs(ts, pixels, "pixels")


def test_the_centroid_test_runs_at_each_tois_catalogue_ephemeris(tmp_path):
    tpf, truth = blend_scenario("blend", seed=11)
    save_tpf(tpf, tpf_cache_path(tmp_path, "TIC 5", 14))
    blend = TOI(tic=5, toi="5.01", disposition="FP", period=truth["period"],
                epoch_bjd=truth["epoch"] + 2457000.0, duration_hours=24 * truth["duration"], sectors=(14,))
    no_ephemeris = TOI(tic=5, toi="5.02", disposition="PC", sectors=(14,))
    elsewhere = TOI(tic=6, toi="6.01", disposition="CP", period=3.0, epoch_bjd=2458700.0,
                    duration_hours=2.0, sectors=(15,))
    targets = [
        BenchmarkTarget(tic=5, label=0, sector=14, reference=blend, tois=(blend, no_ephemeris)),
        BenchmarkTarget(tic=6, label=1, sector=15, reference=elsewhere, tois=(elsewhere,)),
    ]
    ts = tiny_set(3)
    ts.kepids[:] = [5, 5, 6]
    ts.tce_ids[:] = ["5.01", "5.02", "6.01"]

    tests = tv.catalogue_centroid_tests(ts, targets, tmp_path, n_jobs=1)
    assert tests[0]["status"] == "ok" and tests[0]["significant"]
    assert tests[1] is None  # no ephemeris
    assert tests[2] is None  # no pixel file for that star and sector
    pixels = centroid_features(tests).to_numpy()
    assert pixels[0, 0] >= 3.0 and np.isnan(pixels[1:]).all()
    assert tv.placed_fraction(pixels) == pytest.approx(1 / 3)


def test_a_training_star_is_taken_from_its_next_sector():
    def toi(tic, sectors):
        return TOI(tic=tic, toi=f"{tic}.01", disposition="CP", sectors=sectors)

    stars = {1: (3, 5, 9), 2: (3,), 3: (4, 30)}
    targets = [BenchmarkTarget(tic=t, label=1, sector=s[0], reference=toi(t, s), tois=(toi(t, s),))
               for t, s in stars.items()]
    available = {("TIC 1", 9), ("TIC 3", 4)}  # star 1 only in its third sector, star 2 nowhere
    asked = []

    def fetch(wanted):
        asked.append([(t.target_id, t.sector) for t in wanted])
        time = np.linspace(0.0, 27.0, 50)
        return [LightCurve(t.target_id, time, np.ones(50), np.full(50, 1e-3), label=t.label,
                           meta={"sector": t.sector})
                for t in wanted if (t.target_id, t.sector) in available]

    final, curves = fetch_with_fallback(targets, [3, 4, 5, 9, 30], fetch)
    assert [t.sector for t in final] == [9, 3, 4]
    assert [lc.target_id for lc in curves] == ["TIC 1", "TIC 3"]
    assert asked == [[("TIC 1", 3), ("TIC 2", 3), ("TIC 3", 4)], [("TIC 1", 5)], [("TIC 1", 9)]]


def test_a_model_is_the_mean_of_its_fits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + rng.normal(0.0, 0.7, 300) > 0).astype(int)
    X_test = rng.normal(size=(40, 4))
    one = tv.mean_of_fits(X, y, X_test, seed=3, n_fits=1)
    np.testing.assert_allclose(one, fit_gbm(X, y, 3).predict_proba(X_test)[:, 1])
    three = tv.mean_of_fits(X, y, X_test, seed=3, n_fits=3)
    each = [fit_gbm(X, y, s).predict_proba(X_test)[:, 1] for s in (3, 4, 5)]
    np.testing.assert_allclose(three, np.mean(each, axis=0))


def test_every_input_set_is_compared_and_a_run_can_be_read_back(tmp_path):
    rng = np.random.default_rng(4)
    train = labelled(tiny_set(240, seed=1))
    test = tiny_set(120, seed=2)
    stars = np.unique(test.kepids)
    planet = {int(s): int((test.labels[test.kepids == s] == 1).any()) for s in stars}
    tess_runs = {
        "features": [
            {"target_id": f"TIC {s}", "disposition": "CP" if planet[s] else "FP",
             "model_score": float(rng.random()), "period_recovered": s % 3 != 0}
            for s in planet
        ]
    }
    other = {"one_year": {kind: {s: float(rng.random()) for s in planet} for kind in tv.INPUT_SETS}}
    pixels = [rng.normal(size=(len(ts), 3)) for ts in (train, test)]
    for p in pixels:
        p[::4] = np.nan  # no pixel file
        p[1::4, :2] = np.nan  # a file, but the dip was not placed
    out = tv.compare_views(train, pixels[0], test, pixels[1], tess_runs, other_runs=other, n_fits=2)
    assert out["n_stars"] == len(planet)
    assert out["average_precision"]["views"] > 0.9  # the planted dip is easy
    assert out["average_precision"]["features"] < 0.8
    for pair in ("views_depth - views", "views_depth_pixels - views_depth",
                 "views - features", "views_depth_pixels - one_year_views_depth_pixels"):
        assert pair in out["differences"], pair
    assert set(out["operating_point"]) == {"probability", "planets_kept", "false_positives_rejected"}

    result = tv.ViewsResult(
        train_sectors="1-13", test_sectors="14-26", n_train_tois=len(train),
        n_train_stars=int(np.unique(train.kepids).size), n_train_planets=int(train.labels.sum()),
        n_train_from_later_sector=2, n_test_tois=len(test),
        centroid_placed={"train": 0.0, "test": 0.0}, n_fits=2, seed=0, **out,
    )
    report = tv.format_views_report(result)
    assert "views + depth and scatter + centroid test" in report
    assert "views (one year)" in report and "2 stars taken from a later sector" in report
    assert "views minus pipeline features" in report
    (tmp_path / "metrics.json").write_text(json.dumps(result.to_dict(), default=str))
    back = tv.read_run(tmp_path)
    assert back["views"] == {row["tic"]: row["views"] for row in out["stars"]}


def test_command_line_defaults():
    args = tv.parse_args([])
    assert args.train_sectors == "1-13" and args.test_sectors == "14-26" and args.fits == 5
    assert args.compare is None and args.tpfs is None
    args = tv.parse_args(["--compare", "a", "a.json", "--compare-run", "one_year", "results/toi_views"])
    assert args.compare == [["a", "a.json"]] and args.compare_run == [["one_year", "results/toi_views"]]


def test_without_pixel_files_the_pixel_model_is_left_out():
    train, test = labelled(tiny_set(120, seed=5)), tiny_set(60, seed=6)
    planet = {int(s): int((test.labels[test.kepids == s] == 1).any()) for s in np.unique(test.kepids)}
    stars = [{"target_id": f"TIC {s}", "disposition": "KP" if p else "FA", "model_score": 0.5,
              "period_recovered": True} for s, p in planet.items()]
    no_pixels = np.full((len(train), 3), np.nan)
    assert tv.usable_input_sets(no_pixels) == ("views", "views_depth")
    placed_once = no_pixels.copy()
    placed_once[0, 2] = 5.0  # a difference image somewhere, but no dip ever placed
    assert tv.usable_input_sets(placed_once) == ("views", "views_depth")
    out = tv.compare_views(train, no_pixels, test, np.full((len(test), 3), np.nan),
                           {"features": stars}, n_fits=1)
    assert set(out["average_precision"]) == {"views", "views_depth", "features"}
    assert out["operating_point"] == {}
