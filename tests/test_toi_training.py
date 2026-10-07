"""Training on real TOI labels, offline, on synthetic stand-ins for the hosts of two sectors."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from transitml.benchmark import (
    CENTROID_FEATURE_NAMES,
    benchmark,
    build_benchmark_dataset,
    centroid_features,
    centroid_tests,
    format_benchmark_report,
    tpf_cache_path,
    with_centroid_features,
)
from transitml.calibration import PlattScaling, fit_platt
from transitml.config import default_config
from transitml.data.base import LightCurve
from transitml.data.loader import Dataset
from transitml.data.synthetic_tpf import blend_scenario
from transitml.data.toi import TOI, BenchmarkTarget
from transitml.data.tpf import save_tpf
from transitml.features import FEATURE_NAMES
from transitml.model import (
    TrainedModel,
    build_model,
    load_model,
    probability_threshold,
    save_model,
    train,
)
from transitml.toi_training import (
    format_training_report,
    summarise_training,
    train_on_hosts,
)

from .conftest import make_source

PIXEL_NAMES = FEATURE_NAMES + CENTROID_FEATURE_NAMES


@pytest.fixture(scope="module")
def small_config():
    base = default_config()
    return replace(base, bls=replace(base.bls, n_periods=400))


def test_probability_threshold_maps_back_to_the_probability():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    raw = rng.normal(loc=1.5 * y - 0.75, scale=1.0)
    calibration = fit_platt(raw, y)
    threshold, rule, precision, recall = probability_threshold(y, raw, calibration, 0.5)

    keep = expit(raw) >= threshold
    assert precision == pytest.approx(y[keep].mean())
    assert recall == pytest.approx(keep[y == 1].mean())
    assert calibration.probability(np.array([logit(threshold)]))[0] == pytest.approx(0.5)
    assert "calibrated P(planet) >= 0.50" in rule
    strict, *_ = probability_threshold(y, raw, calibration, 0.8)
    assert strict > threshold
    with pytest.raises(ValueError, match="probability"):
        probability_threshold(y, raw, calibration, 1.0)


def test_the_probability_rule_moves_only_the_threshold(tiny_trained):
    split, by_precision = tiny_trained
    by_probability = train(
        split, n_folds=3, seed=default_config().seed, target_precision=0.5,
        operating_probability=0.5,
    )
    assert by_probability.threshold_probability == pytest.approx(0.5)
    assert by_probability.threshold_rule.startswith("calibrated P(planet) >= 0.50")
    assert by_probability.calibration == by_precision.calibration
    np.testing.assert_array_equal(
        by_probability.score(split.X_test), by_precision.score(split.X_test)
    )
    with pytest.raises(ValueError, match="feature columns"):
        train(split, n_folds=3, seed=0, target_precision=0.5, feature_names=FEATURE_NAMES[:3])


def _toy_hosts(n: int, planet_rate: float, seed: int) -> Dataset:
    """Light-curve features where depth carries some of the label, as among real TOIs."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < planet_rate).astype(int)
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    features["log_depth"] += 1.2 * y
    meta = pd.DataFrame({"target_id": [f"TIC {i}" for i in range(n)]})
    return Dataset(features=features, labels=y, meta=meta)


def test_the_survey_rule_keeps_every_star_at_the_catalogue_mix():
    """At 60% planets a floor of 0.5 on precision is met by keeping everyone."""
    dataset = _toy_hosts(240, 0.6, seed=5)
    trained, split = train_on_hosts(
        dataset, feature_names=FEATURE_NAMES, n_folds=5, seed=0,
        target_precision=0.5, precision_lcb_z=1.0,
    )
    assert len(split.y_test) == 0 and len(split.y_train) == len(dataset)
    summary = summarise_training(
        dataset, split, trained, sectors=list(range(1, 14)), selection={"selected": 250},
        n_without_curve=10, n_folds=5, target_precision=0.5, precision_lcb_z=1.0, seed=0,
        n_bootstrap=100,
    )
    assert summary.precision_rule["cv_planet_recall"] == 1.0
    assert summary.precision_rule["cv_false_positive_rejection"] == 0.0
    assert 0.0 < summary.false_positive_rejection < 1.0
    assert summary.threshold_probability == pytest.approx(0.5)
    assert summary.model.average_precision > summary.positive_rate + 0.05

    report = format_training_report(summary)
    assert "TOI HOSTS OF SECTORS 1 TO 13" in report
    assert "the survey rule would keep 1.00 of the planets and reject 0.00" in report
    assert "Scored, unchanged" not in report
    payload = json.loads(json.dumps(summary.to_dict(), default=str))
    assert payload["n_stars"] == 240 and "n_with_pixels" not in payload


def test_centroid_features_blank_what_the_test_could_not_place():
    tests = [
        None,
        {"status": "weak_difference_image", "offset_sigma": 9.0,
         "offset_distance_pixels": 2.0, "difference_snr": 1.5},
        {"status": "ok", "offset_sigma": 4.0, "offset_distance_pixels": 1.2,
         "difference_snr": 30.0},
    ]
    frame = centroid_features(tests)
    assert tuple(frame.columns) == CENTROID_FEATURE_NAMES
    assert frame.iloc[0].isna().all()
    assert frame.iloc[1].isna().tolist() == [True, True, False]
    assert frame.iloc[1]["centroid_difference_snr"] == 1.5
    assert frame.iloc[2].tolist() == [4.0, 1.2, 30.0]


@pytest.fixture(scope="module")
def scenes(tmp_path_factory, tiny_model_config):
    """Three planets on their targets (CP) and three binaries on a neighbour (FP)."""
    cache = tmp_path_factory.mktemp("toi_tpfs")
    curves, targets = [], []
    for i in range(6):
        kind = "on_target" if i < 3 else "blend"
        tpf, truth = blend_scenario(kind, seed=300 + i)
        tic = 9100 + i
        tpf.target_id = f"TIC {tic}"
        label = int(kind == "on_target")
        toi = TOI(tic=tic, toi=f"{700 + i}.01", disposition="CP" if label else "FP",
                  period=truth["period"], sectors=(14,))
        targets.append(BenchmarkTarget(tic=tic, label=label, sector=14, reference=toi, tois=(toi,)))
        save_tpf(tpf, tpf_cache_path(cache, f"TIC {tic}", 14))
        curves.append(replace(tpf.to_light_curve(), label=label, meta={"kind": "real", "sector": 14}))
    dataset = build_benchmark_dataset(
        curves, preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls, n_jobs=2
    )
    paths = {t.target_id: tpf_cache_path(cache, t.target_id, 14) for t in targets}
    tests = centroid_tests(dataset, paths, n_jobs=2)
    return dataset, targets, tests


def _pixel_model() -> TrainedModel:
    """A classifier that has learned one thing: a dip off the target is not a planet."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(PIXEL_NAMES)))
    column = PIXEL_NAMES.index("centroid_offset_pixels")
    X[:, column] = rng.uniform(0.0, 3.0, 300)
    y = (X[:, column] < 0.5).astype(int)
    return TrainedModel(
        estimator=build_model(0).fit(X, y),
        oof_scores=np.zeros(0),
        threshold=0.5,
        threshold_rule="test",
        achieved_cv_precision=float("nan"),
        achieved_cv_recall=float("nan"),
        calibration=PlattScaling(1.0, 0.0, 0.5),
        feature_names=PIXEL_NAMES,
    )


def test_a_model_with_pixel_features_is_scored_on_them(scenes, tiny_trained):
    dataset, targets, tests = scenes
    with_pixels = with_centroid_features(dataset, tests)
    np.testing.assert_array_equal(with_pixels.X, dataset.X)
    assert with_pixels.inputs(PIXEL_NAMES).shape == (6, len(PIXEL_NAMES))

    trained = _pixel_model()
    result = benchmark(
        with_pixels, targets, trained, sectors=[14], selection={}, top_k=2,
        n_bootstrap=50, centroids=tests, importance_repeats=3,
    )
    assert result.model.average_precision == 1.0
    assert result.feature_names == list(PIXEL_NAMES)
    assert result.feature_importance[0]["feature"] == "centroid_offset_pixels"
    assert {row["feature"] for row in result.feature_importance} == set(PIXEL_NAMES)
    payload = result.to_dict()
    assert payload["model_features"] == list(PIXEL_NAMES)
    report = format_benchmark_report(result)
    assert "model inputs: the 23 light-curve features and centroid_offset_sigma" in report
    assert "What the model leans on here" in report

    with pytest.raises(ValueError, match="has no"):
        benchmark(dataset, targets, trained, sectors=[14], selection={}, n_bootstrap=0)
    with pytest.raises(ValueError, match="centroid results"):
        with_centroid_features(dataset, tests[:-1])
    split, _ = tiny_trained
    with pytest.raises(ValueError, match="light-curve features"):
        save_model(trained, split, "unused.joblib", preprocess=None, bls=None)


def test_a_light_curve_model_ignores_the_pixel_columns(scenes, tiny_trained):
    dataset, targets, tests = scenes
    _, trained = tiny_trained
    plain = benchmark(dataset, targets, trained, sectors=[14], selection={}, n_bootstrap=0)
    widened = benchmark(
        with_centroid_features(dataset, tests), targets, trained, sectors=[14], selection={},
        n_bootstrap=0,
    )
    np.testing.assert_array_equal(plain.model.scores, widened.model.scores)
    assert "model_features" not in widened.to_dict()
    assert "feature_importance" not in widened.to_dict()


def _hosts(config, n: int, *, seed: int, tic0: int, sector: int):
    """Planets as CP hosts and eclipsing binaries as FP hosts, as MAST would serve them."""
    source = make_source(config, n_curves=n, positive_rate=0.55, eb_rate=0.45, seed=seed)
    curves, tois = [], []
    for i, lc in enumerate(source):
        tic = tic0 + i
        label = int(lc.meta["kind"] == "planet")
        tois.append(
            TOI(tic=tic, toi=f"{tic}.01", disposition="CP" if label else "FP",
                period=float(lc.meta["period"]), snr=float(lc.meta["true_snr"]),
                sectors=(sector,))
        )
        curves.append(
            LightCurve(target_id=f"TIC {tic}", time=lc.time, flux=lc.flux,
                       flux_err=lc.flux_err, label=label,
                       meta={"kind": "real", "sector": sector})
        )
    return curves, tois


def test_training_flags_and_their_defaults(tmp_path):
    import run_pipeline

    table = tmp_path / "toi.csv"
    base = ["--train-sectors", "1-13", "--benchmark-tois", str(table), "--benchmark-sectors", "14-26"]
    args = run_pipeline.parse_args(base)
    assert args.results_dir == run_pipeline.ROOT / "results" / "toi_trained"
    assert args.train_cache == args.benchmark_cache == args.results_dir / "toi_curves.npz"
    assert args.train_tpfs == args.benchmark_tpfs == args.results_dir / "toi_tpfs"
    assert not args.benchmark_centroids

    pixels = run_pipeline.parse_args([*base, "--pixel-features"])
    assert pixels.results_dir == run_pipeline.ROOT / "results" / "toi_trained" / "pixels"
    assert pixels.figures_dir == run_pipeline.ROOT / "figures" / "toi_trained" / "pixels"
    # The run with pixel features reuses what the run without them downloaded.
    assert pixels.train_cache == args.train_cache and pixels.train_tpfs == args.train_tpfs
    assert pixels.benchmark_centroids

    for bad in (
        ["--pixel-features"],
        base[:4],
        [*base, "--inject-into", "targets.txt"],
        [*base, "--systematics"],
    ):
        with pytest.raises(SystemExit):
            run_pipeline.parse_args(bad)


def test_run_pipeline_trains_on_one_sector_and_scores_another(small_config, tmp_path, monkeypatch):
    """The ``--train-sectors`` path end to end, from a TOI table and one shared curve cache."""
    import run_pipeline
    import transitml.benchmark as bench
    import transitml.data.tic as tic_module
    from transitml.data.injection import save_curves

    train_curves, train_tois = _hosts(small_config, 60, seed=11, tic0=6000, sector=14)
    test_curves, test_tois = _hosts(small_config, 24, seed=12, tic0=8000, sector=15)
    shared, moved = train_tois[0], train_tois[1]
    tois = [
        # Seen in the benchmark's sector too: scored there and never trained on,
        # so the benchmark scores the same stars whatever the model learned from.
        replace(shared, sectors=(14, 15)),
        # First seen in sector 13, where MAST has no curve: trained on sector 14.
        replace(moved, sectors=(13, 14)),
        *train_tois[2:],
        *test_tois,
    ]
    table = tmp_path / "exofop_toi.csv"
    table.write_text(
        "TIC ID,TOI,TFOPWG Disposition,Period (days),Planet SNR,Sectors\n"
        + "".join(
            f'{t.tic},{t.toi},{t.disposition},{t.period},{t.snr},"{",".join(map(str, t.sectors))}"\n'
            for t in tois
        )
    )
    results = tmp_path / "results"
    results.mkdir()
    shared_in_15 = replace(train_curves[0], meta={**train_curves[0].meta, "sector": 15})
    save_curves(train_curves + test_curves + [shared_in_15], results / "toi_curves.npz")

    asked: list[tuple[str, int]] = []

    def mast(targets, **_):
        asked.extend((t.target_id, t.sector) for t in targets)
        return []

    monkeypatch.setattr(bench, "fetch_benchmark_curves", mast)
    monkeypatch.setattr(
        tic_module,
        "fetch_tic_stars",
        lambda ids: {tic: {"teff_k": 5800.0, "rho_star_cgs": 1.4} for tic in ids},
    )
    args = run_pipeline.parse_args(
        [
            "--train-sectors", "13-14",
            "--benchmark-tois", str(table),
            "--benchmark-sectors", "15",
            "--results-dir", str(results),
            "--no-figures",
            "--n-jobs", "2",
        ]
    )
    assert run_pipeline.run_toi_training(args, small_config, started=0.0) == 0

    # The one star not in the cache at its first sector, asked for once.
    assert asked == [(f"TIC {moved.tic}", 13)]
    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["dataset"]["n_curves"] == 59 and metrics["dataset"]["n_test"] == 0
    selection = metrics["training"]["selection"]
    assert selection["in_benchmark_sectors"] == 1 and selection["from_a_later_sector"] == 1
    assert metrics["training"]["operating_point"]["rule"].startswith("calibrated P(planet)")
    assert metrics["toi_benchmark"]["n_stars"] == 25

    scored = json.loads((results / "toi_benchmark.json").read_text())
    assert scored["selection"]["in_training_set"] == 0
    trained_on = {lc.target_id for lc in train_curves[1:]}
    assert {s["target_id"] for s in scored["stars"]}.isdisjoint(trained_on)
    assert f"TIC {shared.tic}" in {s["target_id"] for s in scored["stars"]}
    assert len(scored["feature_importance"]) == len(FEATURE_NAMES)
    assert "model_features" not in scored

    calibration = metrics["toi_benchmark"]["calibration"]
    assert sum(row["n"] for row in calibration["reliability"]) == 25
    assert calibration["planet_rate"] == pytest.approx(
        sum(s["disposition"] == "CP" for s in scored["stars"]) / 25
    )

    report = (results / "report.txt").read_text()
    assert "TRAINED ON REAL LABELS: TOI HOSTS OF SECTORS 13, 14" in report
    assert "observed in the benchmark's sectors, so left out: 1" in report
    assert "of which 1 from a later sector than their first" in report
    assert "Scored, unchanged, on the TOI hosts of sectors 15" in report
    assert "calibrated P(planet): Brier score" in report

    saved = load_model(results / "model.joblib")
    assert saved.provenance["source"] == "TOI hosts, sectors 13-14"
    assert saved.threshold_probability == pytest.approx(0.5)
    assert (results / "dataset.npz").exists()
