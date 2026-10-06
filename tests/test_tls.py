"""Transit Least Squares as the periodic search: same features, different engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from astropy.timeseries import BoxLeastSquares

from tests.conftest import clean_transit_curve
from transitml import features
from transitml.features import FEATURE_NAMES, extract_features, run_bls
from transitml.preprocess import flatten
from transitml.tls import box_statistics


def test_box_statistics_are_astropys():
    lc, truth = clean_transit_curve()
    flat = flatten(lc)
    err = np.where(flat.flux_err > 0, flat.flux_err, flat.scatter)
    period, duration, epoch = truth["period"], truth["duration"], truth["epoch"]
    ours = box_statistics(flat.time, flat.flux, err, period, duration, epoch)
    ref = BoxLeastSquares(flat.time, flat.flux, err).compute_stats(period, duration, epoch)
    assert ours["depth"] == pytest.approx(float(ref["depth"][0]), rel=1e-9)
    assert ours["depth_err"] == pytest.approx(float(ref["depth"][1]), rel=1e-9)
    assert ours["depth"] == pytest.approx(truth["depth"], rel=0.1)


def test_tls_finds_the_period_and_every_feature_is_computed(fast_bls):
    pytest.importorskip("transitleastsquares")
    lc, truth = clean_transit_curve(depth=1.5e-3, sigma=5e-4)
    flat = flatten(lc)
    config = replace(fast_bls, search="tls")
    result = run_bls(flat, config)
    assert result["searched_with"] == "tls"
    assert result["period"] == pytest.approx(truth["period"], rel=0.005)
    values = extract_features(flat, config)
    assert set(values) == set(FEATURE_NAMES)
    assert values["bls_depth_snr"] > 10
    assert values["log_period"] == pytest.approx(np.log10(truth["period"]), abs=0.003)


def test_a_tls_failure_falls_back_to_bls(fast_bls, monkeypatch):
    lc, truth = clean_transit_curve()

    def broken(*args, **kwargs):
        raise RuntimeError("TLS returned no finite solution")

    import transitml.tls

    monkeypatch.setattr(transitml.tls, "run_tls", broken)
    result = run_bls(flatten(lc), replace(fast_bls, search="tls"))
    assert result["searched_with"] == "bls"
    assert result["period"] == pytest.approx(truth["period"], rel=0.01)


def test_an_unknown_search_is_refused(fast_bls):
    lc, _ = clean_transit_curve()
    with pytest.raises(ValueError, match="'bls' or 'tls'"):
        features.run_bls(flatten(lc), replace(fast_bls, search="fft"))


def test_the_pipeline_flag_selects_tls_and_its_own_results(tmp_path):
    import run_pipeline

    args = run_pipeline.parse_args(["--search", "tls"])
    assert args.results_dir == run_pipeline.ROOT / "results" / "tls"
    assert args.figures_dir == run_pipeline.ROOT / "figures" / "tls"
    config = run_pipeline.apply_overrides(run_pipeline.default_config(), args)
    assert config.bls.search == "tls"
    plain = run_pipeline.apply_overrides(
        run_pipeline.default_config(), run_pipeline.parse_args([])
    )
    assert plain.bls.search == "bls"
    args = run_pipeline.parse_args(["--search", "tls", "--results-dir", str(tmp_path)])
    assert args.results_dir == tmp_path


def test_the_search_comparison_runs_end_to_end(tmp_path):
    pytest.importorskip("transitleastsquares")
    import json

    from transitml import search_benchmark

    argv = ["--n-curves", "3", "--n-jobs", "1", "--results-dir", str(tmp_path)]
    assert search_benchmark.main(argv) == 0
    summary = json.loads((tmp_path / "metrics.json").read_text())
    assert sum(row["n"] for row in summary["by_snr"]) == 3
    assert summary["tls_ms_per_curve"] > 0 and summary["bls_ms_per_curve"] > 0
    assert "BLS against TLS" in (tmp_path / "report.txt").read_text()
