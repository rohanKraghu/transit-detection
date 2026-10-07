"""The array BLS reproduces astropy's periodogram, so the engine never changes a result."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.timeseries import BoxLeastSquares

from tests.conftest import clean_transit_curve, make_source
from transitml.fastbls import array_module, bls_power
from transitml.features import BLS_ENGINE_ENV, bls_engine, extract_features, period_grid
from transitml.preprocess import flatten

FIELDS = ("power", "depth", "depth_err", "duration", "depth_snr", "log_likelihood")


def _curves(config, n=4):
    source = make_source(config, n, 0.34, 0.33, seed=21)
    return [flatten(source.generate(i), config.preprocess) for i in range(n)]


def _grid(flat, config):
    periods = period_grid(flat.baseline_days, config.bls)
    durations = np.asarray(config.bls.durations_days)
    return periods, durations[durations < 0.5 * periods.min()]


def test_periodogram_matches_astropy_on_planets_binaries_and_noise(config):
    for flat in _curves(config):
        err = np.where(flat.flux_err > 0, flat.flux_err, flat.scatter)
        periods, durations = _grid(flat, config)
        ref = BoxLeastSquares(flat.time, flat.flux, err).power(periods, durations, objective="snr")
        ours = bls_power(flat.time, flat.flux, err, periods, durations)
        for field in FIELDS:
            np.testing.assert_allclose(ours[field], np.asarray(getattr(ref, field)), rtol=1e-9)
        best = int(np.argmax(ref.power))
        assert int(np.argmax(ours["power"])) == best
        assert ours["transit_time"][best] == pytest.approx(float(ref.transit_time[best]), abs=1e-9)


def test_unweighted_input_matches_too():
    lc, _ = clean_transit_curve()
    periods = np.linspace(1.0, 8.0, 300)
    durations = np.array([0.05, 0.1, 0.2])
    ref = BoxLeastSquares(lc.time, lc.flux).power(periods, durations, objective="snr")
    ours = bls_power(lc.time, lc.flux, None, periods, durations)
    np.testing.assert_allclose(ours["power"], np.asarray(ref.power), rtol=1e-9)


def test_small_blocks_give_the_same_answer():
    lc, truth = clean_transit_curve()
    periods = np.linspace(1.0, 8.0, 200)
    durations = np.array([0.06, 0.12])
    whole = bls_power(lc.time, lc.flux, lc.flux_err, periods, durations)
    pieces = bls_power(lc.time, lc.flux, lc.flux_err, periods, durations, block_elements=5000)
    for field in FIELDS:
        np.testing.assert_array_equal(whole[field], pieces[field])
    assert periods[np.argmax(whole["power"])] == pytest.approx(truth["period"], rel=0.01)


def test_the_engine_does_not_change_a_feature(config, monkeypatch):
    curves = _curves(config, n=3)
    monkeypatch.setenv(BLS_ENGINE_ENV, "astropy")
    reference = [extract_features(flat, config.bls) for flat in curves]
    monkeypatch.setenv(BLS_ENGINE_ENV, "cpu")
    assert bls_engine() == "cpu"
    for flat, ref in zip(curves, reference, strict=True):
        ours = extract_features(flat, config.bls)
        for name, value in ref.items():
            if np.isfinite(value):
                assert ours[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name
            else:
                assert not np.isfinite(ours[name]), name


def test_bad_engines_and_grids_are_refused(monkeypatch):
    monkeypatch.setenv(BLS_ENGINE_ENV, "quantum")
    with pytest.raises(ValueError, match="astropy, cpu or gpu"):
        bls_engine()
    with pytest.raises(ValueError, match="cpu"):
        array_module("tpu")
    lc, _ = clean_transit_curve()
    with pytest.raises(ValueError, match="shortest period"):
        bls_power(lc.time, lc.flux, lc.flux_err, np.array([0.3, 1.0]), np.array([0.5]))


def test_the_gpu_engine_says_what_it_needs():
    try:
        import cupy  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="CuPy"):
            array_module("gpu")
    else:  # pragma: no cover - only on a machine with CuPy
        assert array_module("gpu") is cupy


def test_the_timing_command_checks_agreement(capsys):
    from transitml import fastbls

    assert fastbls.main(["--n-curves", "2"]) == 0
    out = capsys.readouterr().out
    assert "same best period on 2 of 2" in out and "cpu engine" in out
