"""Curves too long for one BLS grid: candidates from the densest stretch, chosen on all of it."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from transitml.config import BLSConfig
from transitml.data.base import LightCurve
from transitml.features import (
    densest_window,
    distinct_peaks,
    extract_features,
    run_bls,
    signal_detection_efficiency,
)
from transitml.preprocess import flatten
from transitml.search import transit_mask

CADENCE = 30.0 / 1440.0


def segments(starts, length):
    """Cadences every 30 minutes in ``[start, start + length)`` for each start."""
    return np.concatenate([s + np.arange(0.0, length, CADENCE) for s in starts])


def planet_curve(time, period, epoch, depth=2e-3, duration=0.12, sigma=5e-4, seed=0):
    rng = np.random.default_rng(seed)
    flux = np.ones_like(time)
    flux[transit_mask(time, period, epoch, duration, 0.5)] -= depth
    flux += rng.normal(0.0, sigma, time.size)
    return LightCurve("TEST-WINDOW", time, flux, np.full(time.size, sigma))


def test_densest_window_is_the_fullest_stretch_and_the_earliest_of_ties():
    time = np.concatenate([np.arange(0, 10, 1.0), np.arange(100, 130, 0.5), np.arange(500, 510, 1.0)])
    assert densest_window(time, 50.0) == (100.0, 150.0)
    assert densest_window(np.array([0.0, 1.0, 10.0, 11.0]), 2.0) == (0.0, 2.0)


def test_distinct_peaks_skip_a_peaks_neighbours_and_its_aliases():
    periods = np.linspace(1.0, 20.0, 1901)
    power = np.zeros_like(periods)
    for centre, height in [(5.0, 10.0), (10.0, 8.0), (7.3, 6.0), (2.5, 5.0), (13.1, 4.0)]:
        power += height * np.exp(-0.5 * ((periods - centre) / 0.02) ** 2)

    taken = periods[distinct_peaks(periods, power, 3)]
    # 10 and 2.5 are the double and half of 5; the rest of each bump is the same peak.
    np.testing.assert_allclose(taken, [5.0, 7.3, 13.1], atol=0.006)
    assert distinct_peaks(periods, np.full(periods.size, np.nan), 3) == []


def test_off_unless_the_curve_is_longer_than_the_window(fast_bls):
    lc = flatten(planet_curve(segments([0.0], 27.0), 3.1, 1.2))
    plain = run_bls(lc, fast_bls)
    windowed = run_bls(lc, replace(fast_bls, max_search_baseline_days=400.0))
    assert "search_window" not in plain and "search_window" not in windowed
    assert windowed["period"] == plain["period"]
    np.testing.assert_array_equal(windowed["power"], plain["power"])


def test_a_planet_found_in_the_window_is_timed_on_the_whole_curve():
    period, epoch = 7.31234, 2.1
    # Three sectors in a row, then one a year for three years.
    time = segments([0.0, 27.4, 54.8, 700.0, 1100.0, 1500.0], 26.0)
    lc = flatten(planet_curve(time, period, epoch))
    config = BLSConfig(max_search_baseline_days=400.0)

    found = run_bls(lc, config)
    assert found["search_window"][0] == 0.0
    # One grid over the window alone cannot time the transits years later.
    window_period = found["periods"][found["best_index"]]
    late = 1525.0 - epoch
    assert abs(late / window_period - late / period) * window_period > 0.12
    # Refined on every sector, the predicted transit 200 orbits later is on time.
    assert found["period"] == pytest.approx(period, rel=2e-5)
    drift = (found["transit_time"] - epoch + 0.5 * period) % period - 0.5 * period
    assert abs(drift) < 0.03
    assert found["depth"] == pytest.approx(2e-3, rel=0.15)


def test_the_rest_of_the_curve_settles_a_period_the_window_cannot_tell_from_its_double():
    period, epoch = 6.0, 1.0
    # A year of 5.5-day stretches 12 days apart: every other transit falls in a gap,
    # so 6 and 12 days fold the same events.  Two whole sectors later have them all.
    window = segments(np.arange(0.0, 360.0, 12.0), 5.5)
    time = np.concatenate([window, segments([800.0, 1300.0], 26.0)])
    lc = flatten(planet_curve(time, period, epoch, seed=3))
    config = BLSConfig(n_periods=600, max_search_baseline_days=400.0)

    alone = run_bls(
        flatten(planet_curve(window, period, epoch, seed=3)),
        replace(config, max_search_baseline_days=None),
    )
    near = lambda p, q: abs(p / q - 1.0) < 0.01
    peak = lambda p: alone["power"][np.abs(np.log(alone["periods"] / p)).argmin()]
    assert peak(6.0) == pytest.approx(peak(12.0), rel=0.1), "the window alone cannot tell them"

    found = run_bls(lc, config)
    assert found["search_window"][1] < 800.0
    assert near(found["period"], period)


def test_features_describe_the_peak_that_won():
    time = segments([0.0, 27.4, 54.8, 700.0, 1100.0, 1500.0], 26.0)
    lc = flatten(planet_curve(time, 7.31234, 2.1))
    config = BLSConfig(n_periods=600, max_search_baseline_days=400.0)
    found = run_bls(lc, config)

    features = extract_features(lc, config, search=found)
    peak = found["power"][found["best_index"]]
    assert features["bls_sde"] == signal_detection_efficiency(found["power"], peak)
    assert features["log_period"] == pytest.approx(np.log10(found["period"]))
    assert features["n_transits"] > 20
    assert all(np.isfinite(features[k]) for k in ("bls_depth_snr", "power_contrast"))
