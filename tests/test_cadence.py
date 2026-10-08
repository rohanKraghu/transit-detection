"""Faster full-frame photometry (sectors 27 onward) averaged to the 30 minutes the features expect."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.data.base import LightCurve, bin_light_curve
from transitml.data.mast import fetch_exposure_seconds, ffi_exposure_seconds
from transitml.data.toi import TOI, BenchmarkTarget
from transitml.features import extract_features
from transitml.preprocess import flatten
from transitml.search import transit_mask

PERIOD, EPOCH, DURATION, DEPTH = 6.0, 1.3, 0.12, 2e-3


def curve(cadence_seconds, *, sigma=1e-3, seed=0, length=27.0):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, length, cadence_seconds / 86400.0)
    flux = np.ones_like(time)
    flux[transit_mask(time, PERIOD, EPOCH, DURATION, 0.5)] -= DEPTH
    flux += rng.normal(0.0, sigma, time.size)
    return LightCurve("TIC 9", time, flux, np.full(time.size, sigma), label=1, meta={"sector": 40})


def test_ten_minute_photometry_is_averaged_three_cadences_to_one():
    fast = curve(600)
    slow = bin_light_curve(fast, 1800)

    assert slow.n_cadences == pytest.approx(fast.n_cadences / 3, abs=1)
    assert np.median(np.diff(slow.time)) * 86400 == pytest.approx(1800)
    np.testing.assert_allclose(slow.flux_err, 1e-3 / np.sqrt(3))
    assert slow.flux[:5] == pytest.approx([fast.flux[3 * i : 3 * i + 3].mean() for i in range(5)])
    assert slow.meta["binned_from_seconds"] == 600 and slow.meta["sector"] == 40
    assert slow.label == 1
    # The transit keeps its depth; the scatter shrinks as the root of the bin.
    in_transit = transit_mask(slow.time, PERIOD, EPOCH, DURATION, 0.5)
    assert 1.0 - np.median(slow.flux[in_transit]) == pytest.approx(DEPTH, rel=0.15)
    assert np.std(slow.flux[~in_transit]) == pytest.approx(1e-3 / np.sqrt(3), rel=0.1)


def test_gap_edges_lose_their_part_filled_bins_and_slow_curves_pass_through():
    fast = curve(200)  # nine cadences to a 30-minute bin, counted from the first
    drop = np.zeros(fast.n_cadences, dtype=bool)
    drop[9:16] = True  # the second bin keeps 2 of its 9 cadences: dropped
    drop[18:22] = True  # the third keeps 5: kept, as their mean
    gappy = LightCurve("TIC 9", fast.time[~drop], fast.flux[~drop], fast.flux_err[~drop])
    slow = bin_light_curve(gappy, 1800)

    n_bins = int(np.ceil(fast.n_cadences / 9))
    assert slow.n_cadences == n_bins - 1
    assert slow.time[1] == pytest.approx(fast.time[22:27].mean())
    assert slow.flux_err[1] == pytest.approx(1e-3 / np.sqrt(5))
    assert np.all(np.diff(slow.time) > 0)

    thirty = curve(1800)
    np.testing.assert_array_equal(bin_light_curve(thirty, 1800).flux, thirty.flux)


def test_binned_photometry_gives_the_features_of_thirty_minute_photometry(fast_bls):
    """Same star, same noise per unit time: the features should not see the cadence."""
    ten_minute = curve(600, sigma=1e-3 * np.sqrt(3), seed=3)
    a = extract_features(flatten(bin_light_curve(ten_minute, 1800)), fast_bls)
    b = extract_features(flatten(curve(1800, sigma=1e-3, seed=4)), fast_bls)
    for name in ("log_depth", "log_period", "min_points_per_transit", "n_transits"):
        assert a[name] == pytest.approx(b[name], rel=0.1, abs=0.1), name
    assert a["log_scatter"] == pytest.approx(b["log_scatter"], abs=0.1)
    assert a["bls_depth_snr"] == pytest.approx(b["bls_depth_snr"], rel=0.25)


def test_ffi_cadence_by_sector_and_what_is_asked_of_mast():
    assert [ffi_exposure_seconds(s) for s in (1, 26, 27, 55, 56, 90)] == [
        1800, 1800, 600, 600, 200, 200,
    ]
    assert fetch_exposure_seconds("TESS-SPOC", 1800, 13) == 1800
    assert fetch_exposure_seconds("TESS-SPOC", 1800, 40) == 600
    assert fetch_exposure_seconds("QLP", 1800, 70) == 200
    # Two-minute targets and other pipelines are asked for as given.
    assert fetch_exposure_seconds("SPOC", 120, 40) == 120
    assert fetch_exposure_seconds("TESS-SPOC", None, 40) is None


def test_a_later_sector_is_fetched_at_its_cadence_and_averaged(monkeypatch):
    from transitml import benchmark as bench

    asked: list[tuple[int | None, int | None]] = []

    class FakeSource:
        def __init__(self, targets, *, exposure_time, sector, **kwargs):
            asked.append((sector, exposure_time))
            self.cadence = exposure_time

        def __iter__(self):
            yield curve(self.cadence)

    monkeypatch.setattr("transitml.data.mast.MASTLightCurveSource", FakeSource)
    toi = TOI(tic=9, toi="9.01", disposition="CP", sectors=(13, 40))
    targets = [
        BenchmarkTarget(tic=9, label=1, sector=40, reference=toi, tois=(toi,)),
    ]
    (lc,) = bench.fetch_benchmark_curves(targets, author="TESS-SPOC", exposure_time=1800)
    assert asked == [(40, 600)]
    assert np.median(np.diff(lc.time)) * 86400 == pytest.approx(1800)
    assert lc.meta["binned_from_seconds"] == 600


def test_pixels_are_averaged_like_the_light_curve():
    from transitml.data.tpf import TargetPixelData, bin_target_pixels

    rng = np.random.default_rng(5)
    time = np.arange(0.0, 2.0, 200.0 / 86400.0)
    flux = 100.0 + rng.normal(0.0, 1.0, (time.size, 3, 4))
    flux[:, 0, 0] = np.nan  # a dead pixel
    flux[10, 1, 1] = np.nan  # one bad value inside the second bin
    aperture = np.zeros((3, 4), dtype=bool)
    aperture[1:, 1:3] = True
    tpf = TargetPixelData("TIC 9", time, flux, aperture, flux_err=np.ones_like(flux),
                          target_position=(1.5, 1.0), meta={"sector": 60})
    binned = bin_target_pixels(tpf, 1800)

    assert binned.n_cadences == time.size // 9
    assert binned.time[1] == pytest.approx(time[9:18].mean())
    assert binned.flux[0, 2, 3] == pytest.approx(flux[:9, 2, 3].mean())
    assert binned.flux[1, 1, 1] == pytest.approx(np.nanmean(flux[9:18, 1, 1]))
    assert np.all(np.isnan(binned.flux[:, 0, 0]))
    assert binned.flux_err[0, 2, 3] == pytest.approx(1.0 / 3.0)
    assert binned.flux_err[1, 1, 1] == pytest.approx(1.0 / np.sqrt(8))
    assert binned.target_position == (1.5, 1.0) and binned.meta["binned_from_seconds"] == 200
    assert bin_target_pixels(binned, 1800) is binned
