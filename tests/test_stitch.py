"""Multi-sector stitching: per-sector normalisation, gaps kept, longer baseline searched."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.config import BLSConfig, MultiPlanetConfig, PreprocessConfig
from transitml.data.base import LightCurve, stitch_light_curves
from transitml.features import period_grid, run_bls, signal_detection_efficiency
from transitml.preprocess import flatten, robust_sigma, spline_basis
from transitml.search import iterative_search, transit_mask

PERIOD, EPOCH, DURATION = 10.0, 2.0, 0.15


def sector(start, *, level=1.0, depth=1.5e-3, sigma=1e-3, seed=0, length=27.4):
    """One 27.4-day, 30-minute sector of a variable star with a 10-day planet.

    Transits fall at 2, 12, 22 d, then 32, 42, 52 d: three per sector when the
    second sector starts at 28.4 d.  ``level`` is an arbitrary flux scale, as
    two sectors through different apertures would have.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(start, start + length, 30.0 / 1440.0)
    flux = 1.0 + 1.5e-3 * np.sin(2.0 * np.pi * time / 6.3 + seed)
    flux[transit_mask(time, PERIOD, EPOCH, DURATION, 0.5)] -= depth
    flux += rng.normal(0.0, sigma, time.size)
    return LightCurve("TIC 1", time, flux * level, np.full(time.size, sigma * level),
                      label=1, meta={"sector": 1 if start < 28 else 2})


def test_each_sector_is_normalised_to_its_own_median_and_gaps_are_kept():
    a, b = sector(0.0, level=1.0, seed=1), sector(28.4, level=1.07, seed=2)
    stitched = stitch_light_curves([b, a])  # order of input does not matter

    assert stitched.n_cadences == a.n_cadences + b.n_cadences
    assert np.all(np.diff(stitched.time) > 0)
    first = stitched.time < 28.0
    assert np.median(stitched.flux[first]) == pytest.approx(1.0)
    assert np.median(stitched.flux[~first]) == pytest.approx(1.0)
    np.testing.assert_allclose(stitched.flux_err[~first], b.flux_err / np.median(b.flux))
    # The inter-sector gap is preserved, not filled.
    assert np.max(np.diff(stitched.time)) == pytest.approx(28.4 - a.time[-1])
    assert stitched.baseline_days == pytest.approx(b.time[-1])
    assert stitched.meta["sectors"] == [2, 1] and stitched.meta["n_sectors"] == 2
    assert stitched.label == 1


def test_stitching_refuses_mixed_targets_and_empty_input():
    a = sector(0.0)
    other = LightCurve("TIC 2", a.time, a.flux, a.flux_err)
    with pytest.raises(ValueError, match="different targets"):
        stitch_light_curves([a, other])
    with pytest.raises(ValueError, match="nothing"):
        stitch_light_curves([])


def test_duplicate_cadences_are_kept_once():
    a = sector(0.0, seed=1)
    stitched = stitch_light_curves([a, a])
    np.testing.assert_array_equal(stitched.time, a.time)


def test_period_grid_reaches_half_the_stitched_baseline():
    a, b = sector(0.0, seed=1), sector(28.4, seed=2)
    flat = flatten(stitch_light_curves([a, b]))
    config = BLSConfig()
    grid = period_grid(flat.baseline_days, config)
    assert grid[-1] == pytest.approx(0.5 * flat.baseline_days)
    assert grid[-1] > 0.5 * flatten(a).baseline_days * 1.9
    result = run_bls(flat, config)
    assert result["periods"].max() == pytest.approx(grid[-1])


def test_no_spline_basis_function_spans_a_sector_gap():
    """Each sector, and each side of a multi-week gap, gets its own spline segment."""
    a, b = sector(0.0, seed=1), sector(80.0, seed=2)  # a seven-week gap
    stitched = stitch_light_curves([a, b])
    basis = spline_basis(stitched.time, PreprocessConfig())
    before = stitched.time < 50.0
    on_both_sides = (np.abs(basis[before]).sum(axis=0) > 0) & (
        np.abs(basis[~before]).sum(axis=0) > 0
    )
    assert not on_both_sides.any()


def test_detrending_absorbs_a_level_jump_across_the_gap():
    """Even un-normalised sectors (a 7% step) detrend cleanly: the step sits in the gap."""
    a, b = sector(0.0, depth=0.0, seed=1), sector(80.0, level=1.07, depth=0.0, seed=2)
    raw = LightCurve("TIC 1", np.concatenate([a.time, b.time]),
                     np.concatenate([a.flux, b.flux]), np.concatenate([a.flux_err, b.flux_err]))
    flat = flatten(raw)
    for side in (flat.time < 50.0, flat.time >= 50.0):
        assert abs(np.median(flat.flux[side]) - 1.0) < 2e-4
        assert robust_sigma(flat.flux[side]) == pytest.approx(1e-3 / 1.035, rel=0.15)


def test_two_marginal_sectors_stitch_into_a_detection():
    """Neither sector alone reaches the search threshold at 10 d; together they do.

    Three transits per sector at depth SNR ~7 each.  Seeds are fixed: in one
    sector the best peak is at the wrong period, in the other it is at 10 d but
    below ``min_sde``.  Stitched, the six transits fold to a peak at 10 d well
    above it.  The default 2000-period grid is used because the stitched
    baseline needs the finer period steps.
    """
    bls, threshold = BLSConfig(), MultiPlanetConfig().min_sde
    a, b = sector(0.0, seed=4), sector(28.4, level=1.07, seed=5)

    for single in (a, b):
        result = run_bls(flatten(single), bls)
        on_period = abs(result["period"] / PERIOD - 1.0) < 0.01
        sde = signal_detection_efficiency(result["power"])
        assert not (on_period and sde >= threshold)

    flat = flatten(stitch_light_curves([a, b]))
    result = run_bls(flat, bls)
    assert result["period"] == pytest.approx(PERIOD, rel=0.002)
    assert signal_detection_efficiency(result["power"]) >= threshold + 1.0
    found = iterative_search(flat, bls)
    assert found and found[0].period == pytest.approx(PERIOD, rel=0.002)
