"""Iterative multi-planet search: mask the strongest signal, search again."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.config import MultiPlanetConfig
from transitml.data.base import LightCurve
from transitml.features import run_bls, signal_detection_efficiency
from transitml.preprocess import flatten
from transitml.search import iterative_search, transit_mask


def box_planets(planets, *, sigma=3e-4, seed=0, baseline=27.4):
    """White noise plus exact box transits ``(period, epoch, depth, duration)``."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, baseline, 30.0 / 1440.0)
    flux = np.ones_like(time)
    for period, epoch, depth, duration in planets:
        flux[transit_mask(time, period, epoch, duration, 0.5)] -= depth
    flux += rng.normal(0.0, sigma, time.size)
    return LightCurve("TEST-MULTI", time, flux, np.full(time.size, sigma))


def matches(found, period, tol=0.01):
    return abs(found / period - 1.0) < tol


@pytest.mark.parametrize("seed", [0, 1])
def test_two_planets_with_different_periods_are_both_recovered(fast_bls, seed):
    lc = box_planets([(3.0, 1.3, 3e-3, 0.12), (7.3, 2.1, 3e-3, 0.16)], seed=seed)
    found = iterative_search(flatten(lc), fast_bls)

    assert len(found) == 2
    assert matches(found[0].period, 3.0)
    assert matches(found[1].period, 7.3)
    assert [c.rank for c in found] == [1, 2]
    # The second search ran on a masked curve.
    assert found[1].n_cadences_searched < found[0].n_cadences_searched
    assert found[1].depth == pytest.approx(3e-3, rel=0.25)


def test_single_planet_yields_one_candidate_matching_the_primary(fast_bls):
    lc = box_planets([(3.0, 1.3, 3e-3, 0.12)], seed=2)
    flat = flatten(lc)
    found = iterative_search(flat, fast_bls)

    assert len(found) == 1
    assert matches(found[0].period, 3.0)
    # The first pass is exactly the single-signal search the classifier uses.
    primary = run_bls(flat, fast_bls)
    assert found[0].period == primary["period"]
    assert found[0].epoch == primary["transit_time"]
    assert found[0].sde == signal_detection_efficiency(primary["power"])


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_pure_noise_yields_no_candidates(fast_bls, seed):
    """Defined behaviour: a peak below ``min_sde`` is never reported, even the first."""
    flat = flatten(box_planets([], seed=seed))
    config = MultiPlanetConfig()
    assert iterative_search(flat, fast_bls, config) == []
    # And that is because the best peak really is below the threshold.
    sde = signal_detection_efficiency(run_bls(flat, fast_bls)["power"])
    assert sde < config.min_sde


def test_max_signals_caps_the_list(fast_bls):
    lc = box_planets([(3.0, 1.3, 3e-3, 0.12), (7.3, 2.1, 3e-3, 0.16)], seed=0)
    found = iterative_search(flatten(lc), fast_bls, MultiPlanetConfig(max_signals=1))
    assert len(found) == 1 and matches(found[0].period, 3.0)


def test_candidates_serialise_to_plain_numbers(fast_bls):
    lc = box_planets([(3.0, 1.3, 3e-3, 0.12)], seed=2)
    row = iterative_search(flatten(lc), fast_bls)[0].to_dict()
    assert set(row) >= {"period", "epoch", "duration", "depth", "depth_snr", "sde"}
    assert all(isinstance(v, (int, float)) for v in row.values())
