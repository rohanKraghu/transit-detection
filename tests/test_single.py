"""Single and duo transits: dips the periodic search excludes by construction."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.config import SingleEventConfig
from transitml.data.base import LightCurve
from transitml.physics import RHO_SUN_CGS, expected_central_duration
from transitml.preprocess import flatten
from transitml.single import (
    binned_noise_factor,
    period_from_duration,
    search_single_events,
)

GAP = (13.2, 14.2)


def box_events(events, *, sigma=5e-4, seed=0, red=0.0, flares=(), gap=GAP):
    """White (plus optional red) noise, a mid-sector gap and exact box dips."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, 27.4, 30.0 / 1440.0)
    time = time[(time < gap[0]) | (time > gap[1])]
    flux = 1.0 + rng.normal(0.0, sigma, time.size)
    if red:
        walk = np.convolve(rng.normal(0.0, 1.0, time.size), np.ones(12) / 12, "same")
        flux += red * sigma * walk / walk.std()
    for t0, duration, depth in events:
        flux[np.abs(time - t0) < duration / 2] -= depth
    for t0, amp in flares:
        after = time >= t0
        flux[after] += amp * np.exp(-(time[after] - t0) / 0.03)
    return flatten(LightCurve("TEST-SINGLE", time, flux, np.full(time.size, sigma)))


def test_one_transit_is_found_with_its_time_depth_and_duration():
    result = search_single_events(box_events([(8.3, 0.25, 3e-3)]))
    assert len(result.events) == 1 and not result.duos
    event = result.events[0]
    assert event.time == pytest.approx(8.3, abs=0.03)
    assert event.duration == pytest.approx(0.25, rel=0.25)
    assert event.depth == pytest.approx(3e-3, rel=0.2)
    assert event.snr > 10 and not event.near_gap


def test_a_single_transit_bounds_its_own_period():
    """No other transit on flat data: the period exceeds the longer side of the window."""
    event = search_single_events(box_events([(8.3, 0.25, 3e-3)])).events[0]
    assert event.min_period == pytest.approx(27.4 - 8.3, abs=0.3)
    central = expected_central_duration(event.period_estimate, RHO_SUN_CGS)
    assert central == pytest.approx(event.duration, rel=1e-3)


def test_a_transit_whose_partner_could_hide_in_a_gap_allows_that_period():
    """Event at 4 d, data missing from 11 to 16 d: a period of 11.7 to 12 d hides
    the next transit in the gap and the one after past the end, so it stands,
    although it is far shorter than the 23.4 d of data after the event."""
    event = search_single_events(box_events([(4.0, 0.2, 3e-3)], gap=(11.0, 16.0))).events[0]
    assert event.min_period == pytest.approx(11.7, abs=0.15)
    one_gap = search_single_events(box_events([(6.0, 0.2, 3e-3)])).events[0]
    assert one_gap.min_period == pytest.approx(27.4 - 6.0, abs=0.3)


def test_two_matching_transits_make_a_duo_with_only_unexcluded_aliases():
    result = search_single_events(box_events([(4.1, 0.2, 2.5e-3), (22.1, 0.2, 2.5e-3)]))
    assert len(result.events) == 2
    (duo,) = result.duos
    assert duo.separation == pytest.approx(18.0, abs=0.05)
    # 18 d is allowed; 9 d would put a transit at 13.1 d, on data with no dip.
    assert duo.allowed_periods[0] == pytest.approx(18.0, abs=0.05)
    assert all(abs(p - 9.0) > 0.1 for p in duo.allowed_periods)
    assert len(duo.duration_ratios) == len(duo.allowed_periods)


def test_dips_of_different_depth_do_not_pair():
    result = search_single_events(box_events([(4.1, 0.2, 1.5e-3), (22.1, 0.2, 6e-3)]))
    assert len(result.events) == 2 and not result.duos


@pytest.mark.parametrize("seed", range(6))
def test_noise_alone_raises_no_event(seed):
    assert search_single_events(box_events([], seed=seed)).events == ()


def test_flares_are_not_events():
    flares = [(3.0, 8e-3), (17.5, 1.2e-2)]
    assert search_single_events(box_events([], flares=flares)).events == ()


def test_red_noise_is_charged_for():
    lc = box_events([], red=3.0, seed=2)
    assert binned_noise_factor(lc, 0.2) > 1.5
    assert binned_noise_factor(box_events([], seed=2), 0.2) == pytest.approx(1.0, abs=0.25)


def test_one_deep_event_does_not_raise_its_own_noise_floor():
    deep = box_events([(8.3, 0.25, 1e-2)])
    assert binned_noise_factor(deep, 0.25) == pytest.approx(
        binned_noise_factor(box_events([]), 0.25), abs=0.15
    )


def test_period_from_duration_inverts_the_central_duration():
    for period in (20.0, 60.0, 300.0):
        duration = expected_central_duration(period, RHO_SUN_CGS)
        assert period_from_duration(duration, RHO_SUN_CGS) == pytest.approx(period, rel=0.02)


def test_the_threshold_and_event_cap_are_respected():
    events = [(3.0, 0.15, 4e-3), (9.0, 0.15, 4e-3), (17.0, 0.15, 4e-3), (24.0, 0.15, 4e-3)]
    lc = box_events(events)
    assert len(search_single_events(lc, SingleEventConfig(max_events=2)).events) == 2
    assert search_single_events(lc, SingleEventConfig(min_snr=1e6)).events == ()


def test_results_serialise_to_plain_numbers():
    import json

    result = search_single_events(box_events([(4.1, 0.2, 2.5e-3), (22.1, 0.2, 2.5e-3)]))
    text = json.dumps(result.to_dict())
    assert '"allowed_periods"' in text and '"min_period"' in text


def test_the_benchmark_runs_end_to_end(tmp_path):
    import json

    from transitml import single_benchmark

    argv = ["--n-curves", "8", "--n-jobs", "1", "--results-dir", str(tmp_path)]
    assert single_benchmark.main(argv) == 0
    summary = json.loads((tmp_path / "metrics.json").read_text())
    assert summary["clean_stars"] + summary["injected"] + summary["errors"] == 8
    assert sum(summary["by_transits_seen"].values()) == summary["injected"]
    assert "False alarms" in (tmp_path / "report.txt").read_text()
