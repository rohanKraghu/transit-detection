"""Centroid tests on synthetic pixels: on-target transits pass, blends are caught."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from transitml.centroid import (
    CentroidConfig,
    _significance,
    centroid_test,
    gaussian_sigma_2d,
)
from transitml.data.synthetic_tpf import PixelStar, blend_scenario, synthetic_tpf
from transitml.data.tpf import TESS_PIXEL_SCALE_ARCSEC


def run(tpf, truth, config=None):
    return centroid_test(
        tpf, truth["period"], truth["epoch"], truth["duration"], config
    )


def towards(result, truth) -> float:
    """Cosine between the measured offset and the target-to-neighbour direction."""
    offset = np.asarray(result.offset_pixels)
    direction = np.subtract(truth["neighbour_position"], truth["target_position"])
    return float(offset @ direction / (np.hypot(*offset) * np.hypot(*direction)))


def test_on_target_transit_has_no_significant_offset():
    tpf, truth = blend_scenario("on_target", seed=11)
    result = run(tpf, truth)
    assert result.status == "ok" and result.reference == "target_position"
    assert result.difference_snr > 50
    assert result.offset_sigma < 3.0 and not result.significant
    assert result.offset_distance_pixels < 0.5
    assert "no significant offset" in result.verdict


def test_blended_neighbour_gives_a_significant_offset_towards_it():
    tpf, truth = blend_scenario("blend", seed=11)
    result = run(tpf, truth)
    assert result.status == "ok" and result.significant
    assert result.offset_sigma >= 3.0
    assert towards(result, truth) > 0.95
    separation = np.hypot(
        *np.subtract(truth["neighbour_position"], truth["target_position"])
    )
    assert 0.6 * separation < result.offset_distance_pixels <= 1.05 * separation
    assert result.offset_arcsec == pytest.approx(
        result.offset_distance_pixels * TESS_PIXEL_SCALE_ARCSEC
    )
    assert result.verdict.startswith("OFFSET")
    # the centroid moves away from the neighbour while it fades, far more than
    # an on-target transit of that depth could move it
    assert np.dot(result.centroid_shift_pixels, towards_vector(truth)) < 0
    assert result.shift_vs_on_target_sigma >= 3.0


def towards_vector(truth):
    return np.subtract(truth["neighbour_position"], truth["target_position"])


def test_out_of_transit_reference_is_biased_by_the_neighbour_but_not_used_for_the_flag():
    """A crowded stamp's out-of-transit centroid sits between the stars.

    An on-target transit therefore shows an offset from it pointing away from
    the neighbour; the catalogue position does not have that bias, so it is
    the reference the flag uses when it is known.
    """
    tpf, truth = blend_scenario("on_target", seed=12, neighbour_delta_mag=1.0)
    result = run(tpf, truth)
    from_oot = np.asarray(result.offset_from_oot_pixels)
    assert np.dot(from_oot, towards_vector(truth)) < 0
    assert np.hypot(*from_oot) > np.hypot(*result.offset_pixels)
    assert not result.significant
    # an on-target transit moves the centroid by about what is expected
    assert result.shift_vs_on_target_sigma < 3.0


@pytest.mark.parametrize("kind, flagged", [("on_target", False), ("blend", True)])
def test_without_a_catalogue_position_the_out_of_transit_centroid_is_the_reference(
    kind, flagged
):
    tpf, truth = blend_scenario(kind, seed=13)
    tpf.target_position = None
    result = run(tpf, truth)
    assert result.reference == "out_of_transit_centroid"
    assert result.expected_shift_on_target_pixels is None
    assert result.significant is flagged
    if flagged:
        assert towards(result, truth) > 0.9


def test_few_transits_fall_back_to_a_cadence_bootstrap():
    on, truth = blend_scenario("on_target", baseline=4.0, seed=14)
    blend, _ = blend_scenario("blend", baseline=4.0, binary_depth=0.15, seed=14)
    on_result, blend_result = run(on, truth), run(blend, truth)
    for result in (on_result, blend_result):
        assert result.n_transits == 1
        assert "cadences" in result.uncertainty_method
    assert not on_result.significant
    assert blend_result.significant and towards(blend_result, truth) > 0.9


def test_false_alarm_rate_and_power_over_many_scenes():
    on_flags = blend_flags = 0
    on_sigmas = []
    for seed in range(40):
        on, truth = blend_scenario("on_target", seed=100 + seed)
        blend, _ = blend_scenario("blend", seed=100 + seed)
        on_result = run(on, truth)
        on_sigmas.append(on_result.offset_sigma)
        on_flags += on_result.significant
        blend_flags += run(blend, truth).significant
    assert on_flags <= 1
    assert np.mean(np.asarray(on_sigmas) > 2.0) < 0.15  # nominal 4.6%
    assert blend_flags >= 36


def test_no_event_leaves_the_difference_image_undetected():
    tpf, truth = blend_scenario("none", seed=15)
    result = run(tpf, truth)
    assert result.status == "weak_difference_image" and not result.significant
    assert result.verdict.startswith("inconclusive")


# --------------------------------------------------------------------------
# degenerate input does not crash
# --------------------------------------------------------------------------
def test_no_in_transit_cadences():
    tpf, truth = blend_scenario("blend", seed=16, baseline=1.0)  # ends before epoch 1.4
    result = run(tpf, truth)
    assert result.status == "no_in_transit_cadences" and not result.significant
    json.dumps(result.to_dict(), default=float)


def test_no_out_of_transit_cadences():
    tpf, truth = blend_scenario("blend", seed=16)
    config = CentroidConfig(gap_durations=50.0)
    result = run(tpf, truth, config)
    assert result.status == "no_out_of_transit_cadences" and not result.significant


@pytest.mark.parametrize("period", [float("nan"), 0.0, -1.0])
def test_bad_ephemeris(period):
    tpf, truth = blend_scenario("blend", seed=16)
    result = centroid_test(tpf, period, truth["epoch"], truth["duration"])
    assert result.status == "bad_ephemeris" and not result.significant


def test_nan_pixels_and_cadences_are_skipped():
    tpf, truth = blend_scenario("blend", seed=17)
    rng = np.random.default_rng(0)
    tpf.flux[:, :, 0] = np.nan  # a dead column
    tpf.flux[10] = np.nan  # a whole missing frame
    hits = rng.integers(0, tpf.n_cadences, size=30)
    tpf.flux[hits, 5, 5] = np.nan  # scattered NaNs in a bright pixel
    tpf.flux[rng.integers(0, tpf.n_cadences, size=200), 3, 8] = (
        np.nan
    )  # a mostly bad pixel
    result = run(tpf, truth)
    assert result.status == "ok" and result.significant
    assert towards(result, truth) > 0.95
    assert not result.window[:, 0].any()
    assert np.isnan(result.difference_image[:, 0]).all()


def test_all_nan_pixels():
    tpf, truth = blend_scenario("blend", seed=18, baseline=6.0)
    tpf.flux[:] = np.nan
    result = run(tpf, truth)
    assert result.status != "ok" and not result.significant


def test_a_target_off_the_stamp_has_no_window():
    tpf, truth = blend_scenario("blend", seed=19, baseline=6.0)
    tpf.target_position = (40.0, 40.0)
    result = run(tpf, truth)
    assert result.status == "no_valid_pixels" and not result.significant


def test_isolated_star_without_position():
    star = PixelStar(5.3, 4.6, 2e4, eclipse_depth=0.004)
    tpf = synthetic_tpf([star], period=2.5, epoch=0.7, duration=0.12, seed=20)
    tpf.target_position = None
    result = centroid_test(tpf, 2.5, 0.7, 0.12)
    assert result.status == "ok" and not result.significant
    assert result.reference_position == pytest.approx((5.3, 4.6), abs=0.05)


def test_dictionary_is_json_ready_after_nan_cleaning():
    from transitml.vet import _json_safe

    tpf, truth = blend_scenario("blend", seed=21, baseline=8.0)
    payload = _json_safe(run(tpf, truth).to_dict())
    text = json.dumps(payload, allow_nan=False)
    assert "difference_image" not in payload and "OFFSET" in text
    assert payload["pixel_scale_arcsec"] == TESS_PIXEL_SCALE_ARCSEC


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def test_gaussian_equivalent_significance():
    p3 = math.erfc(3.0 / math.sqrt(2.0))  # two-sided 3 sigma
    assert gaussian_sigma_2d(-2.0 * math.log(p3)) == pytest.approx(3.0, abs=1e-9)
    assert gaussian_sigma_2d(0.0) == 0.0
    assert math.isfinite(gaussian_sigma_2d(1e5)) and gaussian_sigma_2d(1e5) > 300
    assert math.isnan(gaussian_sigma_2d(float("nan")))


def test_few_units_make_the_same_distance_less_significant():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 0.1, size=(2000, 2))
    offset = np.array([0.4, 0.0])
    as_chi2, errors = _significance(offset, samples)
    five, _ = _significance(offset, samples, n_units=5)
    fifty, _ = _significance(offset, samples, n_units=50)
    assert errors == pytest.approx([0.1, 0.1], rel=0.1)
    assert five < fifty < as_chi2


def test_divergent_bootstrap_samples_do_not_set_the_covariance():
    rng = np.random.default_rng(1)
    samples = rng.normal(0.0, 0.1, size=(1000, 2))
    samples[:5] = 1e6  # a resample whose difference flux was close to zero
    sigma, errors = _significance(np.array([0.5, 0.0]), samples)
    assert errors == pytest.approx([0.1, 0.1], rel=0.15) and sigma > 4
