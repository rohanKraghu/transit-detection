"""BLS must recover what was injected, and the vetting statistics must fire."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.config import PreprocessConfig
from transitml.data.base import LightCurve
from transitml.data.synthetic import trapezoid_transit
from transitml.features import (
    FEATURE_NAMES,
    _pair_sigma,
    beta_inflation,
    depth_scatter_ratio,
    event_depths,
    extract_features,
    flat_bottom_fraction,
    flatten_masked,
    period_grid,
    red_noise_beta,
    run_bls,
    secondary_excess,
    signal_detection_efficiency,
)
from transitml.physics import max_occultation_fraction
from transitml.preprocess import flatten

from .conftest import clean_transit_curve, make_source


def _matches_period(found: float, true: float) -> bool:
    """Accept the true period or a low-order alias, as any BLS search would."""
    ratio = found / true
    return min(abs(ratio - m) for m in (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)) < 0.03


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_injected_transits_are_recoverable_at_high_snr(seed, fast_bls):
    """A high-SNR injected transit must come back out of the full pipeline.

    This is the end-to-end sanity check on generate -> detrend -> BLS: period
    within a low-order alias, depth within 15%, and the search SNR high.
    """
    lc, truth = clean_transit_curve(
        period=3.7,
        depth=4e-3,
        duration=0.13,
        sigma=2e-4,
        variability_amplitude=2e-3,
        variability_period=5.0,
        seed=seed,
    )
    assert truth["snr"] > 30

    result = run_bls(flatten(lc, PreprocessConfig()), fast_bls)

    assert _matches_period(result["period"], truth["period"]), (
        f"BLS found P = {result['period']:.3f} d, injected {truth['period']:.3f} d"
    )
    assert result["depth"] == pytest.approx(truth["depth"], rel=0.15)
    assert result["depth_snr"] > 10


def test_recovery_rate_over_the_injected_population(config, fast_bls):
    """Above SNR ~20 the search should recover essentially every injection.

    Below it, it should not -- a pipeline that claims to recover 5-sigma
    signals is measuring something other than the transit.
    """
    source = make_source(config, 400, 1.0, 0.0, seed=7)
    strong_hits, strong_total = 0, 0
    weak_hits, weak_total = 0, 0

    for index in range(45):
        lc = source.generate(index)
        result = run_bls(flatten(lc, config.preprocess), fast_bls)
        hit = _matches_period(result["period"], lc.meta["period"])
        if lc.meta["true_snr"] > 20:
            strong_total += 1
            strong_hits += hit
        elif lc.meta["true_snr"] < 5:
            weak_total += 1
            weak_hits += hit

    assert strong_total >= 15
    assert strong_hits / strong_total > 0.85, (
        f"only {strong_hits}/{strong_total} high-SNR injections recovered"
    )
    if weak_total >= 3:
        assert weak_hits / weak_total < 0.5


def test_feature_vector_schema(config, fast_bls):
    lc = make_source(config, 20, 1.0, 0.0, seed=1).generate(0)
    features = extract_features(flatten(lc, config.preprocess), fast_bls)

    assert tuple(features) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in features.values())
    # NaN is allowed (it means "the test could not be run") but not everywhere.
    finite = sum(np.isfinite(v) for v in features.values())
    assert finite >= len(FEATURE_NAMES) - 3


def test_secondary_eclipse_is_detected(config, fast_bls):
    """The secondary-eclipse test has to fire on a binary that has one."""
    lc, truth = clean_transit_curve(period=2.9, depth=6e-3, duration=0.12, sigma=1.5e-4, seed=3)
    with_secondary = lc.flux - trapezoid_transit(
        lc.time, truth["period"], truth["epoch"] + truth["period"] / 2, 3e-3, 0.12, 0.09
    )
    lc_secondary = type(lc)(
        lc.target_id, lc.time, with_secondary, lc.flux_err, 0, dict(lc.meta)
    )

    plain = extract_features(flatten(lc, config.preprocess), fast_bls)
    binary = extract_features(flatten(lc_secondary, config.preprocess), fast_bls)

    assert binary["secondary_sigma"] > 5.0
    assert binary["secondary_sigma"] > 3 * abs(plain["secondary_sigma"])


def _with_secondary(lc, truth, fraction, duration):
    """``lc`` with a flat-bottomed dip of ``fraction`` times the transit depth at phase 0.5."""
    dip = trapezoid_transit(
        lc.time,
        truth["period"],
        truth["epoch"] + truth["period"] / 2,
        fraction * truth["depth"],
        duration,
        0.75 * duration,
    )
    return LightCurve(lc.target_id, lc.time, lc.flux - dip, lc.flux_err, lc.label, dict(lc.meta))


def test_a_planets_occultation_is_not_a_binarys_secondary(config, fast_bls):
    """A hot Jupiter's own occultation must not read as a secondary eclipse.

    On a bright star the occultation of a close-in giant, a few percent of the
    transit depth, is significant by itself: on the TOI benchmark it was
    what still rejected several confirmed hot Jupiters.  The test counts only
    what is deeper than any planet's occultation could be, so the planet
    passes while a binary's secondary at the same period still fires.
    """
    lc, truth = clean_transit_curve(period=1.7, depth=1e-2, duration=0.1, sigma=1.5e-4, seed=3)
    planet = _with_secondary(lc, truth, 0.03, 0.1)
    binary = _with_secondary(lc, truth, 0.3, 0.1)

    flat = flatten(planet, config.preprocess)
    res = run_bls(flat, fast_bls)
    st = res["bls"].compute_stats(res["period"], res["duration"], res["transit_time"])
    sec_value, sec_err = (float(v) for v in np.ravel(st["depth_phased"])[:2])
    assert sec_value / sec_err > 5.0  # the occultation itself is plainly detected
    assert extract_features(flat, fast_bls)["secondary_sigma"] == 0.0

    binary_features = extract_features(flatten(binary, config.preprocess), fast_bls)
    assert binary_features["secondary_sigma"] > 5.0


def test_secondary_excess_allows_only_a_planets_occultation():
    period, depth = 2.0, 1e-2
    allowance = max_occultation_fraction(period) * depth
    assert 0.0 < allowance < 0.2 * depth
    assert secondary_excess(0.5 * allowance, depth, period) == 0.0
    assert secondary_excess(allowance + 1e-4, depth, period) == pytest.approx(1e-4)
    # A brightening at phase 0.5 is not a dip to forgive.
    assert secondary_excess(-3e-4, depth, period) == -3e-4
    # The allowance shrinks fast with period: hot Jupiters are hot because they are close.
    assert max_occultation_fraction(1.0) > 5 * max_occultation_fraction(4.0)


def test_odd_even_difference_is_detected(config, fast_bls):
    """Alternating eclipse depths are the other classic binary giveaway."""
    time = np.arange(0.0, 27.4, 1 / 48)
    rng = np.random.default_rng(0)
    period, depth = 3.1, 8e-3

    def build(odd_even: float):
        dip = trapezoid_transit(time, period, 1.0, depth, 0.14, 0.1, odd_even_fraction=odd_even)
        flux = 1.0 - dip + rng.normal(0.0, 1e-4, time.size)
        from transitml.data.base import LightCurve

        return LightCurve("EB", time, flux, np.full(time.size, 1e-4), 0, {})

    even_depths = extract_features(flatten(build(0.0), config.preprocess), fast_bls)
    alternating = extract_features(flatten(build(0.3), config.preprocess), fast_bls)

    assert alternating["odd_even_sigma"] > 5.0
    assert alternating["odd_even_sigma"] > 3 * even_depths["odd_even_sigma"]


def test_flat_bottom_fraction_separates_box_from_v_shape(config):
    """V-shaped events must score lower than flat-bottomed ones."""
    time = np.arange(0.0, 27.4, 1 / 48)
    rng = np.random.default_rng(1)
    period, depth = 3.3, 1e-2

    def shape_of(t23: float) -> float:
        dip = trapezoid_transit(time, period, 1.0, depth, 0.16, t23)
        flux = 1.0 - dip + rng.normal(0.0, 5e-5, time.size)
        from transitml.data.base import LightCurve

        lc = LightCurve("S", time, flux, np.full(time.size, 5e-5), None, {})
        flat = flatten(lc, config.preprocess)
        return flat_bottom_fraction(flat, period, 0.16, 1.0)

    box = shape_of(0.13)
    vee = shape_of(0.0)
    assert box > vee + 0.2, f"box {box:.2f} vs V {vee:.2f}"


def test_duration_ratio_flags_impossible_geometry(config, fast_bls):
    """An event far longer than Kepler's third law permits is not a planet."""
    lc, _ = clean_transit_curve(period=1.2, depth=5e-3, duration=0.3, sigma=1e-4, seed=6)
    features = extract_features(flatten(lc, config.preprocess), fast_bls)
    # log10(measured / maximum-allowed); a real planet sits at or below 0.
    assert features["log_duration_ratio"] > 0.0


def test_red_noise_beta_responds_to_correlated_noise(config, fast_bls):
    """beta ~ 1 for white noise, > 1 when structure lives on transit timescales."""
    from transitml.data.base import LightCurve
    from transitml.data.synthetic import power_law_noise

    time = np.arange(0.0, 27.4, 1 / 48)
    rng = np.random.default_rng(2)
    white = 1.0 + rng.normal(0.0, 3e-4, time.size)
    red = white + power_law_noise(time.size, 1 / 48, alpha=2.0, rms=6e-4, rng=rng)

    def beta(flux):
        lc = LightCurve("N", time, flux, np.full(time.size, 3e-4), 0, {})
        return extract_features(flatten(lc, config.preprocess), fast_bls)["red_noise_beta"]

    assert beta(white) == pytest.approx(1.0, abs=0.45)
    assert beta(red) > beta(white)


def _raw_binary_sigmas(flat, bls_config):
    """The odd/even and secondary significances against white-noise error bars,
    plus the two things that scale them: the noise beta (primary and secondary
    windows masked) and the event-to-event depth scatter ratios.  The secondary
    is net of a planet's occultation, as in ``extract_features``.

    Recomputed here, independently of ``extract_features``, from the same BLS
    solution, so the tests can compare the shipped (scaled) values to the
    unscaled ones.  Returns ``(odd_even, secondary, beta, primary_ratio,
    secondary_ratio)``.
    """
    res = run_bls(flat, bls_config)
    st = res["bls"].compute_stats(res["period"], res["duration"], res["transit_time"])
    odd_even = _pair_sigma(st["depth_odd"], st["depth_even"])
    sec_value, sec_err = (float(v) for v in np.ravel(st["depth_phased"])[:2])
    excess = secondary_excess(sec_value, res["depth"], res["period"])
    secondary = excess / sec_err if sec_err > 0 else float("nan")
    period, duration, epoch = res["period"], res["duration"], res["transit_time"]
    phase = (flat.time - epoch + 0.5 * period) % period - 0.5 * period
    mask = (np.abs(phase) < duration) | (np.abs(np.abs(phase) - 0.5 * period) < duration)
    beta = red_noise_beta(flat, duration, mask)
    primary_ratio = depth_scatter_ratio(
        *event_depths(flat, period, duration, epoch), by_parity=True
    )
    secondary_ratio = depth_scatter_ratio(
        *event_depths(flat, period, duration, epoch + 0.5 * period), by_parity=False
    )
    return odd_even, secondary, beta, primary_ratio, secondary_ratio


def test_feature_beta_is_inflated_by_transit_leakage_but_vetting_beta_is_not(config, fast_bls):
    """Why the binary tests use their own, wider-masked beta.

    With the exact-box mask of the ``red_noise_beta`` feature, ingress/egress
    cadences leak into the out-of-transit bins and a white-noise planet light
    curve reads beta well above 1.  Masking two box-widths brings it back to ~1.
    """
    lc, _ = clean_transit_curve(period=3.4, depth=3e-3, duration=0.16, sigma=2e-4, seed=11)
    flat = flatten(lc, config.preprocess)
    features = extract_features(flat, fast_bls)
    _, _, vetting_beta, _, _ = _raw_binary_sigmas(flat, fast_bls)
    assert features["red_noise_beta"] > 1.5
    assert vetting_beta == pytest.approx(1.0, abs=0.2)


def test_beta_inflation_is_floored_at_one_and_ignores_nan():
    assert beta_inflation(0.7) == 1.0
    assert beta_inflation(1.0) == 1.0
    assert beta_inflation(2.3) == pytest.approx(2.3)
    assert beta_inflation(float("nan")) == 1.0


def test_binary_tests_are_scaled_by_the_larger_of_beta_and_event_scatter(config, fast_bls):
    """Each binary test is divided by max(beta, its event scatter ratio, 1)."""
    lc, _ = clean_transit_curve(period=3.4, depth=3e-3, duration=0.12, sigma=2e-4, seed=11)
    flat = flatten(lc, config.preprocess)
    features = extract_features(flat, fast_bls)
    odd_even_raw, secondary_raw, beta, primary_ratio, secondary_ratio = _raw_binary_sigmas(
        flat, fast_bls
    )

    odd_even_scale = max(beta_inflation(beta), beta_inflation(primary_ratio))
    secondary_scale = max(beta_inflation(beta), beta_inflation(secondary_ratio))
    assert features["odd_even_sigma"] == pytest.approx(odd_even_raw / odd_even_scale, rel=1e-9)
    assert features["secondary_sigma"] == pytest.approx(
        secondary_raw / secondary_scale, rel=1e-9
    )


def test_binary_tests_are_nearly_unchanged_on_white_noise(config, fast_bls):
    """On white noise beta and both scatter ratios sit near 1, so the scaling is small.

    The ratios are sample statistics over a handful of events, so this is
    checked over many noise draws rather than one: their mean square is 1 when
    the events agree to within their errors.
    """
    betas, primary, secondary = [], [], []
    for seed in range(12):
        lc, _ = clean_transit_curve(period=3.4, depth=3e-3, duration=0.12, sigma=2e-4, seed=seed)
        _, _, beta, primary_ratio, secondary_ratio = _raw_binary_sigmas(
            flatten(lc, config.preprocess), fast_bls
        )
        betas.append(beta)
        primary.append(primary_ratio)
        secondary.append(secondary_ratio)

    assert np.median(betas) == pytest.approx(1.0, abs=0.2)
    # The primary events scatter a little more than white noise: the trend
    # under each transit is interpolated, and its error is the event's own.
    assert 0.7 < np.mean(np.square(primary)) < 1.6
    assert 0.6 < np.mean(np.square(secondary)) < 1.4
    assert np.median([max(b, p, 1.0) for b, p in zip(betas, primary)]) < 1.3


def test_binary_tests_are_deflated_by_beta_under_red_noise(config, fast_bls):
    """Correlated noise inflates the raw binary statistics; dividing by beta undoes it.

    This is failure mode 4 in the README: a genuine planet on a red-noise star
    picking up a spurious odd/even difference and secondary eclipse.
    """
    from transitml.data.base import LightCurve
    from transitml.data.synthetic import power_law_noise

    lc, _ = clean_transit_curve(period=3.4, depth=3e-3, duration=0.12, sigma=2e-4, seed=11)
    rng = np.random.default_rng(5)
    red = power_law_noise(lc.time.size, 1 / 48, alpha=1.5, rms=5e-4, rng=rng)
    noisy = LightCurve(lc.target_id, lc.time, lc.flux + red, lc.flux_err, 1, dict(lc.meta))
    flat = flatten(noisy, config.preprocess)
    features = extract_features(flat, fast_bls)
    odd_even_raw, secondary_raw, beta, primary_ratio, secondary_ratio = _raw_binary_sigmas(
        flat, fast_bls
    )

    assert beta > 1.3, f"injected red noise should give beta > 1.3, got {beta:.2f}"
    odd_even_scale = max(beta, beta_inflation(primary_ratio))
    secondary_scale = max(beta, beta_inflation(secondary_ratio))
    assert features["odd_even_sigma"] == pytest.approx(odd_even_raw / odd_even_scale, rel=1e-9)
    assert features["secondary_sigma"] == pytest.approx(
        secondary_raw / secondary_scale, rel=1e-9
    )
    assert features["odd_even_sigma"] < odd_even_raw
    assert abs(features["secondary_sigma"]) < abs(secondary_raw)


def test_depth_scatter_ratio_is_one_for_consistent_events():
    """Depths that agree to within their errors give a ratio near 1.

    An alternating depth is not scatter when odd and even events are compared
    with their own means, and one missing event is a lot of it.
    """
    rng = np.random.default_rng(2)
    epochs = np.arange(400)
    errors = np.full(epochs.size, 1e-4)
    depths = 5e-3 + rng.normal(0.0, 1e-4, epochs.size)
    assert depth_scatter_ratio(epochs, depths, errors, by_parity=False) == pytest.approx(
        1.0, abs=0.08
    )

    alternating = depths + np.where(epochs % 2 == 0, 1e-3, -1e-3)
    assert depth_scatter_ratio(epochs, alternating, errors, by_parity=True) == pytest.approx(
        1.0, abs=0.08
    )
    assert depth_scatter_ratio(epochs, alternating, errors, by_parity=False) > 9.0

    erased = depths.copy()
    erased[7] = 0.0
    assert depth_scatter_ratio(epochs, erased, errors, by_parity=True) > 2.0


def test_depth_scatter_ratio_needs_two_degrees_of_freedom():
    epochs, depths, errors = np.array([0, 1, 2]), np.array([1.0, 2.0, 1.5]), np.ones(3)
    # Odd and even means use up two of three events: one degree of freedom.
    assert np.isnan(depth_scatter_ratio(epochs, depths, errors, by_parity=True))
    assert np.isfinite(depth_scatter_ratio(epochs, depths, errors, by_parity=False))


def test_event_depths_measure_each_event_on_its_own():
    """A noiseless box transit whose depth changes every event, read back exactly."""
    time = np.arange(0.0, 27.4, 1 / 48)
    period, epoch, duration = 3.1, 1.0, 0.12
    number = np.round((time - epoch) / period).astype(int)
    inside = np.abs(time - epoch - number * period) < duration / 2
    truth = 4e-3 + 1e-4 * np.arange(number.max() + 1)
    flux = 1.0 - inside * truth[number]
    flat = flatten(LightCurve("box", time, flux, np.full(time.size, 1e-4), 1, {}))

    epochs, depths, errors = event_depths(flat, period, duration, epoch)
    np.testing.assert_array_equal(epochs, np.unique(number[inside]))
    np.testing.assert_allclose(depths, truth[epochs], atol=2e-6)
    counts = np.array([np.sum(inside & (number == k)) for k in epochs])
    np.testing.assert_allclose(errors, 1e-4 / np.sqrt(counts), rtol=1e-9)


def _wandering_depths(alternation: float, seed: int = 4) -> LightCurve:
    """A deep transit whose depth wanders 5% from event to event, as a
    pulsating star or residual systematics make it, optionally also
    alternating between odd and even events by ``alternation``."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, 27.4, 1 / 48)
    period, epoch, depth, duration = 2.3, 0.9, 8e-3, 0.12
    number = np.round((time - epoch) / period).astype(int)
    inside = np.abs(time - epoch - number * period) < duration / 2
    wander = 1.0 + 0.05 * rng.normal(size=number.max() + 1)
    parity = np.where(number % 2 == 0, 1.0 + alternation / 2, 1.0 - alternation / 2)
    flux = 1.0 - inside * depth * wander[number] * parity + rng.normal(0.0, 1e-4, time.size)
    return LightCurve("wander", time, flux, np.full(time.size, 1e-4), 1, {})


def test_wandering_depths_are_not_a_binary_but_alternating_ones_are(config, fast_bls):
    """The high-SNR failure on real TOIs, and the binary signature it must not hide.

    Against white-noise errors, 5% event-to-event depth scatter on an 8000 ppm
    transit is a many-sigma odd/even difference.  Scaled by the scatter ratio
    it is not; a 30% alternation on top of the same scatter still is.
    """
    flat_planet = flatten(_wandering_depths(0.0), config.preprocess)
    flat_binary = flatten(_wandering_depths(0.3), config.preprocess)
    raw_planet, _, _, ratio_planet, _ = _raw_binary_sigmas(flat_planet, fast_bls)
    _, _, _, ratio_binary, _ = _raw_binary_sigmas(flat_binary, fast_bls)
    planet = extract_features(flat_planet, fast_bls)
    binary = extract_features(flat_binary, fast_bls)

    assert raw_planet > 5.0, "what the unscaled statistic used to report"
    assert ratio_planet > 3.0 and ratio_binary > 3.0
    assert planet["odd_even_sigma"] < 3.0
    assert binary["odd_even_sigma"] > 5.0


def test_flatten_masked_keeps_a_transit_against_a_gap(config, fast_bls):
    """End to end: the masked second pass finds the signal and keeps every event."""
    from .test_preprocess import transit_against_a_gap

    lc, period, epoch, duration, depth = transit_against_a_gap()
    blind = flatten(lc, config.preprocess)
    masked = flatten_masked(lc, config.preprocess, fast_bls)
    assert masked.n_masked > 0

    _, blind_depths, _ = event_depths(blind, period, duration, epoch)
    _, masked_depths, _ = event_depths(masked, period, duration, epoch)
    assert blind_depths.min() < 0.2 * depth
    np.testing.assert_allclose(masked_depths, depth, rtol=0.1)


def test_flatten_masked_falls_back_to_the_blind_detrend(config, fast_bls):
    from dataclasses import replace

    lc, _ = clean_transit_curve(period=3.4, depth=3e-3, duration=0.12, sigma=2e-4, seed=1)
    blind = flatten(lc, config.preprocess)
    assert flatten_masked(lc, config.preprocess, fast_bls).n_masked > 0
    for preprocess in (
        replace(config.preprocess, mask_signal=False),
        replace(config.preprocess, mask_min_sde=1e9),
        replace(config.preprocess, mask_max_fraction=0.0),
    ):
        same = flatten_masked(lc, preprocess, fast_bls)
        assert same.n_masked == 0
        np.testing.assert_array_equal(same.flux, blind.flux)


def test_a_noise_peak_is_not_masked(config, fast_bls):
    """Masking a peak the search does not believe in would only feed it."""
    quiet, _ = clean_transit_curve(depth=0.0, variability_amplitude=2e-3, seed=8)
    res = run_bls(flatten(quiet, config.preprocess), fast_bls)
    assert signal_detection_efficiency(res["power"]) < config.preprocess.mask_min_sde
    assert flatten_masked(quiet, config.preprocess, fast_bls).n_masked == 0


def test_period_grid_is_log_spaced_and_bounded(config):
    grid = period_grid(27.4, config.bls)
    assert grid[0] == pytest.approx(config.bls.min_period_days)
    assert grid[-1] == pytest.approx(27.4 * config.bls.max_period_fraction_of_baseline)
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-9)  # geometric, not arithmetic


def test_sde_is_robust_to_the_peak_itself():
    """Using np.std would let the peak suppress its own significance."""
    power = np.concatenate([np.random.default_rng(0).normal(1.0, 0.05, 999), [8.0]])
    assert signal_detection_efficiency(power) > 50
