"""BLS must recover what was injected, and the vetting statistics must fire."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.config import PreprocessConfig
from transitml.data.synthetic import trapezoid_transit
from transitml.features import (
    FEATURE_NAMES,
    _pair_sigma,
    beta_inflation,
    extract_features,
    flat_bottom_fraction,
    period_grid,
    red_noise_beta,
    run_bls,
    signal_detection_efficiency,
)
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
    plus the noise beta (primary and secondary windows masked) used to scale them.

    Recomputed here, independently of ``extract_features``, from the same BLS
    solution, so the tests can compare the shipped (beta-scaled) values to the
    unscaled ones.
    """
    res = run_bls(flat, bls_config)
    st = res["bls"].compute_stats(res["period"], res["duration"], res["transit_time"])
    odd_even = _pair_sigma(st["depth_odd"], st["depth_even"])
    sec_value, sec_err = (float(v) for v in np.ravel(st["depth_phased"])[:2])
    secondary = sec_value / sec_err if sec_err > 0 else float("nan")
    period, duration = res["period"], res["duration"]
    phase = (flat.time - res["transit_time"] + 0.5 * period) % period - 0.5 * period
    mask = (np.abs(phase) < duration) | (np.abs(np.abs(phase) - 0.5 * period) < duration)
    beta = red_noise_beta(flat, duration, mask)
    return odd_even, secondary, beta


def test_feature_beta_is_inflated_by_transit_leakage_but_vetting_beta_is_not(config, fast_bls):
    """Why the binary tests use their own, wider-masked beta.

    With the exact-box mask of the ``red_noise_beta`` feature, ingress/egress
    cadences leak into the out-of-transit bins and a white-noise planet light
    curve reads beta well above 1.  Masking two box-widths brings it back to ~1.
    """
    lc, _ = clean_transit_curve(period=3.4, depth=3e-3, duration=0.16, sigma=2e-4, seed=11)
    flat = flatten(lc, config.preprocess)
    features = extract_features(flat, fast_bls)
    _, _, vetting_beta = _raw_binary_sigmas(flat, fast_bls)
    assert features["red_noise_beta"] > 1.5
    assert vetting_beta == pytest.approx(1.0, abs=0.2)


def test_beta_inflation_is_floored_at_one_and_ignores_nan():
    assert beta_inflation(0.7) == 1.0
    assert beta_inflation(1.0) == 1.0
    assert beta_inflation(2.3) == pytest.approx(2.3)
    assert beta_inflation(float("nan")) == 1.0


def test_binary_tests_are_unchanged_on_white_noise(config, fast_bls):
    """With white noise beta ~ 1, so the beta scaling must be (nearly) a no-op."""
    lc, _ = clean_transit_curve(period=3.4, depth=3e-3, duration=0.12, sigma=2e-4, seed=11)
    flat = flatten(lc, config.preprocess)
    features = extract_features(flat, fast_bls)
    odd_even_raw, secondary_raw, beta = _raw_binary_sigmas(flat, fast_bls)

    assert beta == pytest.approx(1.0, abs=0.2)
    inflation = beta_inflation(beta)
    assert inflation < 1.2
    assert features["odd_even_sigma"] == pytest.approx(odd_even_raw / inflation, rel=1e-9)
    assert features["secondary_sigma"] == pytest.approx(secondary_raw / inflation, rel=1e-9)
    # i.e. within the white-noise sampling scatter of beta, unchanged.
    assert features["odd_even_sigma"] == pytest.approx(odd_even_raw, rel=0.2)
    assert abs(features["secondary_sigma"]) == pytest.approx(abs(secondary_raw), rel=0.2)


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
    odd_even_raw, secondary_raw, beta = _raw_binary_sigmas(flat, fast_bls)

    assert beta > 1.3, f"injected red noise should give beta > 1.3, got {beta:.2f}"
    assert features["odd_even_sigma"] == pytest.approx(odd_even_raw / beta, rel=1e-9)
    assert features["secondary_sigma"] == pytest.approx(secondary_raw / beta, rel=1e-9)
    assert features["odd_even_sigma"] < odd_even_raw
    assert abs(features["secondary_sigma"]) < abs(secondary_raw)


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
