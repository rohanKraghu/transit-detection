"""Transit fits: the geometry, the prior, the noise model and recovery of a known planet."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from transitml.fit import (
    FitConfig,
    FitError,
    TransitModel,
    _Problem,
    a_from_t14,
    a_over_rs,
    default_exposure_minutes,
    density,
    durations,
    fit_transit,
    log_density_jacobian,
    q_to_u,
    quick_config,
    time_averaging_beta,
    u_to_q,
)
from transitml.preprocess import FlattenedLightCurve

TRUTH = {"t0": 2.3, "period": 3.7, "k": 0.08, "b": 0.3, "rho": 1.4}


def _flat(time, flux, sigma, target="INJ"):
    n = time.size
    return FlattenedLightCurve(target, time, flux, np.full(n, sigma), np.ones(n), sigma, 0)


def _theta(truth, exposure_u=(0.4, 0.25)):
    a = float(a_over_rs(truth["period"], truth["rho"]))
    t14 = float(durations(truth["period"], a, truth["k"], truth["b"])[0])
    q1, q2 = u_to_q(*exposure_u)
    return np.array([truth["t0"], truth["period"], truth["k"], truth["b"], math.log10(t14), q1, q2]), t14


@pytest.fixture(scope="module")
def injected():
    """A batman transit, integrated over 30-minute exposures, in white noise."""
    rng = np.random.default_rng(3)
    time = np.arange(0.0, 27.4, 30.0 / 1440.0)
    theta, t14 = _theta(TRUTH)
    flux = TransitModel(time, 30.0 / 1440.0, 3.0 / 1440.0).flux(theta)
    sigma = 5e-4
    return _flat(time, flux + rng.normal(0.0, sigma, time.size), sigma), theta, t14


@pytest.fixture(scope="module")
def fitted(injected):
    flat, theta, t14 = injected
    config = quick_config(min_steps=2000, max_steps=2000, min_burn=500, seed=1)
    return fit_transit(
        flat, TRUTH["period"] * (1 + 3e-4), TRUTH["t0"] + 0.01, 0.8 * t14, TRUTH["k"] ** 2,
        config, stellar_density=(TRUTH["rho"], 0.14), stellar_radius=1.0,
    )


def test_limb_darkening_parametrisation_round_trips():
    q1, q2 = np.meshgrid(np.linspace(0.05, 0.95, 7), np.linspace(0.05, 0.95, 7))
    back = u_to_q(*q_to_u(q1, q2))
    np.testing.assert_allclose(back[0], q1, rtol=1e-12)
    np.testing.assert_allclose(back[1], q2, rtol=1e-12)
    u1, u2 = q_to_u(q1, q2)
    # Every point of the unit square is a physical law: positive, decreasing to the limb.
    assert np.all(u1 + u2 <= 1.0 + 1e-12) and np.all(u1 >= 0) and np.all(u1 + 2 * u2 >= 0)


def test_duration_and_scaled_distance_invert_each_other():
    rng = np.random.default_rng(0)
    period = rng.uniform(1, 12, 200)
    k = rng.uniform(0.01, 0.2, 200)
    b = rng.uniform(0, 0.95, 200)
    a = a_over_rs(period, rng.uniform(0.3, 10, 200))
    t14, t23 = durations(period, a, k, b)
    np.testing.assert_allclose(a_from_t14(period, k, b, t14), a, rtol=1e-9)
    assert np.all(t23 < t14)
    np.testing.assert_allclose(density(period, a_over_rs(period, 2.5)), 2.5, rtol=1e-12)


def test_the_density_jacobian_matches_a_finite_difference():
    period, k, b, t14 = 3.7, 0.07, 0.6, 0.12
    a = float(a_from_t14(period, k, b, t14))
    h = 1e-6
    up = np.log(density(period, a_from_t14(period, k, b, t14 * math.exp(h))))
    down = np.log(density(period, a_from_t14(period, k, b, t14 * math.exp(-h))))
    assert abs((up - down) / (2 * h)) == pytest.approx(float(log_density_jacobian(period, k, b, t14, a)), rel=1e-6)


def test_the_prior_is_flat_in_log_density():
    """Sampled in log T14, weighted by the prior: log density comes out uniform."""
    config = FitConfig()
    problem = _Problem(
        model=None, flux=np.zeros(1), sigma=np.ones(1), t0_centre=0.0, t0_halfwidth=1.0,
        period_centre=10.0, config=config,
    )
    rng = np.random.default_rng(1)
    log_t14 = rng.uniform(math.log10(0.005), math.log10(4.9), 40_000)
    k, b = 0.05, 0.3
    log_prior = np.array([
        problem.log_prior(np.array([0.0, 10.0, k, b, x, 0.5, 0.5])) for x in log_t14
    ])
    weights = np.where(np.isfinite(log_prior), np.exp(np.where(np.isfinite(log_prior), log_prior, 0.0)), 0.0)
    a = a_from_t14(10.0, k, b, 10.0**log_t14)
    log_rho = np.log10(density(10.0, a))
    hist, _ = np.histogram(log_rho, bins=8, range=config.log10_rho_range, weights=weights)
    fractions = hist / hist.sum()
    np.testing.assert_allclose(fractions, 1 / 8, atol=0.02)


def test_beta_is_one_for_white_noise_and_larger_for_red_noise():
    rng = np.random.default_rng(2)
    time = np.arange(0.0, 27.0, 30.0 / 1440.0)
    widths = np.linspace(0.25, 1.0, 6) * 0.12
    white = rng.normal(0.0, 1.0, time.size)
    assert time_averaging_beta(time, white, widths, gap_days=0.1) < 1.15
    red = white + 0.8 * np.convolve(rng.normal(0.0, 1.0, time.size), np.ones(12) / np.sqrt(12), "same")
    assert time_averaging_beta(time, red / red.std(), widths, gap_days=0.1) > 1.25


def test_the_fit_recovers_an_injected_transit(fitted, injected):
    _, _, t14 = injected
    p = fitted.parameters
    truth = {
        "period": TRUTH["period"], "k": TRUTH["k"], "t14_hours": 24 * t14,
    }
    for name, value in truth.items():
        half = (p[name]["upper"] - p[name]["lower"]) / 2
        assert abs(p[name]["median"] - value) < 3.5 * half, name
    assert p["b"]["lower95"] <= TRUTH["b"] <= p["b"]["upper95"]
    assert p["rho_star"]["lower95"] <= TRUTH["rho"] <= p["rho_star"]["upper95"]
    # t0 is reported at the transit nearest the middle of the data.
    n = round((p["t0"]["median"] - TRUTH["t0"]) / TRUTH["period"])
    t0_truth = TRUTH["t0"] + n * TRUTH["period"]
    assert abs(p["t0"]["median"] - t0_truth) < 3.5 * (p["t0"]["upper"] - p["t0"]["lower"]) / 2
    assert abs(p["t0"]["median"] - float(np.median(fitted.data["time"]))) < TRUTH["period"]
    assert p["rp_earth"]["median"] == pytest.approx(p["k"]["median"] * 109.076, rel=1e-6)
    assert fitted.noise["beta"] < 1.3
    assert fitted.noise["sigma_ppm"] == pytest.approx(500, rel=0.1)
    assert fitted.samples.shape[1] == 7 and len(fitted.samples) == 500


def test_the_density_check_passes_the_right_star_and_flags_a_wrong_one(fitted, injected):
    assert fitted.density_check["consistent"]
    flat, _, t14 = injected
    wrong = fit_transit(
        flat, TRUTH["period"], TRUTH["t0"], 0.8 * t14, TRUTH["k"] ** 2,
        quick_config(min_steps=1500, max_steps=1500, min_burn=500, seed=2),
        stellar_density=(TRUTH["rho"] * 10, TRUTH["rho"]),
    )
    assert not wrong.density_check["consistent"]
    assert wrong.density_check["ratio"]["upper"] < 0.5


def test_the_result_is_strict_json(fitted):
    payload = json.loads(json.dumps(fitted.to_dict(), allow_nan=False))
    assert set(payload["parameters"]["k"]) == {"median", "lower", "upper", "lower95", "upper95"}
    assert payload["sampler"]["n_steps"] == 2000
    assert "samples" not in payload and "data" not in payload


def test_a_signal_with_too_few_points_in_transit_is_refused():
    time = np.arange(0.0, 27.4, 30.0 / 1440.0)
    flat = _flat(time, np.ones(time.size), 1e-3)
    with pytest.raises(FitError, match="cadences in transit"):
        fit_transit(flat, 30.0, 5.0, 0.04, 1e-3, quick_config())
    with pytest.raises(FitError, match="no usable signal"):
        fit_transit(flat, float("nan"), 5.0, 0.04, 1e-3, quick_config())


def test_the_exposure_follows_where_the_curve_came_from():
    assert default_exposure_minutes({"kind": "planet"}) == 0.0
    assert default_exposure_minutes({}) is None
    assert default_exposure_minutes({"kind": "noise", "exposure_minutes": 2.0}) == 2.0


def test_the_model_integrates_over_the_exposure():
    """A 30-minute exposure smears ingress: shallower at the contact points, same area."""
    time = np.linspace(-0.15, 0.15, 3001)
    theta, _ = _theta({**TRUTH, "t0": 0.0})
    sharp = TransitModel(time, 0.0, 1.0).flux(theta)
    smeared = TransitModel(time, 30.0 / 1440.0, 1.0 / 1440.0).flux(theta)
    assert smeared.min() > sharp.min()
    assert np.trapezoid(1 - smeared, time) == pytest.approx(np.trapezoid(1 - sharp, time), rel=2e-3)
