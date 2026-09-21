"""Detrending must remove the star and keep the planet.

These are the tests that matter most.  Everything downstream measures a signal
that preprocessing either preserved or destroyed, and a detrender that quietly
eats 40% of the depth still produces a plausible-looking pipeline with a
quietly worse answer.  That specific failure is not hypothetical -- it is what
the first version of this module did, and
``test_running_median_eats_the_transit_on_a_steep_star`` is the test that
caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

from transitml.config import PreprocessConfig
from transitml.data.base import LightCurve
from transitml.preprocess import (
    flatten,
    fit_trend,
    harmonic_basis,
    point_to_point_sigma,
    robust_least_squares,
    robust_sigma,
    running_median_trend,
    spline_basis,
    split_on_gaps,
)

from .conftest import clean_transit_curve


def measured_depth(time, flux, period, epoch, duration):
    """Out-of-transit level minus in-transit level, measured from the data alone.

    Only the inner half of the transit is used, so that ingress and egress
    cadences -- which are genuinely shallower -- do not bias the number.
    """
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    inside = np.abs(phase) < 0.25 * duration
    outside = np.abs(phase) > 1.5 * duration
    return float(np.mean(flux[outside]) - np.mean(flux[inside]))


def depth_uncertainty(time, flux, period, epoch, duration):
    """1-sigma uncertainty on :func:`measured_depth`.

    The estimator averages a handful of in-transit cadences, so for a shallow
    transit its own noise is percent-level.  Tests compare against this rather
    than a round number, otherwise they are asserting on noise.
    """
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    inside = np.abs(phase) < 0.25 * duration
    outside = np.abs(phase) > 1.5 * duration
    sigma = robust_sigma(flux[outside])
    return float(sigma * np.sqrt(1.0 / max(inside.sum(), 1) + 1.0 / max(outside.sum(), 1)))


def depth_ratio(lc, truth, config=None):
    """Return (recovered / injected depth, that ratio's own 1-sigma error)."""
    flat = flatten(lc, config or PreprocessConfig())
    args = (flat.time, flat.flux, truth["period"], truth["epoch"], truth["duration"])
    return (
        measured_depth(*args) / truth["depth"],
        depth_uncertainty(*args) / truth["depth"],
    )


@pytest.mark.parametrize("variability_amplitude", [0.0, 5e-4, 2e-3, 4e-3, 8e-3])
@pytest.mark.parametrize("variability_period", [0.5, 1.2, 6.0, 20.0])
def test_detrending_preserves_transit_depth(variability_amplitude, variability_period):
    """The recovered depth must stay within 5% of the injected depth.

    This is the headline claim of the preprocessing stage, and it has to hold
    across the whole variability range -- including half-day rotators, where the
    spline alone cannot follow the star and the Fourier term does the work, and
    including 0.8% amplitudes, where the variability is nearly three times the
    transit depth.
    """
    lc, truth = clean_transit_curve(
        depth=3e-3,
        duration=0.14,
        sigma=1.5e-4,
        variability_amplitude=variability_amplitude,
        variability_period=variability_period,
        seed=17,
    )
    ratio, error = depth_ratio(lc, truth)
    tolerance = max(0.05, 3.0 * error)
    assert ratio == pytest.approx(1.0, abs=tolerance), (
        f"depth changed by {100 * (ratio - 1):+.1f}% (+/- {100 * error:.1f}% "
        f"measurement error) for A={variability_amplitude}, P_var={variability_period}"
    )


@pytest.mark.parametrize("depth", [5e-4, 1e-3, 5e-3, 1.2e-2])
def test_depth_preserved_across_transit_depths(depth):
    """Shallow and deep transits alike; the bias must not grow with depth.

    The tolerance widens for the shallowest case because the *measurement* gets
    noisier, not because the detrender gets worse: a 500 ppm transit over nine
    cadences at 120 ppm leaves the estimator with several percent of its own
    scatter.  Comparing against that scatter is the honest test.
    """
    lc, truth = clean_transit_curve(
        depth=depth, duration=0.18, sigma=1.2e-4,
        variability_amplitude=3e-3, variability_period=2.0, seed=23,
    )
    ratio, error = depth_ratio(lc, truth)
    assert ratio == pytest.approx(1.0, abs=max(0.06, 3.0 * error))


def test_detrending_actually_removes_the_variability():
    """Not just "preserves the transit" -- it has to remove the star too."""
    lc, truth = clean_transit_curve(
        depth=2e-3, variability_amplitude=6e-3, variability_period=1.1, seed=5
    )
    raw_scatter = robust_sigma(lc.flux)
    flat = flatten(lc, PreprocessConfig())

    assert flat.scatter < 0.2 * raw_scatter
    # ...and down to the white-noise floor it was built with.
    assert flat.scatter < 1.5 * truth["sigma"]
    assert flat.rotation_periods, "a 1.1 d rotator should have been modelled"
    # The reported period may be a low-order multiple of the true one: a model
    # at 2P with twice the harmonics contains the true frequency as its second
    # harmonic and removes the star just as completely.  What matters for the
    # transit search is that nothing coherent is left behind, which the scatter
    # assertions above already establish.
    ratios = [flat.rotation_periods[0] / (1.1 * m) for m in (1, 2, 3)]
    assert min(abs(r - 1.0) for r in ratios) < 0.1, flat.rotation_periods


def test_running_median_eats_the_transit_on_a_steep_star():
    """Why the pipeline does not use a running median.

    The median of a window over a monotone series is that window's centre
    value.  On a star varying fast enough that the trend moves by more than the
    photometric noise across one window, the running median therefore
    reproduces the data -- transit included -- and removing it removes the
    transit.  The robust fit has no such failure mode.

    Both numbers are asserted, so this is a regression test on the decision and
    not just a comment.
    """
    lc, truth = clean_transit_curve(
        depth=3e-3, duration=0.14, sigma=1.5e-4,
        variability_amplitude=8e-3, variability_period=3.0, seed=17,
    )
    config = PreprocessConfig()
    signal = lc.flux / np.median(lc.flux) - 1.0

    median_trend = running_median_trend(lc.time, signal, config.knot_spacing_days)
    median_depth = measured_depth(
        lc.time, signal - median_trend, truth["period"], truth["epoch"], truth["duration"]
    )
    robust_trend, _ = fit_trend(lc.time, signal, config)
    robust_depth = measured_depth(
        lc.time, signal - robust_trend, truth["period"], truth["epoch"], truth["duration"]
    )

    assert median_depth / truth["depth"] < 0.6, "the running median unexpectedly survived"
    assert robust_depth / truth["depth"] == pytest.approx(1.0, abs=0.05)

    # And the mechanism: on a steep stretch the median residual is *identically*
    # zero, because the filter is returning the centre value of the window.
    steep_residual = signal - median_trend
    assert np.mean(steep_residual == 0.0) > 0.1


def test_robust_fit_ignores_the_transit_by_zero_weighting_it():
    """The mechanism behind depth preservation, asserted directly."""
    lc, truth = clean_transit_curve(
        depth=4e-3, duration=0.16, sigma=1.5e-4,
        variability_amplitude=4e-3, variability_period=1.5, seed=13,
    )
    config = PreprocessConfig()
    signal = lc.flux / np.median(lc.flux) - 1.0
    design = np.hstack(
        [spline_basis(lc.time, config), harmonic_basis(lc.time, 1.5, 3)]
    )
    _, weights = robust_least_squares(design, signal, tuning=config.biweight_tuning)

    phase = (lc.time - truth["epoch"] + 0.5 * truth["period"]) % truth["period"] - 0.5 * truth["period"]
    in_transit = np.abs(phase) < 0.3 * truth["duration"]

    assert weights[in_transit].max() < 0.05, "in-transit cadences still influence the fit"
    assert weights[~in_transit].mean() > 0.8, "out-of-transit cadences were over-rejected"


def test_knot_spacing_that_is_too_tight_destroys_the_transit():
    """The knot-spacing floor is load-bearing, not a style choice.

    With knots narrower than the transit, the first unweighted iteration fits
    the dip, its residuals come out small, and the biweight never rejects it.
    """
    lc, truth = clean_transit_curve(depth=4e-3, duration=0.2, sigma=1e-4, seed=19)
    safe = PreprocessConfig()
    reckless = PreprocessConfig(knot_spacing_days=0.05)

    safe_ratio, _ = depth_ratio(lc, truth, safe)
    reckless_ratio, _ = depth_ratio(lc, truth, reckless)

    assert safe_ratio == pytest.approx(1.0, abs=0.05)
    assert reckless_ratio < 0.7


def test_clipping_is_upward_only():
    """Flares are removed; transit cadences are not.

    A symmetric sigma clip is the easiest way to silently destroy this problem,
    because the deepest transits are the ones it clips hardest.
    """
    lc, truth = clean_transit_curve(depth=6e-3, duration=0.15, sigma=2e-4, seed=31)

    flux = lc.flux.copy()
    for t0 in (4.0, 11.0, 19.0):
        after = lc.time >= t0
        flux[after] += 0.02 * np.exp(-(lc.time[after] - t0) / 0.03)
    flared = LightCurve(lc.target_id, lc.time, flux, lc.flux_err, lc.label, dict(lc.meta))

    flat = flatten(flared, PreprocessConfig())
    assert flat.n_clipped > 0

    phase = (lc.time - truth["epoch"] + 0.5 * truth["period"]) % truth["period"] - 0.5 * truth["period"]
    in_transit = np.abs(phase) < truth["duration"] / 2.0
    kept = np.isin(lc.time[in_transit], flat.time)
    assert kept.all(), f"{(~kept).sum()} in-transit cadences were clipped away"

    recovered = measured_depth(
        flat.time, flat.flux, truth["period"], truth["epoch"], truth["duration"]
    )
    assert recovered == pytest.approx(truth["depth"], rel=0.06)


def test_quiet_stars_are_not_given_a_spurious_rotation_term():
    """A transit is periodic too; the rotation model must not be aimed at it."""
    lc, _ = clean_transit_curve(
        period=1.6, depth=8e-3, duration=0.1, sigma=1.5e-4,
        variability_amplitude=0.0, seed=41,
    )
    flat = flatten(lc, PreprocessConfig())
    assert flat.rotation_periods == ()


def test_trend_is_recoverable_additively():
    """``flux + trend - 1`` must reproduce the normalised input exactly."""
    lc, _ = clean_transit_curve(variability_amplitude=3e-3, seed=2)
    flat = flatten(lc, PreprocessConfig())
    kept = np.isin(lc.time, flat.time)
    np.testing.assert_allclose(
        flat.flux + flat.trend - 1.0,
        lc.flux[kept] / np.median(lc.flux),
        rtol=0,
        atol=1e-12,
    )


def test_gap_splitting_and_block_diagonal_basis():
    time = np.concatenate([np.arange(0, 5, 0.02), np.arange(7, 12, 0.02)])
    segments = split_on_gaps(time, gap_threshold_days=0.25)
    assert len(segments) == 2
    assert segments[0].size + segments[1].size == time.size
    assert time[segments[1]][0] == pytest.approx(7.0)

    basis = spline_basis(time, PreprocessConfig())
    left, right = segments
    # No basis function is supported on both sides of the gap.
    assert not np.any(
        (np.abs(basis[left]).sum(axis=0) > 0) & (np.abs(basis[right]).sum(axis=0) > 0)
    )


def test_trend_does_not_leak_across_a_gap():
    """A step at the downlink gap must not bleed backwards into the first half."""
    config = PreprocessConfig()
    left = np.arange(0.0, 6.0, 1 / 48)
    right = np.arange(8.0, 14.0, 1 / 48)
    time = np.concatenate([left, right])
    rng = np.random.default_rng(0)
    signal = np.concatenate([np.zeros(left.size), np.full(right.size, 0.01)])
    signal = signal + rng.normal(0, 1e-4, time.size)

    trend, _ = fit_trend(time, signal, config)
    assert abs(float(np.median(trend[: left.size]))) < 5e-4
    assert float(np.median(trend[left.size :])) > 0.009


def test_point_to_point_sigma_is_insensitive_to_smooth_variability():
    """The noise estimate must not be inflated by the signal we are removing.

    Differencing suppresses a smooth trend by roughly ``omega * dt``, so the
    estimator is not perfect for the very fastest variables -- the tolerance
    below is the honest one, not a round number.
    """
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 27.0, 1 / 48)
    noise = rng.normal(0.0, 3e-4, size=time.size)
    quiet = point_to_point_sigma(noise)
    variable = point_to_point_sigma(noise + 5e-3 * np.sin(2 * np.pi * time / 1.5))

    assert quiet == pytest.approx(3e-4, rel=0.1)
    assert variable == pytest.approx(quiet, rel=0.4)
    # The naive estimator, by contrast, is inflated more than tenfold.
    naive = robust_sigma(noise + 5e-3 * np.sin(2 * np.pi * time / 1.5))
    assert naive > 10 * quiet


def test_flatten_rejects_degenerate_input():
    with pytest.raises(ValueError):
        flatten(
            LightCurve("TOO-SHORT", np.arange(5.0), np.ones(5), np.full(5, 1e-4))
        )
    with pytest.raises(ValueError, match="median flux is zero"):
        time = np.arange(0.0, 5.0, 1 / 48)
        flatten(
            LightCurve("ZERO", time, np.zeros(time.size), np.full(time.size, 1e-4))
        )
