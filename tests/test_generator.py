"""The generator must produce the class imbalance it was asked for, reproducibly."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.data.synthetic import (
    SyntheticTESSSource,
    planet_signal,
    power_law_noise,
    trapezoid_transit,
    white_noise_sigma,
)
from transitml.physics import (
    RHO_SUN_CGS,
    T_SUN_K,
    main_sequence_teff,
    max_occultation_fraction,
    occultation_depth,
    scaled_semi_major_axis,
    tess_brightness_ratio,
    transit_durations,
)

from .conftest import make_source


@pytest.mark.parametrize(
    ("n_curves", "positive_rate", "eb_rate"),
    [(500, 0.02, 0.05), (400, 0.04, 0.06), (300, 0.05, 0.10)],
)
def test_requested_imbalance_is_exact(config, n_curves, positive_rate, eb_rate):
    """The realised positive rate must equal the requested one, not approximate it.

    Drawing each label from a Bernoulli would make the positive rate a random
    variable; at 4% and 2400 curves that is a +/- 0.8% swing, enough to move
    average precision between runs for reasons that have nothing to do with the
    model.  The generator fixes the counts and shuffles.
    """
    source = make_source(config, n_curves, positive_rate, eb_rate, seed=11)
    kinds = source.kinds

    assert len(kinds) == n_curves
    assert kinds.count("planet") == round(n_curves * positive_rate)
    assert kinds.count("eclipsing_binary") == round(n_curves * eb_rate)
    assert kinds.count("noise") == n_curves - kinds.count("planet") - kinds.count(
        "eclipsing_binary"
    )

    realised = kinds.count("planet") / n_curves
    assert realised == pytest.approx(positive_rate, abs=1.0 / n_curves)


def test_labels_match_kinds(config):
    """Only planets are positives; eclipsing binaries are hard negatives."""
    source = make_source(config, 60, 0.1, 0.2, seed=3)
    for lc in source:
        expected = 1 if lc.meta["kind"] == "planet" else 0
        assert lc.label == expected
    # And the eclipsing binaries really are in there, labelled 0.
    kinds = source.kinds
    assert kinds.count("eclipsing_binary") == 12
    assert all(
        source.generate(i).label == 0
        for i, k in enumerate(kinds)
        if k == "eclipsing_binary"
    )


def test_generation_is_reproducible(config):
    """Same seed and index, same light curve -- bit for bit."""
    a = make_source(config, 50, 0.1, 0.1, seed=5)
    b = make_source(config, 50, 0.1, 0.1, seed=5)
    for index in (0, 7, 23):
        first, second = a.generate(index), b.generate(index)
        np.testing.assert_array_equal(first.time, second.time)
        np.testing.assert_array_equal(first.flux, second.flux)
        assert first.meta["kind"] == second.meta["kind"]

    different = make_source(config, 50, 0.1, 0.1, seed=6)
    assert not np.array_equal(a.generate(0).flux, different.generate(0).flux)


def test_curves_are_finite_and_well_formed(config):
    source = make_source(config, 40, 0.15, 0.15, seed=9)
    for lc in source:
        assert lc.n_cadences > 1000
        assert np.all(np.isfinite(lc.flux))
        assert np.all(np.isfinite(lc.time))
        assert np.all(lc.flux_err > 0)
        assert np.all(np.diff(lc.time) > 0)
        # Normalised to a relative-flux scale.
        assert abs(float(np.median(lc.flux)) - 1.0) < 0.05


def test_time_grid_has_the_downlink_gap(config):
    """A TESS sector is not gap-free, and the detrender has to cope."""
    lc = make_source(config, 10, 0.0, 0.0, seed=1).generate(0)
    gaps = np.diff(lc.time)
    assert gaps.max() > 0.5 * config.survey.downlink_gap_days
    assert lc.baseline_days == pytest.approx(config.survey.baseline_days, rel=0.05)


def test_injected_planet_parameters_are_physical(config):
    """Depth, duration and period must be mutually consistent, not independent draws.

    If depth and duration were drawn independently, a classifier could separate
    planets from binaries on an unphysical combination that would never occur in
    real data, and the reported metric would be meaningless.
    """
    source = make_source(config, 400, 1.0, 0.0, seed=2)
    for index in range(40):
        meta = source.generate(index).meta
        assert config.planet.period_range_days[0] <= meta["period"] <= config.planet.period_range_days[1]
        assert 0.0 <= meta["impact_parameter"] <= config.planet.impact_parameter_max
        assert meta["duration_t23"] <= meta["duration_t14"]

        # Depth follows (Rp/Rs)^2 with the limb-darkening boost.
        expected_depth = meta["radius_ratio"] ** 2 * config.planet.limb_darkening_boost
        assert meta["depth"] == pytest.approx(expected_depth, rel=1e-9)

        # Duration follows from Kepler's third law at the host's density.
        t14, _ = transit_durations(
            meta["period"],
            scaled_semi_major_axis(meta["period"], meta["rho_star_cgs"]),
            meta["radius_ratio"],
            meta["impact_parameter"],
        )
        assert meta["duration_t14"] == pytest.approx(t14, rel=1e-9)
        # Sanity: a transit is a small fraction of an orbit.
        assert 0.0 < meta["duration_t14"] / meta["period"] < 0.25


def test_grazing_geometry_produces_v_shapes():
    """b > 1 - k must close the flat bottom -- that is what "V-shaped" means."""
    a_rs = scaled_semi_major_axis(4.0, RHO_SUN_CGS)
    central_t14, central_t23 = transit_durations(4.0, a_rs, 0.1, 0.0)
    grazing_t14, grazing_t23 = transit_durations(4.0, a_rs, 0.1, 0.95)
    assert central_t23 > 0
    assert grazing_t23 == 0.0
    assert grazing_t14 < central_t14


def test_occultation_physics():
    assert main_sequence_teff(RHO_SUN_CGS) == pytest.approx(T_SUN_K)
    assert tess_brightness_ratio(4000.0, 4000.0) == pytest.approx(1.0)
    assert tess_brightness_ratio(2000.0, 6000.0) < 0.01

    def depth(a_rs, t_eff=6000.0):
        return occultation_depth(
            0.1, a_rs, t_eff, geometric_albedo=0.1, bond_albedo=0.15, redistribution=0.5
        )

    # Closer in and around a hotter star, the dayside is hotter and brighter.
    assert depth(3.0) > 5 * depth(8.0)
    assert depth(4.0, t_eff=7000.0) > depth(4.0, t_eff=5000.0)
    # Far out, only reflected light is left: A_g (Rp / a)^2.
    assert depth(60.0) == pytest.approx(0.1 * (0.1 / 60.0) ** 2, rel=0.01)


def test_planet_occultations_stay_inside_the_secondary_test_allowance(config):
    """Every planet's occultation is one the secondary-eclipse test forgives.

    Compared against k^2 rather than the limb-darkened depth, because the
    box-fit depth the test scales its allowance by can fall a little short of
    the true one.
    """
    time = np.arange(0.0, 27.4, 1 / 48)
    rng = np.random.default_rng(0)
    fractions = []
    for r_star in np.linspace(*config.star.radius_range_rsun, 6):
        rho = RHO_SUN_CGS * r_star**0.9 / r_star**3
        for _ in range(60):
            _, meta = planet_signal(time, rng, rho, config.planet)
            allowed = max_occultation_fraction(meta["period"]) * meta["radius_ratio"] ** 2
            assert 0.0 < meta["secondary_depth"] < allowed
            fractions.append(meta["secondary_depth"] / meta["depth"])
    # Close-in giants around the hottest hosts reach the percent level, as real ones do.
    assert max(fractions) > 0.01


def test_eclipsing_binaries_carry_their_discriminants(config):
    """Secondaries and odd/even offsets must actually be injected sometimes."""
    source = make_source(config, 200, 0.0, 1.0, seed=4)
    metas = [source.generate(i).meta for i in range(60)]
    assert any(m["secondary_depth"] > 0 for m in metas)
    assert any(m["odd_even_fraction"] > 0.05 for m in metas)
    assert any(m["grazing"] for m in metas)
    assert any(not m["grazing"] for m in metas)
    # Depths overlap the planet range at the shallow end: the problem would be
    # trivial if every binary were deeper than every planet.
    depths = np.array([m["depth"] for m in metas])
    assert depths.min() < config.planet.radius_ratio_range[1] ** 2


def test_trapezoid_shape_and_odd_even():
    time = np.linspace(0.0, 10.0, 4001)
    box = trapezoid_transit(time, 2.0, 0.5, 0.01, 0.2, 0.16)
    vee = trapezoid_transit(time, 2.0, 0.5, 0.01, 0.2, 0.0)
    assert box.max() == pytest.approx(0.01, rel=1e-6)
    assert vee.max() == pytest.approx(0.01, rel=1e-2)
    # The V shape spends less total flux in the eclipse than the flat-bottomed one.
    assert vee.sum() < box.sum()

    modulated = trapezoid_transit(time, 2.0, 0.5, 0.01, 0.2, 0.16, odd_even_fraction=0.2)
    depths = []
    for n in range(0, 4):
        centre = 0.5 + n * 2.0
        near = np.abs(time - centre) < 0.02
        if near.any():
            depths.append(modulated[near].max())
    assert max(depths) > min(depths) * 1.3  # odd and even eclipses differ

    secondary = trapezoid_transit(time, 2.0, 0.5, 0.01, 0.2, 0.16, secondary_depth=0.004)
    at_half_phase = np.abs(time - (0.5 + 1.0)) < 0.02
    assert secondary[at_half_phase].max() == pytest.approx(0.004, rel=0.05)


def test_power_law_noise_has_requested_rms_and_slope():
    rng = np.random.default_rng(0)
    n, dt = 8192, 1.0 / 48.0
    red = power_law_noise(n, dt, alpha=2.0, rms=1e-3, rng=rng)
    assert float(np.std(red)) == pytest.approx(1e-3, rel=1e-6)
    assert abs(float(np.mean(red))) < 1e-3

    # Steeper alpha puts the power at low frequency, so successive samples are
    # strongly correlated -- which is the whole point of injecting it.
    white = power_law_noise(n, dt, alpha=0.0, rms=1e-3, rng=rng)

    def lag1(x):
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    assert lag1(red) > 0.95
    assert abs(lag1(white)) < 0.1


def test_white_noise_follows_the_magnitude_relation(config):
    bright = white_noise_sigma(8.0, config.noise)
    faint = white_noise_sigma(14.0, config.noise)
    assert bright == pytest.approx(60e-6, rel=1e-9)
    assert faint > 10 * bright  # faint stars are far noisier


def test_true_snr_is_recorded_and_scales_with_depth(config):
    """The evaluation slices recall by injected SNR, so it has to be right."""
    source = make_source(config, 200, 1.0, 0.0, seed=8)
    records = [source.generate(i).meta for i in range(50)]
    snr = np.array([m["true_snr"] for m in records])
    ratio = np.array([m["depth"] / m["sigma_white"] for m in records])
    assert np.all(snr > 0)
    # SNR and depth/noise must be strongly rank-correlated.
    order = np.argsort(ratio)
    assert np.corrcoef(np.argsort(order), np.argsort(np.argsort(snr)))[0, 1] > 0.8

    for meta in records:
        assert meta["n_in_transit_cadences"] > 0
        assert meta["n_transits_in_window"] >= 1


def test_negatives_have_no_injected_eclipse(config):
    source = make_source(config, 60, 0.0, 0.0, seed=12)
    for lc in source:
        assert lc.meta["kind"] == "noise"
        assert lc.meta["true_snr"] == 0.0
        assert "depth" not in lc.meta


def test_rates_are_validated():
    with pytest.raises(ValueError):
        SyntheticTESSSource(10, 0.8, 0.5, seed=0)
    with pytest.raises(ValueError):
        SyntheticTESSSource(10, -0.1, 0.0, seed=0)
