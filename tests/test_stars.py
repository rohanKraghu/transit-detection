"""Host-star parameters: where they come from, and the occultation allowance they size."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from transitml.data.base import stellar_parameters
from transitml.physics import (
    OCCULTATION_LIMIT_FRACTION,
    RHO_SUN_CGS,
    T_SUN_K,
    density_from_gravity,
    main_sequence_density,
    main_sequence_teff,
    max_occultation_fraction,
)
from transitml.preprocess import flatten

from .conftest import clean_transit_curve, make_source


def test_density_relations():
    assert density_from_gravity(4.438, 1.0) == pytest.approx(RHO_SUN_CGS, rel=0.01)
    for rho in (0.3, 1.0, 5.0):
        assert main_sequence_density(main_sequence_teff(rho)) == pytest.approx(rho)
    assert main_sequence_density(T_SUN_K) == pytest.approx(RHO_SUN_CGS)


def test_the_allowance_grows_with_a_hotter_star_and_a_closer_orbit():
    sun = max_occultation_fraction(1.0, T_SUN_K, RHO_SUN_CGS)
    assert 0.02 < sun < 0.05
    assert max_occultation_fraction(3.0, T_SUN_K, RHO_SUN_CGS) < sun / 5
    assert max_occultation_fraction(2.0, 7500.0, 0.4) > max_occultation_fraction(2.0, T_SUN_K, 0.4)
    # A lower density puts the same period's orbit closer to the star.
    assert max_occultation_fraction(2.0, T_SUN_K, 0.3) > max_occultation_fraction(2.0, T_SUN_K, 1.4)


def test_the_allowance_is_capped_and_needs_a_known_star():
    assert max_occultation_fraction(0.5, 10000.0, 0.05) == OCCULTATION_LIMIT_FRACTION
    assert math.isnan(max_occultation_fraction(2.0, math.nan, 1.0))
    assert math.isnan(max_occultation_fraction(2.0, 6000.0, math.nan))


@pytest.mark.parametrize(
    ("meta", "teff", "density"),
    [
        ({"teff_k": 6000.0, "rho_star_cgs": 0.8}, 6000.0, 0.8),
        # Density from mass and radius, then from gravity and radius.
        ({"teff_k": 6000.0, "m_star_msun": 1.0, "r_star_rsun": 2.0}, 6000.0, RHO_SUN_CGS / 8),
        ({"teff_k": 5772.0, "logg_cgs": 4.438, "r_star_rsun": 1.0}, 5772.0, RHO_SUN_CGS),
        # One known, the other from the main sequence.
        ({"teff_k": T_SUN_K}, T_SUN_K, RHO_SUN_CGS),
        ({"rho_star_cgs": RHO_SUN_CGS}, T_SUN_K, RHO_SUN_CGS),
        # Unknown, missing or unusable values.
        ({}, math.nan, math.nan),
        ({"teff_k": None, "rho_star_cgs": "nan", "r_star_rsun": 0.0}, math.nan, math.nan),
    ],
)
def test_stellar_parameters_from_metadata(meta, teff, density):
    got_teff, got_density = stellar_parameters(meta)
    np.testing.assert_allclose([got_teff, got_density], [teff, density], rtol=0.01)


def test_the_detrended_curve_keeps_its_star(config):
    lc, _ = clean_transit_curve(seed=1)
    flat = flatten(replace(lc, meta={"teff_k": 6500.0, "rho_star_cgs": 0.6}), config.preprocess)
    assert (flat.teff_k, flat.density_cgs) == (6500.0, 0.6)
    unknown = flatten(lc, config.preprocess)
    assert math.isnan(unknown.teff_k) and math.isnan(unknown.density_cgs)


def test_synthetic_curves_carry_their_star(config):
    for lc in make_source(config, 6, 0.5, 0.2, seed=8):
        teff, density = lc.star
        assert density == lc.meta["rho_star_cgs"]
        assert teff == pytest.approx(main_sequence_teff(density))

