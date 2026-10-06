"""Host-star parameters: where they come from, and the occultation allowance they size."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from transitml.data.base import stellar_parameters
from transitml.data.tic import (
    STAR_FIELDS,
    load_or_fetch_stars,
    read_star_table,
    with_star,
    write_star_table,
)
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


def test_star_tables_round_trip_with_blanks(tmp_path):
    stars = {
        42: {
            "teff_k": 6100.0,
            "logg_cgs": 4.3,
            "r_star_rsun": 1.2,
            "m_star_msun": 1.1,
            "rho_star_cgs": 0.9,
        },
        7: {name: math.nan for name in STAR_FIELDS},
    }
    path = tmp_path / "stars.csv"
    write_star_table(path, stars)
    assert path.read_text().splitlines()[1] == "7,,,,,"
    back = read_star_table(path)
    assert back[42] == pytest.approx(stars[42])
    assert all(math.isnan(v) for v in back[7].values())


def test_only_stars_the_table_lacks_are_looked_up(tmp_path):
    path = tmp_path / "stars.csv"
    asked: list[list[int]] = []

    def fake_tic(ids):
        asked.append(list(ids))
        return {tic: {"teff_k": 5000.0 + tic, "rho_star_cgs": 1.0} for tic in ids if tic != 3}

    first = load_or_fetch_stars([1, 2, 3], path, fetch=fake_tic)
    assert asked == [[1, 2, 3]]
    assert first[2]["teff_k"] == 5002.0 and math.isnan(first[3]["teff_k"])
    # Star 3 is not in the TIC: remembered as blank rather than asked for again.
    second = load_or_fetch_stars([2, 3, 4], path, fetch=fake_tic)
    assert asked[-1] == [4]
    assert sorted(second) == [2, 3, 4]
    assert sorted(read_star_table(path)) == [1, 2, 3, 4]


def test_a_failed_lookup_writes_nothing(tmp_path):
    path = tmp_path / "stars.csv"

    def offline(_ids):
        raise ConnectionError("no network")

    with pytest.raises(ConnectionError):
        load_or_fetch_stars([1], path, fetch=offline)
    assert not path.exists()


def test_with_star_fills_only_known_values():
    lc, _ = clean_transit_curve(seed=2)
    lc = replace(lc, target_id="TIC 42", meta={"sector": 14})
    stars = {
        42: {
            "teff_k": 6100.0,
            "logg_cgs": math.nan,
            "r_star_rsun": 1.2,
            "m_star_msun": math.nan,
            "rho_star_cgs": 0.9,
        }
    }
    tagged = with_star(lc, stars)
    assert tagged.meta == {"sector": 14, "teff_k": 6100.0, "r_star_rsun": 1.2, "rho_star_cgs": 0.9}
    assert tagged.star == (6100.0, 0.9)
    assert with_star(lc, {}) is lc
