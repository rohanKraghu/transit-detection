"""The synthetic pixel scenes put the right light in the right pixels."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.data.synthetic_tpf import (
    PixelStar,
    blend_scenario,
    box_in_eclipse,
    gaussian_psf_image,
    synthetic_tpf,
    tess_flux,
)


def test_psf_is_unit_flux_and_centred():
    image = gaussian_psf_image((15, 15), column=7.3, row=6.6, sigma=0.8)
    assert image.sum() == pytest.approx(1.0, abs=1e-6)
    rows, cols = np.indices(image.shape)
    assert (image * cols).sum() == pytest.approx(7.3, abs=1e-6)
    assert (image * rows).sum() == pytest.approx(6.6, abs=1e-6)


def test_a_scene_without_noise_sources_has_the_right_total_flux():
    star = PixelStar(5.0, 5.0, tess_flux(8.0))
    tpf = synthetic_tpf(
        [star], period=2.0, epoch=0.5, duration=0.1, sky_e_per_s=0.0, seed=1
    )
    totals = np.nansum(tpf.flux, axis=(1, 2))
    assert np.median(totals) == pytest.approx(star.flux, rel=1e-3)
    assert tpf.target_position == (5.0, 5.0)
    assert tpf.aperture[5, 5] and not tpf.aperture[0, 0]


@pytest.mark.parametrize("kind", ["on_target", "blend"])
def test_aperture_depth_is_planet_like_in_both_scenarios(kind):
    tpf, truth = blend_scenario(kind, seed=2)
    lc = tpf.to_light_curve()
    inside = box_in_eclipse(lc.time, truth["period"], truth["epoch"], truth["duration"])
    depth = 1.0 - np.mean(lc.flux[inside]) / np.mean(lc.flux[~inside])
    assert 0.002 < depth < 0.01  # shallow enough to pass for a planet


def test_blend_dip_comes_from_the_neighbours_pixels():
    tpf, truth = blend_scenario("blend", jitter_pixels=0.0, seed=3)
    inside = box_in_eclipse(
        tpf.time, truth["period"], truth["epoch"], truth["duration"]
    )
    difference = tpf.flux[~inside].mean(axis=0) - tpf.flux[inside].mean(axis=0)
    row, col = np.unravel_index(np.argmax(difference), difference.shape)
    n_col, n_row = truth["neighbour_position"]
    assert abs(col - n_col) <= 0.5 and abs(row - n_row) <= 0.5


def test_no_event_scene_is_flat():
    tpf, truth = blend_scenario("none", seed=4)
    assert all(s["eclipse_depth"] == 0.0 for s in tpf.meta["stars"])


def test_jitter_moves_every_star_together():
    stars = [PixelStar(4.0, 5.0, 1e4), PixelStar(6.0, 5.0, 1e4)]
    still = synthetic_tpf(
        stars, period=2, epoch=0.5, duration=0.1, read_noise_e=0, seed=5
    )
    shaky = synthetic_tpf(
        stars,
        period=2,
        epoch=0.5,
        duration=0.1,
        read_noise_e=0,
        jitter_pixels=0.1,
        seed=5,
    )
    cols = np.arange(11.0)

    def centroid(t):
        return (t.flux.sum(axis=1) * cols).sum(axis=1) / t.flux.sum(axis=(1, 2))

    assert np.std(centroid(shaky)) > 5 * np.std(centroid(still))
