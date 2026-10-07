"""Sector systematics: shared by every star in a sector, as real ones are."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from transitml.config import SurveyConfig, SystematicsConfig, default_config
from transitml.data.base import LightCurve
from transitml.data.synthetic import (
    SYSTEMATIC_COMPONENTS,
    SectorSystematics,
    SyntheticTESSSource,
)
from transitml.features import run_bls
from transitml.preprocess import flatten

ON = SystematicsConfig(enabled=True)


def _source(n=60, seed=5, config=ON):
    return SyntheticTESSSource(n, 0.0, 0.0, seed=seed, systematics=config)


def test_systematics_are_off_by_default():
    plain = SyntheticTESSSource(30, 0.1, 0.1, seed=3)
    explicit = SyntheticTESSSource(30, 0.1, 0.1, seed=3, systematics=SystematicsConfig())
    assert plain.sector is None
    for i in range(0, 30, 7):
        a, b = plain.generate(i), explicit.generate(i)
        np.testing.assert_array_equal(a.time, b.time)
        np.testing.assert_array_equal(a.flux, b.flux)
        assert "camera" not in a.meta


def test_every_star_shares_the_gap_and_loses_the_dump_cadences():
    source = _source()
    sector = source.sector
    flagged = sector.time[sector.flagged]
    half = source.survey.downlink_gap_days / 2.0
    for i in range(10):
        time = source.generate(i).time
        assert not np.any(np.abs(time - sector.gap_centre) <= half)
        assert not np.any(np.isin(np.round(time, 9), np.round(flagged, 9)))

    def gap_start(lc):
        return round(float(lc.time[np.argmax(np.diff(lc.time))]), 6)

    # Without systematics the gap moves from star to star; with them it is the sector's.
    # (A dropped cadence at the edge can move a star's gap by one cadence.)
    plain = SyntheticTESSSource(10, 0.0, 0.0, seed=5)
    cadence = source.survey.cadence_days
    assert np.ptp([gap_start(plain.generate(i)) for i in range(10)]) > 5 * cadence
    assert np.ptp([gap_start(source.generate(i)) for i in range(10)]) <= 2 * cadence


def test_a_star_is_its_camera_series_times_its_couplings():
    sector = _source().sector
    time = sector.time[~sector.flagged]
    sigma = 4e-4
    signal, meta = sector.for_star(time, sigma, np.random.default_rng(0))
    series = sector.components(meta["camera"] - 1)
    index = np.rint(time / sector.cadence_days).astype(int)
    rebuilt = (
        meta["scattered_light_coupling"] * series["scattered_light"][index]
        + meta["jitter_coupling"] * series["jitter"][index]
        - meta["momentum_dump_depth"] * series["momentum_dumps"][index]
        - meta["focus_coupling"] * series["focus"][index]
    )
    np.testing.assert_allclose(signal, rebuilt, rtol=0, atol=1e-15)
    lo, hi = ON.momentum_dump_sigma_range
    assert lo * sigma <= meta["momentum_dump_depth"] <= hi * sigma
    assert meta["focus_coupling"] >= 0


def test_cameras_share_dumps_and_light_shape_but_not_jitter_or_focus():
    sector = _source().sector
    cams = [sector.components(c) for c in range(ON.n_cameras)]
    for c in range(1, ON.n_cameras):
        assert cams[c]["momentum_dumps"] is cams[0]["momentum_dumps"]
        ratio = cams[c]["scattered_light"] / cams[0]["scattered_light"]
        np.testing.assert_allclose(
            ratio, ON.camera_scattered_light[c] / ON.camera_scattered_light[0]
        )
        assert abs(np.corrcoef(cams[c]["jitter"], cams[0]["jitter"])[0, 1]) < 0.5
        assert not np.allclose(cams[c]["focus"], cams[0]["focus"])


def test_scattered_light_rises_into_each_perigee_on_the_orbit():
    sector = _source().sector
    light = sector.components(0)["scattered_light"]
    t = sector.time
    for p in sector.perigees:
        near = (np.abs(t - p) < 0.6) & (np.abs(t - p) > 0.0)
        mid = np.abs(t - (p + ON.orbit_days / 2.0)) < 0.6
        if near.any() and mid.any():
            assert light[near].mean() > 5 * light[mid].mean()
    np.testing.assert_allclose(np.diff(sector.perigees), ON.orbit_days)
    assert sector.perigees[2] == pytest.approx(sector.gap_centre)


def test_momentum_dumps_cost_flux_at_the_same_times_in_every_star():
    source = _source()
    sector = source.sector
    dumps = sector.components(0)["momentum_dumps"]
    for t0 in sector.dump_times:
        after = (sector.time > t0) & (sector.time < t0 + 2 * sector.cadence_days)
        assert dumps[after].max() > 0.3
    spacing = np.diff(sector.dump_times)
    np.testing.assert_allclose(spacing, sector.dump_interval)
    lo, hi = ON.momentum_dump_interval_days_range
    assert lo <= sector.dump_interval <= hi
    intervals = {source.generate(i).meta["momentum_dump_interval"] for i in range(5)}
    assert intervals == {sector.dump_interval}


def test_the_sector_is_set_by_the_seed():
    a, b, c = _source(seed=1).sector, _source(seed=1).sector, _source(seed=2).sector
    assert a.gap_centre == b.gap_centre and a.dump_interval == b.dump_interval
    np.testing.assert_array_equal(a.jitter[0], b.jitter[0])
    assert (a.gap_centre, a.dump_interval) != (c.gap_centre, c.dump_interval)
    again = _source(seed=1).generate(3).flux
    np.testing.assert_array_equal(_source(seed=1).generate(3).flux, again)


def test_each_camera_needs_a_scattered_light_level():
    with pytest.raises(ValueError, match="every camera"):
        SectorSystematics(SurveyConfig(), replace(ON, n_cameras=5), seed=0)


def test_strong_dumps_alone_make_a_periodic_transit_like_signal(fast_bls):
    """The reason this matters: a per-star search finds the dump interval."""
    config = replace(
        ON,
        scattered_light_sigma_range=(1e-6, 1e-6),
        jitter_sigma_range=(0.0, 0.0),
        focus_sigma_range=(0.0, 0.0),
        momentum_dump_sigma_range=(6.0, 6.0),
    )
    sector = SectorSystematics(SurveyConfig(), config, seed=11)
    time = sector.time[~sector.flagged]
    sigma = 5e-4
    signal, _ = sector.for_star(time, sigma, np.random.default_rng(0))
    flux = 1.0 + signal + np.random.default_rng(1).normal(0.0, sigma, time.size)
    lc = LightCurve("SYS-DUMPS", time, flux, np.full(time.size, sigma))
    found = run_bls(flatten(lc), fast_bls)["period"]
    ratio = found / sector.dump_interval
    assert min(abs(ratio - k) for k in (0.5, 1.0, 2.0)) < 0.02


def test_components_can_be_switched_off_without_moving_the_rest():
    """Scale 0 is a paired control: same stars, same cadences, no systematics."""

    def flux(**changes):
        return _source(n=12, config=replace(ON, **changes)).generate(7).flux

    control = flux(scale=0.0)
    full = flux()
    parts = [flux(components=(name,)) - control for name in SYSTEMATIC_COMPONENTS]
    np.testing.assert_allclose(full - control, np.sum(parts, axis=0), rtol=0, atol=1e-12)
    assert all(np.ptp(part) > 0 for part in parts)
    np.testing.assert_allclose(flux(scale=2.0) - control, 2 * (full - control), atol=1e-12)
    with pytest.raises(ValueError, match="unknown"):
        _source(config=replace(ON, components=("moon",)))


def test_the_pipeline_flags_keep_the_headline_apart():
    import run_pipeline

    args = run_pipeline.parse_args(
        ["--systematics", "--systematics-components", "momentum_dumps,focus",
         "--systematics-scale", "0.5"]
    )
    assert args.results_dir == run_pipeline.ROOT / "results" / "systematics"
    assert args.figures_dir == run_pipeline.ROOT / "figures" / "systematics"
    config = run_pipeline.apply_overrides(default_config(), args)
    assert config.systematics.enabled and config.systematics.scale == 0.5
    assert config.systematics.components == ("momentum_dumps", "focus")
    plain = run_pipeline.apply_overrides(default_config(), run_pipeline.parse_args([]))
    assert not plain.systematics.enabled
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--systematics", "--inject-into", "targets.txt"])


def test_the_sector_figure_is_written(tmp_path):
    from transitml.plots import plot_sector_systematics

    path = plot_sector_systematics(_source().sector, tmp_path / "sector.png")
    assert path.exists() and path.stat().st_size > 10_000
