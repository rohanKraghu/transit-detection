"""Injection into real photometry: labels known by construction, noise left alone."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from transitml.data.base import LightCurve, LightCurveSource
from transitml.data.injection import (
    InjectionSource,
    exclude_known_hosts,
    load_curves,
    load_excluded_tic_ids,
    read_target_list,
    robust_white_sigma,
    save_curves,
    tic_number,
)
from transitml.data.loader import build_dataset
from transitml.evaluate import period_recovered

from .conftest import make_source


@pytest.fixture(scope="module")
def base_curves(config):
    """Stand-ins for real quiet stars: variable stars with no eclipse, label 0.

    Their metadata is stripped to what a MAST curve would carry, so nothing
    from the generator's ground truth can leak into the injection.
    """
    source = make_source(config, n_curves=50, positive_rate=0.0, eb_rate=0.0, seed=11)
    curves = []
    for i, lc in enumerate(source):
        curves.append(
            LightCurve(
                target_id=f"TIC {1000 + i}",
                time=lc.time,
                flux=lc.flux,
                flux_err=lc.flux_err,
                label=None,
                meta={"kind": "real", "sector": 14},
            )
        )
    return curves


def test_class_counts_are_exact_and_reproducible(base_curves):
    a = InjectionSource(base_curves, 0.2, 0.1, seed=3)
    b = InjectionSource(base_curves, 0.2, 0.1, seed=3)
    assert a.kinds.count("planet") == 10
    assert a.kinds.count("eclipsing_binary") == 5
    assert a.kinds == b.kinds
    assert np.array_equal(a.generate(7).flux, b.generate(7).flux)


def test_labels_follow_the_injected_kind(base_curves):
    source = InjectionSource(base_curves, 0.2, 0.1, seed=3)
    for lc, kind in zip(source, source.kinds):
        assert lc.meta["kind"] == kind
        assert lc.label == (1 if kind == "planet" else 0)
        assert lc.meta["base_target_id"].startswith("TIC ")


def test_uninjected_curves_are_returned_untouched(base_curves):
    """The whole point: the noise is the real noise, not a model of it."""
    source = InjectionSource(base_curves, 0.2, 0.1, seed=3)
    for i, kind in enumerate(source.kinds):
        if kind == "noise":
            assert np.array_equal(source.generate(i).flux, base_curves[i].flux)


def test_injection_is_multiplicative_and_only_dims(base_curves):
    source = InjectionSource(base_curves, 0.2, 0.1, seed=3)
    i = source.kinds.index("planet")
    lc, base = source.generate(i), base_curves[i]
    ratio = lc.flux / base.flux
    assert np.all(ratio <= 1.0 + 1e-12)
    assert 1.0 - ratio.min() == pytest.approx(lc.meta["depth"], rel=1e-6)


def test_known_planet_hosts_are_refused(base_curves):
    poisoned = list(base_curves)
    poisoned[3] = replace(poisoned[3], label=1)
    with pytest.raises(ValueError, match="exclude known hosts"):
        InjectionSource(poisoned, 0.1, 0.1, seed=1)


def test_white_sigma_ignores_slow_variability_and_the_dip():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 27, 1300)
    flux = 1 + 0.01 * np.sin(2 * np.pi * t / 3) + rng.normal(0, 5e-4, t.size)
    flux[600:610] -= 0.01
    assert robust_white_sigma(flux) == pytest.approx(5e-4, rel=0.1)


def test_injected_planets_are_found_by_the_pipeline(base_curves, config):
    """High-SNR injections into the stand-in curves come back at the right period."""
    planet = replace(config.planet, radius_ratio_range=(0.08, 0.1), period_range_days=(1.5, 5.0))
    source = InjectionSource(base_curves[:12], 0.5, 0.0, seed=5, planet=planet)
    dataset = build_dataset(source, preprocess=config.preprocess, bls=config.bls, n_jobs=1)
    assert int(dataset.y.sum()) == 6
    planets = dataset.meta[dataset.meta["kind"] == "planet"]
    found = 10.0 ** dataset.features.loc[planets.index, "log_period"].to_numpy(dtype=float)
    true = planets["period"].to_numpy(dtype=float)
    assert period_recovered(found, true).mean() >= 5 / 6


def test_target_lists_and_toi_exclusion(tmp_path):
    plain = tmp_path / "targets.txt"
    plain.write_text("# sector 14 quiet stars\nTIC 100\n200\nTIC 100\n")
    assert read_target_list(plain) == ["TIC 100", "TIC 200"]

    as_csv = tmp_path / "targets.csv"
    as_csv.write_text("TIC ID,Tmag\n300,9.1\n400,10.2\n")
    assert read_target_list(as_csv) == ["TIC 300", "TIC 400"]

    spoc = tmp_path / "s0014.csv"  # MAST TESS-SPOC target list layout
    spoc.write_text("#TIC_ID,RA,DEC\n0000000007547522,272.19,47.85\n")
    assert read_target_list(spoc) == ["TIC 7547522"]

    noted = tmp_path / "sample.txt"  # a comment with a comma is not a CSV header
    noted.write_text("# 2,800 stars, seeded\nTIC 500\n600\n")
    assert read_target_list(noted) == ["TIC 500", "TIC 600"]

    toi = tmp_path / "toi.csv"
    toi.write_text("# ExoFOP export\nTIC ID,TOI,Disposition\n200,101.01,PC\n999,102.01,KP\n")
    excluded = load_excluded_tic_ids(toi)
    assert excluded == {200, 999}
    kept, dropped = exclude_known_hosts(["TIC 100", "TIC 200"], excluded)
    assert kept == ["TIC 100"] and dropped == ["TIC 200"]
    assert tic_number("TIC 307210830") == 307210830


def test_curves_round_trip_through_npz(tmp_path, base_curves):
    path = tmp_path / "curves.npz"
    save_curves(base_curves[:3], path)
    back = load_curves(path)
    assert [c.target_id for c in back] == [c.target_id for c in base_curves[:3]]
    assert all(np.array_equal(a.flux, b.flux) for a, b in zip(back, base_curves))
    assert back[0].label is None and back[0].meta["sector"] == 14


def test_run_pipeline_real_mode_runs_offline_from_a_curve_cache(
    base_curves, config, tmp_path, monkeypatch
):
    """The documented real-data command, minus the download, end to end."""
    import run_pipeline

    small = replace(
        config,
        dataset=replace(config.dataset, positive_rate=0.2, eclipsing_binary_rate=0.1),
        bls=replace(config.bls, n_periods=300),
    )
    monkeypatch.setattr(run_pipeline, "default_config", lambda: small)
    cache = tmp_path / "base.npz"
    save_curves(base_curves, cache)
    targets = tmp_path / "targets.txt"
    targets.write_text("TIC 1000\n")

    code = run_pipeline.main(
        [
            "--inject-into",
            str(targets),
            "--curve-cache",
            str(cache),
            "--results-dir",
            str(tmp_path / "out"),
            "--n-jobs",
            "1",
            "--no-figures",
        ]
    )
    assert code == 0
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["dataset"]["source"] == "InjectionSource"
    assert metrics["dataset"]["n_curves"] == len(base_curves)
    assert metrics["dataset"]["n_planets"] == 10


def test_figure_caption_handles_a_real_curve_with_nothing_injected(base_curves):
    from transitml.plots import _curve_caption

    assert _curve_caption(base_curves[0]) == "real TESS star, nothing injected"


def test_curve_labels_reach_the_dataset(base_curves, config):
    """A labelled real curve (no ``kind == "planet"`` in its meta) keeps its label."""
    labelled = [replace(lc, label=i % 2) for i, lc in enumerate(base_curves[:4])]

    class Listed(LightCurveSource):
        def __len__(self):
            return len(labelled)

        def __iter__(self):
            return iter(labelled)

    dataset = build_dataset(Listed(), preprocess=config.preprocess, bls=config.bls, n_jobs=1)
    assert dataset.y.tolist() == [0, 1, 0, 1]
