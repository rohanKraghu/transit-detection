"""The transit fit as ``vet --fit``, ``batch --fit N`` and the coverage study use it."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

from transitml import batch, vet
from transitml.batch import run_batch, synthetic_sector
from transitml.fit import quick_config
from transitml.fit_coverage import run_injection, summarise
from transitml.model import load_model, save_model

from .conftest import clean_transit_curve

QUICK = quick_config(min_steps=600, max_steps=600, min_burn=200)


@pytest.fixture(scope="module")
def model_path(tiny_model_config, tiny_trained, tmp_path_factory):
    split, trained = tiny_trained
    return save_model(
        trained, split, tmp_path_factory.mktemp("model") / "model.joblib",
        preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls,
    )


@pytest.fixture(scope="module")
def planet():
    lc, _ = clean_transit_curve(period=3.0, depth=4e-3, sigma=4e-4, seed=7)
    return replace(lc, target_id="PLANET-1", meta={**lc.meta, "rho_star_cgs": 1.4, "r_star_rsun": 1.0})


def test_vet_fit_adds_a_section_a_figure_and_printed_lines(model_path, planet, tmp_path, capsys):
    from transitml.data.injection import save_curves

    save_curves([planet], tmp_path / "planet.npz")
    out = tmp_path / "reports"
    assert vet.main([
        str(tmp_path / "planet.npz"), "--model", str(model_path), "--out-dir", str(out),
        "--fit", "--fit-max-steps", "600",
    ]) == 0
    payload = json.loads((out / "vet_PLANET_1.json").read_text())
    fit = payload["fit"]
    assert fit["parameters"]["period"]["median"] == pytest.approx(3.0, rel=1e-3)
    # The box transit's depth comes back (the curve came from this package, so no smearing).
    assert fit["parameters"]["depth_ppm"]["median"] == pytest.approx(4000, rel=0.15)
    assert "integrated over 0.0-minute exposures" in fit["model"]
    assert fit["density_check"]["stellar_density"] == 1.4
    assert fit["parameters"]["rp_earth"]["median"] > 0
    assert (out / "vet_PLANET_1_fit.png").stat().st_size > 50_000
    printed = capsys.readouterr().out
    assert "fit: Rp/R* = " in printed and "the star's 1.40 g/cm^3" in printed
    # The fit is beside the score, not in it.
    plain = vet.vet_light_curve(planet, load_model(model_path))[0]
    assert payload["score"] == pytest.approx(plain.score)


def test_vet_fit_without_stellar_parameters_skips_the_density_check(model_path, planet):
    bare = replace(planet, meta={})
    result, flat = vet.vet_light_curve(bare, load_model(model_path))
    fit = vet.fit_primary(result, flat, bare, QUICK)
    assert fit is not None and fit.density_check is None
    assert "rp_earth" not in fit.parameters
    # Survey data: the exposure is the cadence.
    assert fit.exposure_minutes == pytest.approx(30.0)


def test_vet_records_a_fit_that_cannot_run(model_path, planet):
    result, flat = vet.vet_light_curve(planet, load_model(model_path))
    result.primary = {**result.primary, "duration": 1e-4}
    assert vet.fit_primary(result, flat, planet, QUICK) is None
    assert "cadences in transit" in result.to_dict()["fit"]["error"]


@pytest.fixture(scope="module")
def fitted_batch(model_path, planet, tmp_path_factory):
    curves = synthetic_sector(6, seed=7) + [planet]
    out = tmp_path_factory.mktemp("batch")
    first = run_batch(curves, model_path, out, source="t", n_jobs=1, n_reports=0,
                      progress=False, n_fits=1, fit_config=QUICK)
    again = run_batch(curves, model_path, out, source="t", n_jobs=1, n_reports=0,
                      progress=False, n_fits=1, fit_config=QUICK)
    return first, again


def test_batch_fits_only_the_top_flagged_stars(fitted_batch):
    first, _ = fitted_batch
    fitted = [r for r in first.rows if "fit" in r]
    assert len(fitted) == 1 and fitted[0]["flagged"] and fitted[0]["rank"] == min(
        r["rank"] for r in first.rows if r["flagged"]
    )
    fit = fitted[0]["fit"]
    assert fit["status"] == "ok"
    median, lower, upper = fit["parameters"]["k"]
    assert lower <= median <= upper
    summary = first.summary["fits"]
    assert summary["fitted"] == 1 and summary["computed"] == 1 and summary["from_cache"] == 0


def test_batch_fits_are_cached_and_reach_the_csv_and_dashboard(fitted_batch):
    first, again = fitted_batch
    assert again.summary["fits"]["from_cache"] == 1 and again.summary["fits"]["computed"] == 0
    row = next(r for r in again.rows if "fit" in r)
    assert row["fit"] == next(r for r in first.rows if "fit" in r)["fit"]
    with open(again.out_dir / "candidates.csv") as handle:
        table = {t["id"]: t for t in csv.DictReader(handle)}
    cells = table[row["id"]]
    assert float(cells["fit_rp_rs"]) == pytest.approx(row["fit"]["parameters"]["k"][0])
    assert float(cells["fit_rp_rs_err"]) > 0 and cells["fit_status"] == "ok"
    others = [t for i, t in table.items() if i != row["id"]]
    assert all(t["fit_rp_rs"] == "" for t in others)
    page = (again.out_dir / "dashboard.html").read_text()
    assert '"fit":{"status":"ok"' in page and "Transit fit (batman + emcee)" in page


def test_a_new_fit_setting_refits(fitted_batch, model_path, planet):
    first, _ = fitted_batch
    curves = synthetic_sector(6, seed=7) + [planet]
    longer = replace(QUICK, max_steps=800, min_steps=800)
    result = run_batch(curves, model_path, first.out_dir, source="t", n_jobs=1, n_reports=0,
                       progress=False, n_fits=1, fit_config=longer)
    assert result.summary["fits"]["computed"] == 1
    assert batch.fit_key("x", QUICK) != batch.fit_key("x", longer)


def test_the_coverage_study_scores_an_injection():
    row = run_injection(0, seed=2026, fit_config=QUICK)
    assert set(row) >= {"planet", "white", "pipeline"}
    for experiment in ("white", "pipeline"):
        if row[experiment]["recovered"]:
            q = row[experiment]["quantile"]
            assert all(0.0 <= v <= 1.0 for v in q.values())
    summary = summarise([row], "white")
    assert summary["n_injected"] == 1
    if summary["n_recovered"]:
        assert set(summary["parameters"]["k"]) == {"in68", "in95", "median_quantile"}
