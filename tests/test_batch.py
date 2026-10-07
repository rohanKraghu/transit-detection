"""Batch mode: a sector in, a ranked candidate list, a cache and a dashboard out."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import replace

import numpy as np
import pytest

from transitml import batch
from transitml.batch import (
    ResultCache,
    curve_fingerprint,
    fetch_sector_curves,
    read_curve_inputs,
    run_batch,
    synthetic_sector,
)
from transitml.config import MultiPlanetConfig
from transitml.data.base import LightCurve
from transitml.data.injection import save_curves
from transitml.model import load_model, save_model

from .conftest import clean_transit_curve

FAST = MultiPlanetConfig(max_signals=2)


@pytest.fixture(scope="module")
def model_path(tiny_model_config, tiny_trained, tmp_path_factory):
    split, trained = tiny_trained
    return save_model(
        trained, split, tmp_path_factory.mktemp("model") / "model.joblib",
        preprocess=tiny_model_config.preprocess, bls=tiny_model_config.bls,
    )


@pytest.fixture(scope="module")
def sector():
    """Twenty-four synthetic stars plus one hand-made planet that must rank high."""
    curves = synthetic_sector(24, seed=7)
    planet, _ = clean_transit_curve(period=3.0, depth=4e-3, sigma=4e-4, seed=7)
    return curves + [replace(planet, target_id="PLANET-1")]


@pytest.fixture(scope="module")
def first_run(sector, model_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("batch")
    return run_batch(
        sector, model_path, out, source="test sector", n_jobs=2, multi=FAST,
        n_reports=1, progress=False,
    )


def test_every_star_is_ranked_by_score_with_a_calibrated_probability(first_run, sector, model_path):
    rows = first_run.rows
    model = load_model(model_path)
    assert len(rows) == len(sector)
    ok = [r for r in rows if r["status"] == "ok"]
    assert [r["rank"] for r in ok] == list(range(1, len(ok) + 1))
    scores = [r["score"] for r in ok]
    assert scores == sorted(scores, reverse=True)
    for row in ok:
        assert row["flagged"] == (row["score"] >= model.threshold)
        assert row["p_planet"] == pytest.approx(1.0 / (1.0 + np.exp(-row["log_odds"])), rel=1e-5)
        assert len(row["reasons"]) == batch.N_BATCH_REASONS
        assert len(row["fold"]["ppt"]) == batch.FOLD_BINS
    assert next(r for r in ok if r["id"] == "PLANET-1")["rank"] <= 3


def test_the_ranked_csv_matches_the_rows(first_run):
    with open(first_run.out_dir / "candidates.csv") as handle:
        table = list(csv.DictReader(handle))
    assert [t["id"] for t in table] == [r["id"] for r in first_run.rows]
    assert table[0]["rank"] == "1"
    assert float(table[0]["p_planet"]) == pytest.approx(first_run.rows[0]["p_planet"])
    assert "truth_label" in table[0]  # synthetic stars carry ground truth
    assert re.match(r"\w+ [+-]\d+\.\d\d$", table[0]["reason_1"])


def test_summary_counts_and_scores_against_ground_truth(first_run):
    summary = json.loads((first_run.out_dir / "summary.json").read_text())
    rows = first_run.rows
    assert summary["n_stars"] == len(rows)
    assert summary["n_flagged"] == sum(r["flagged"] for r in rows)
    assert summary["computed"] == len(rows) and summary["from_cache"] == 0
    truth = summary["truth"]
    assert truth["n_planets"] == sum(r["truth"]["label"] for r in rows)
    flagged_planets = sum(r["flagged"] and r["truth"]["label"] == 1 for r in rows)
    assert truth["flagged_planets"] == flagged_planets


def test_a_rerun_takes_everything_from_the_cache(first_run, sector, model_path):
    again = run_batch(
        sector, model_path, first_run.out_dir, source="test sector", n_jobs=2, multi=FAST,
        n_reports=0, progress=False,
    )
    assert again.summary["computed"] == 0
    assert again.summary["from_cache"] == len(sector)
    assert [r["score"] for r in again.rows] == [r["score"] for r in first_run.rows]


def test_a_changed_curve_is_recomputed_and_nothing_else(first_run, sector, model_path, tmp_path):
    out = tmp_path / "copy"
    (out / "cache").mkdir(parents=True)
    (out / "cache" / "results.jsonl").write_text(
        (first_run.out_dir / "cache" / "results.jsonl").read_text()
    )
    changed = list(sector)
    lc = changed[3]
    changed[3] = replace(lc, flux=lc.flux * (1.0 + 1e-4))
    result = run_batch(
        changed, model_path, out, source="t", n_jobs=1, multi=FAST, n_reports=0, progress=False
    )
    assert result.summary["computed"] == 1


def test_a_new_model_or_new_settings_invalidates_the_cache(first_run, sector, model_path, tmp_path):
    out = tmp_path / "copy"
    (out / "cache").mkdir(parents=True)
    (out / "cache" / "results.jsonl").write_text(
        (first_run.out_dir / "cache" / "results.jsonl").read_text()
    )
    few = sector[:3]
    settings = run_batch(
        few, model_path, out, source="t", n_jobs=1, multi=MultiPlanetConfig(max_signals=1),
        n_reports=0, progress=False,
    )
    assert settings.summary["computed"] == 3
    other_model = tmp_path / "other.joblib"
    other_model.write_bytes(model_path.read_bytes() + b"\0")  # same model, different bytes
    keys = {batch.result_key(lc, batch.model_fingerprint(other_model), FAST) for lc in few}
    assert not keys & set(ResultCache(out / "cache" / "results.jsonl").rows)


def test_the_planet_rate_moves_probabilities_without_recomputing(first_run, sector, model_path):
    restated = run_batch(
        sector, model_path, first_run.out_dir, source="t", n_jobs=1, multi=FAST,
        planet_rate=0.5, n_reports=0, progress=False,
    )
    assert restated.summary["computed"] == 0
    assert restated.summary["planet_rate"] == 0.5
    before = {r["id"]: r for r in first_run.rows}
    for row in restated.rows:
        if row["status"] == "ok":
            assert row["score"] == before[row["id"]]["score"]
            assert row["p_planet"] >= before[row["id"]]["p_planet"]


def test_an_interrupted_run_keeps_what_it_finished(sector, model_path, tmp_path):
    run_batch(sector[:5], model_path, tmp_path, source="t", n_jobs=1, multi=FAST,
              n_reports=0, progress=False)
    cache = tmp_path / "cache" / "results.jsonl"
    cache.write_text(cache.read_text() + '{"key": "torn line, never fini')  # killed mid-write
    resumed = run_batch(sector[:8], model_path, tmp_path, source="t", n_jobs=1, multi=FAST,
                        n_reports=0, progress=False)
    assert resumed.summary["computed"] == 3 and resumed.summary["from_cache"] == 5


def test_a_bad_light_curve_fails_alone(model_path, tmp_path):
    good, _ = clean_transit_curve(period=3.0, depth=4e-3, sigma=4e-4, seed=3)
    tiny = LightCurve("TINY", np.arange(5.0), np.ones(5), np.full(5, 1e-3))
    result = run_batch([good, tiny], model_path, tmp_path, source="t", n_jobs=1, multi=FAST,
                       n_reports=0, progress=False)
    status = {r["id"]: r["status"] for r in result.rows}
    assert status == {"TEST-0001": "ok", "TINY": "error"}
    assert result.rows[-1]["id"] == "TINY" and result.rows[-1]["rank"] is None
    assert result.summary["n_errors"] == 1 and result.summary["errors"][0]["id"] == "TINY"


def test_reports_are_written_for_the_top_flagged_stars(first_run):
    flagged = [r for r in first_run.rows if r["flagged"]]
    assert flagged, "the hand-made planet should be flagged"
    assert flagged[0]["report"] == first_run.summary["reports"][0]
    assert (first_run.out_dir / flagged[0]["report"]).stat().st_size > 50_000


def test_dashboard_is_self_contained_and_carries_every_star(first_run):
    page = (first_run.out_dir / "dashboard.html").read_text()
    assert not re.search(r"""(src|href)=["']https?://""", page)
    assert "<link" not in page
    data = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    payload = json.loads(data.group(1).replace("<\\/", "</"))
    assert [r["id"] for r in payload["rows"]] == [r["id"] for r in first_run.rows]
    assert payload["summary"]["n_flagged"] == first_run.summary["n_flagged"]
    assert all("key" not in r for r in payload["rows"])


def test_dashboard_cannot_be_broken_out_of_by_a_target_id(model_path, tmp_path):
    lc, _ = clean_transit_curve(period=3.0, depth=4e-3, sigma=4e-4, seed=4)
    nasty = replace(lc, target_id="</script><script>alert(1)</script>")
    result = run_batch([nasty], model_path, tmp_path, source="t", n_jobs=1, multi=FAST,
                       n_reports=0, progress=False)
    page = (result.out_dir / "dashboard.html").read_text()
    assert page.count("</script>") == 2  # the data block and the code block, nothing injected


def test_duplicate_stars_from_two_sectors_get_distinct_ids(model_path, tmp_path):
    lc, _ = clean_transit_curve(period=3.0, depth=4e-3, sigma=4e-4, seed=5)
    a = replace(lc, target_id="TIC 9", meta={"sector": 14})
    b = replace(lc, target_id="TIC 9", flux=lc.flux * 1.0001, meta={"sector": 15})
    result = run_batch([a, b], model_path, tmp_path, source="t", n_jobs=1, multi=FAST,
                       n_reports=0, progress=False)
    assert sorted(r["id"] for r in result.rows) == ["TIC 9 s14", "TIC 9 s15"]
    with pytest.raises(ValueError, match="twice"):
        run_batch([a, a], model_path, tmp_path, source="t", n_jobs=1, multi=FAST,
                  n_reports=0, progress=False)


def test_curve_fingerprint_sees_any_change():
    lc, _ = clean_transit_curve(seed=1)
    assert curve_fingerprint(lc) == curve_fingerprint(replace(lc))
    flux = lc.flux.copy()
    flux[100] += 1e-9
    assert curve_fingerprint(lc) != curve_fingerprint(replace(lc, flux=flux))


def test_files_and_directories_are_read(tmp_path):
    a, _ = clean_transit_curve(seed=1)
    b, _ = clean_transit_curve(seed=2)
    save_curves([a, replace(b, target_id="OTHER")], tmp_path / "two.npz")
    (tmp_path / "dir").mkdir()
    np.savetxt(tmp_path / "dir" / "star.csv", np.column_stack([a.time, a.flux, a.flux_err]),
               delimiter=",", header="time,flux,flux_err", comments="")
    curves = read_curve_inputs([tmp_path / "two.npz", tmp_path / "dir"])
    assert [c.target_id for c in curves] == ["TEST-0001", "OTHER", "star"]
    with pytest.raises(ValueError, match="no such file"):
        read_curve_inputs([tmp_path / "missing.npz"])


def test_mast_download_is_chunked_cached_and_resumable(monkeypatch, tmp_path):
    """Offline: the MAST source is replaced; the cache logic is what is tested."""
    from transitml.data import mast

    calls: list[list[str]] = []
    lc, _ = clean_transit_curve(seed=1)

    class FakeMAST:
        def __init__(self, targets, **kwargs):
            self.targets = [t for t, _ in targets]
            calls.append(self.targets)
            assert kwargs["sector"] == 14

        def __iter__(self):
            for t in self.targets:
                if t != "TIC 3":  # MAST has nothing for this one
                    yield replace(lc, target_id=t, meta={"sector": 14})

    monkeypatch.setattr(mast, "MASTLightCurveSource", FakeMAST)
    targets = ["TIC 1", "TIC 2", "TIC 3", "TIC 4", "TIC 5"]
    got = fetch_sector_curves(targets[:3], 14, tmp_path, chunk_size=2, progress=False)
    assert [c.target_id for c in got] == ["TIC 1", "TIC 2"]
    assert calls == [["TIC 1", "TIC 2"], ["TIC 3"]]
    got = fetch_sector_curves(targets, 14, tmp_path, chunk_size=2, progress=False)
    assert [c.target_id for c in got] == ["TIC 1", "TIC 2", "TIC 4", "TIC 5"]
    assert calls[2:] == [["TIC 4", "TIC 5"]]  # TIC 3 is remembered as tried
    assert len(list(tmp_path.glob("chunk_*.npz"))) == 2


def test_a_failed_download_is_retried_on_the_next_run(monkeypatch, tmp_path):
    """A target whose download raised is not remembered as tried; one MAST lacks is."""
    from transitml.data import mast

    calls: list[list[str]] = []
    lc, _ = clean_transit_curve(seed=1)
    flaky = {"TIC 2"}

    class FakeMAST:
        def __init__(self, targets, **kwargs):
            self.targets = [t for t, _ in targets]
            self.failed: list[str] = []
            calls.append(self.targets)

        def __iter__(self):
            for t in self.targets:
                if t in flaky:
                    self.failed.append(t)
                elif t != "TIC 3":
                    yield replace(lc, target_id=t, meta={"sector": 14})

    monkeypatch.setattr(mast, "MASTLightCurveSource", FakeMAST)
    targets = ["TIC 1", "TIC 2", "TIC 3"]
    got = fetch_sector_curves(targets, 14, tmp_path, progress=False)
    assert [c.target_id for c in got] == ["TIC 1"]
    flaky.clear()
    got = fetch_sector_curves(targets, 14, tmp_path, progress=False)
    assert calls[1] == ["TIC 2"]  # TIC 3 had nothing and stays tried
    assert [c.target_id for c in got] == ["TIC 1", "TIC 2"]


def test_the_mast_source_records_failures_apart_from_empty_searches(monkeypatch):
    """Offline: a search that raises is a failure; one that finds nothing is not."""
    from transitml.data.mast import MASTLightCurveSource

    class FakeSearch(list):
        def download_all(self, **_):
            raise TimeoutError("read timed out")

    class FakeLK:
        @staticmethod
        def search_lightcurve(target, **_):
            if target == "TIC 9":
                raise ConnectionError("reset")
            return FakeSearch([1]) if target == "TIC 8" else FakeSearch()

    monkeypatch.setattr(MASTLightCurveSource, "_import_lightkurve", staticmethod(lambda: FakeLK))
    source = MASTLightCurveSource([("TIC 7", None), ("TIC 8", None), ("TIC 9", None)])
    with pytest.warns(UserWarning) as caught:
        assert list(source) == []
    assert [str(w.message) for w in caught] == [
        "TIC 8: skipped (TimeoutError: read timed out)",
        "TIC 9: skipped (ConnectionError: reset)",
    ]
    assert source.failed == ["TIC 8", "TIC 9"]


def test_cli_runs_a_synthetic_sector(model_path, tmp_path, capsys):
    status = batch.main([
        "--synthetic", "12", "--seed", "3", "--model", str(model_path), "--out-dir", str(tmp_path),
        "--n-jobs", "1", "--reports", "0", "--max-signals", "1",
    ])
    assert status == 0
    printed = capsys.readouterr().out
    assert "12 vetted" in printed and "against ground truth" in printed
    assert (tmp_path / "dashboard.html").exists() and (tmp_path / "candidates.csv").exists()


def test_cli_wants_exactly_one_source(model_path):
    with pytest.raises(SystemExit):
        batch.main(["--synthetic", "3", "--targets", "t.txt", "--model", str(model_path)])
    with pytest.raises(SystemExit):
        batch.main(["--model", str(model_path)])
