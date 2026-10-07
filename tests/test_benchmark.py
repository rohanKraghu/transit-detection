"""The real-label benchmark, run offline on synthetic stand-ins for TOI hosts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from transitml.benchmark import (
    benchmark,
    build_benchmark_dataset,
    format_benchmark_report,
    plot_benchmark,
)
from transitml.config import default_config
from transitml.data.base import LightCurve
from transitml.data.loader import build_default_dataset
from transitml.data.toi import TOI, BenchmarkTarget
from transitml.model import make_split, train

from .conftest import make_source


@pytest.fixture(scope="module")
def small_config():
    base = default_config()
    return replace(
        base,
        dataset=replace(base.dataset, n_curves=160, positive_rate=0.15, eclipsing_binary_rate=0.15),
        bls=replace(base.bls, n_periods=400),
    )


@pytest.fixture(scope="module")
def trained(small_config):
    dataset = build_default_dataset(small_config, n_jobs=2)
    split = make_split(dataset, test_size=0.3, seed=small_config.seed)
    return train(split, n_folds=3, seed=small_config.seed, target_precision=0.5)


@pytest.fixture(scope="module")
def toi_hosts(small_config):
    """Planets as CP hosts and eclipsing binaries as FP hosts, as MAST would serve them.

    A different seed from training, so no star is shared.  Metadata is
    stripped to what a downloaded curve carries; the catalogue period goes on
    the TOI, where the benchmark reads it.
    """
    source = make_source(small_config, n_curves=40, positive_rate=0.5, eb_rate=0.5, seed=99)
    curves, targets = [], []
    for i, lc in enumerate(source):
        tic = 5000 + i
        label = 1 if lc.meta["kind"] == "planet" else 0
        toi = TOI(
            tic=tic,
            toi=f"{900 + i}.01",
            disposition="CP" if label else ("FP" if i % 3 else "FA"),
            period=float(lc.meta["period"]),
            depth_ppm=float(lc.meta["depth"]) * 1e6,
            snr=float(lc.meta["true_snr"]),
            sectors=(14,),
        )
        targets.append(BenchmarkTarget(tic=tic, label=label, sector=14, reference=toi, tois=(toi,)))
        curves.append(
            LightCurve(
                target_id=f"TIC {tic}",
                time=lc.time,
                flux=lc.flux,
                flux_err=lc.flux_err,
                label=label,
                meta={"kind": "real", "sector": 14},
            )
        )
    return curves, targets


@pytest.fixture(scope="module")
def result(small_config, trained, toi_hosts):
    curves, targets = toi_hosts
    # The benchmark dataset may hold fewer stars than were selected.
    dataset = build_benchmark_dataset(
        curves[:-2], preprocess=small_config.preprocess, bls=small_config.bls, n_jobs=2
    )
    return benchmark(
        dataset,
        targets,
        trained,
        sectors=[14],
        selection={"stars_in_table": 50, "selected": len(targets)},
        n_without_curve=2,
        top_k=10,
        n_bootstrap=200,
    )


def test_counts_and_labels(result, toi_hosts):
    _, targets = toi_hosts
    assert result.n_stars == len(targets) - 2
    assert result.n_planets == sum(t.label for t in targets[:-2])
    assert result.chance_average_precision == pytest.approx(result.positive_rate)
    c = result.confusion
    assert sum(c.values()) == result.n_stars
    assert c["true_positive"] + c["false_negative"] == result.n_planets


def test_threshold_is_frozen_from_training(result, trained):
    assert result.threshold == trained.threshold
    assert result.threshold_rule == trained.threshold_rule


def test_operating_point_rates(result):
    c = result.confusion
    assert result.planet_recall == pytest.approx(
        c["true_positive"] / (c["true_positive"] + c["false_negative"])
    )
    assert result.false_positive_rejection == pytest.approx(
        c["true_negative"] / (c["true_negative"] + c["false_positive"])
    )
    n_rejected = sum(r["n"] for r in result.rejection_by_disposition.values())
    assert set(result.rejection_by_disposition) == {"FA", "FP"}
    assert n_rejected == result.n_stars - result.n_planets


def test_the_model_separates_planets_from_binaries(result):
    """Half planets, half binaries: the model must clear chance clearly."""
    assert result.model.average_precision > result.chance_average_precision + 0.15
    assert result.search_recovery["planets"] >= 0.7


def test_missed_and_accepted_lists_agree_with_the_confusion(result):
    assert len(result.missed_planets) == result.confusion["false_negative"]
    assert len(result.accepted_false_positives) == result.confusion["false_positive"]
    for row in result.missed_planets:
        assert row["disposition"] == "CP" and row["model_score"] < result.threshold


def test_report_and_json(result, tmp_path):
    report = format_benchmark_report(result)
    assert "REAL-LABEL BENCHMARK" in report
    assert "known false positives rejected" in report
    assert "accuracy" not in report.lower()
    payload = json.loads(json.dumps(result.to_dict(), default=str))
    assert payload["n_stars"] == result.n_stars
    assert payload["n_without_curve"] == 2


def test_figure_is_written(result, tmp_path):
    path = plot_benchmark(result, tmp_path / "toi.png")
    assert Path(path).stat().st_size > 10_000


def test_curves_must_match_targets(small_config, trained, toi_hosts):
    curves, targets = toi_hosts
    dataset = build_benchmark_dataset(
        curves[:4], preprocess=small_config.preprocess, bls=small_config.bls, n_jobs=1
    )
    with pytest.raises(ValueError, match="match no benchmark target"):
        benchmark(dataset, targets[1:], trained, sectors=[14], selection={})
    flipped = [replace(t, label=1 - t.label) for t in targets]
    with pytest.raises(ValueError, match="disagree"):
        benchmark(dataset, flipped, trained, sectors=[14], selection={})


def test_curve_cache_fetches_only_what_it_has_not_tried(toi_hosts, tmp_path, monkeypatch):
    import transitml.benchmark as bench

    curves, targets = toi_hosts
    served = {lc.target_id: replace(lc, label=None) for lc in curves[:5]}
    asked: list[list[str]] = []

    def fake_fetch(wanted, **_):
        asked.append([t.target_id for t in wanted])
        # MAST has nothing for the sixth star.
        return [served[t.target_id] for t in wanted if t.target_id in served]

    monkeypatch.setattr(bench, "fetch_benchmark_curves", fake_fetch)
    cache = tmp_path / "toi_curves.npz"

    first = bench.load_or_fetch_curves(targets[:6], cache)
    assert len(asked) == 1 and len(asked[0]) == 6
    assert [lc.target_id for lc in first] == [t.target_id for t in targets[:5]]
    assert [lc.label for lc in first] == [t.label for t in targets[:5]]

    again = bench.load_or_fetch_curves(targets[:6], cache)
    assert len(asked) == 1, "a rerun must not touch the network"
    np.testing.assert_array_equal(again[0].flux, first[0].flux)

    # A changed disposition relabels the cached curve; a new star is fetched alone.
    relabelled = [replace(targets[0], label=1 - targets[0].label), *targets[1:7]]
    third = bench.load_or_fetch_curves(relabelled, cache)
    assert asked[-1] == [targets[6].target_id]
    assert third[0].label == 1 - targets[0].label


def test_run_pipeline_benchmark_reads_a_cache_offline(
    small_config, trained, toi_hosts, tmp_path, monkeypatch
):
    """The ``--benchmark-tois`` path end to end, from a TOI table and a curve cache."""
    import run_pipeline
    import transitml.benchmark as bench
    import transitml.data.tic as tic_module
    from transitml.data.injection import save_curves
    from transitml.data.synthetic import SyntheticTESSSource

    curves, targets = toi_hosts
    table = tmp_path / "exofop_toi.csv"
    table.write_text(
        "TIC ID,TOI,TFOPWG Disposition,Period (days),Planet SNR,Sectors\n"
        + "".join(
            f'{t.tic},{t.reference.toi},{t.reference.disposition},'
            f'{t.reference.period},{t.reference.snr},"14,15"\n'
            for t in targets
        )
        + '999,1.01,PC,3.0,10.0,"14"\n'
    )
    cache = tmp_path / "toi_curves.npz"
    save_curves(curves, cache)

    def no_network(*_, **__):
        raise AssertionError("everything is cached; MAST must not be queried")

    monkeypatch.setattr(bench, "fetch_benchmark_curves", no_network)
    looked_up: list[int] = []

    def fake_tic(ids):
        looked_up.extend(ids)
        return {tic: {"teff_k": 6000.0, "rho_star_cgs": 1.0} for tic in ids}

    monkeypatch.setattr(tic_module, "fetch_tic_stars", fake_tic)
    args = run_pipeline.parse_args(
        [
            "--benchmark-tois", str(table),
            "--benchmark-sectors", "14",
            "--benchmark-cache", str(cache),
            "--results-dir", str(tmp_path / "results"),
            "--figures-dir", str(tmp_path / "figures"),
            "--n-jobs", "2",
        ]
    )
    (tmp_path / "results").mkdir()
    source = SyntheticTESSSource(n_curves=1, positive_rate=0.0, eclipsing_binary_rate=0.0, seed=1)
    payload = run_pipeline.run_toi_benchmark(args, small_config, trained, source)

    assert payload["n_stars"] == len(targets)
    assert payload["selection"]["unlabelled"] == 1
    assert (tmp_path / "results" / "toi_benchmark.txt").exists()
    assert json.loads((tmp_path / "results" / "toi_benchmark.json").read_text())["sectors"] == [14]
    assert (tmp_path / "figures" / "05_toi_benchmark.png").exists()
    # Each host's star is looked up once and kept beside the TOI table.
    assert sorted(looked_up) == sorted(t.tic for t in targets)
    assert (tmp_path / "tic_stars.csv").exists()


def test_training_stars_are_named_for_exclusion(toi_hosts):
    import run_pipeline
    from transitml.data.injection import InjectionSource

    curves, _ = toi_hosts
    base = [replace(lc, label=None) for lc in curves[:3]]
    source = InjectionSource(base, 0.0, 0.0, seed=1)
    assert run_pipeline.training_tic_ids(source) == {5000, 5001, 5002}


def test_depth_table_covers_every_star_with_a_depth(result):
    assert sum(r["n_planets"] for r in result.by_toi_depth) == result.n_planets
    assert "by catalogued depth" in format_benchmark_report(result)


def test_every_star_is_listed_with_its_score(result):
    assert len(result.stars) == result.n_stars
    assert sum(s["kept"] for s in result.stars) == (
        result.confusion["true_positive"] + result.confusion["false_positive"]
    )
    assert {s["disposition"] for s in result.stars} <= {"CP", "FP", "FA"}
