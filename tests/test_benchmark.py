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
