"""Ranking the open TOIs: who is selected, how they are ranked, what is written."""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest

from transitml.data.loader import Dataset
from transitml.data.toi import TOI, BenchmarkTarget, select_open_targets
from transitml.features import FEATURE_NAMES
from transitml.open_tois import (
    OPEN_COLUMNS,
    archive_match,
    format_open_report,
    rank_open_hosts,
    read_archive_planets,
    write_open_ranking,
)
from transitml.toi_training import train_on_hosts


def _toi(tic, toi, disposition, sectors, snr=10.0, **kw):
    return TOI(tic=tic, toi=toi, disposition=disposition, snr=snr, sectors=tuple(sectors), **kw)


def test_open_stars_have_an_open_toi_and_no_resolved_planet():
    tois = [
        _toi(1, "100.01", "PC", [3, 30]),
        _toi(2, "101.01", "CP", [3]),
        _toi(2, "101.02", "PC", [3]),  # a planet's star is resolved
        _toi(3, "102.01", "FP", [5]),
        _toi(3, "102.02", "APC", [5], snr=20.0),  # an FP beside an open TOI stays open
        _toi(3, "102.03", "PC", [5], snr=12.0),
        _toi(4, "103.01", "FP", [7]),  # every TOI retired: a negative, not open
        _toi(5, "104.01", "", [7]),  # blank only: nothing to rank
        _toi(6, "105.01", "PC", [40]),  # open, but not observed in the sectors
        _toi(7, "106.01", "PC", [9, 2]),
    ]
    targets, counts = select_open_targets(tois, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert [t.tic for t in targets] == [1, 3, 7]
    assert counts == {"stars_in_table": 7, "not_open": 3, "not_in_sectors": 1, "selected": 3}
    assert all(t.label is None for t in targets)
    by_tic = {t.tic: t for t in targets}
    assert by_tic[3].reference.toi == "102.02"  # the open TOI with the highest SNR
    assert by_tic[7].sector == 2  # the first of the sectors it was observed in
    assert len(by_tic[3].tois) == 3


def _hosts(n: int, seed: int, planet_rate: float = 0.5) -> Dataset:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < planet_rate).astype(int)
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    features["log_depth"] += 1.2 * y
    features["log_period"] = rng.uniform(0.0, 1.5, n)
    meta = pd.DataFrame({"target_id": [f"TIC {1000 * seed + i}" for i in range(n)]})
    return Dataset(features=features, labels=y, meta=meta)


@pytest.fixture(scope="module")
def trained():
    model, _ = train_on_hosts(
        _hosts(200, seed=1), feature_names=FEATURE_NAMES, n_folds=3, seed=0,
        target_precision=0.5, precision_lcb_z=1.0,
    )
    return model


def _open_targets(dataset: Dataset) -> list[BenchmarkTarget]:
    period = 10.0 ** dataset.features["log_period"].to_numpy()
    targets = []
    for i, target_id in enumerate(dataset.meta["target_id"]):
        tic = int(str(target_id).split()[1])
        # Half the catalogued periods agree with the search, half do not.
        toi = _toi(tic, f"{tic}.01", "PC", [14], period=period[i] * (1.0 if i % 2 else 3.7),
                   depth_ppm=500.0 + i, tess_mag=12.0)
        targets.append(BenchmarkTarget(tic=tic, label=None, sector=14, reference=toi, tois=(toi,)))
    return targets


def test_open_hosts_are_ranked_by_the_model(trained):
    dataset = _hosts(40, seed=2)
    targets = _open_targets(dataset)
    resolved = [
        BenchmarkTarget(tic=t, label=1, sector=1, reference=_toi(t, f"{t}.01", "CP", [1],
                        depth_ppm=900.0, tess_mag=10.0, period=3.0))
        for t in range(10)
    ]
    centroids = [None] * len(dataset)
    centroids[0] = {"status": "ok", "significant": True, "offset_distance_pixels": 1.4,
                    "offset_sigma": 6.0}
    centroids[1] = {"status": "weak_difference_image", "significant": False,
                    "offset_distance_pixels": np.nan, "offset_sigma": np.nan}
    result = rank_open_hosts(
        dataset, targets[::-1], trained, sectors=[14], selection={"selected": 41},
        resolved=resolved, n_without_curve=1, centroids=centroids,
    )

    rows = result.rows
    assert [r["rank"] for r in rows] == list(range(1, 41))
    scores = np.array([r["model_score"] for r in rows])
    assert np.all(np.diff(scores) <= 0)
    probability = np.array([r["probability"] for r in rows])
    assert np.all(np.diff(probability) <= 1e-12)  # calibration keeps the order
    assert [r["kept"] for r in rows] == [s >= trained.threshold for s in scores]
    assert result.n_kept == sum(r["kept"] for r in rows)
    assert all(list(r) == list(OPEN_COLUMNS) for r in rows)
    assert sum(r["period_recovered"] for r in rows) == 20
    by_id = {r["target_id"]: r for r in rows}
    first, second = (str(dataset.meta["target_id"].iloc[i]) for i in (0, 1))
    assert by_id[first]["centroid_flagged"] is True
    assert by_id[first]["centroid_offset_pixels"] == 1.4
    assert by_id[second]["centroid_status"] == "weak_difference_image"
    assert np.isnan(by_id[second]["centroid_offset_pixels"])
    assert sum(r["centroid_status"] is None for r in rows) == 38
    assert result.n_centroids == 2
    assert result.comparison["tess_mag"] == {"open": 12.0, "resolved": 10.0}
    assert result.comparison["toi_depth_ppm"]["resolved"] == 900.0
    assert sum(b["n"] for b in result.probability_bins()) == 40
    assert list(result.by_disposition()) == ["PC"]
    assert result.by_disposition()["PC"]["n"] == 40

    assert all(r["archive_planet"] is None for r in rows)
    assert result.archive_check() is None

    with pytest.raises(ValueError, match="match no open target"):
        rank_open_hosts(dataset, targets[1:], trained, sectors=[14], selection={}, resolved=[])


def test_archive_planets_are_matched_at_the_tois_period(tmp_path):
    table = tmp_path / "planets.csv"
    table.write_text(
        "pl_name,hostname,tic_id,pl_orbper,disc_year\n"
        "A b,A,TIC 1,3.000,2025\n"
        "A c,A,TIC 1,7.5,2025\n"
        "B b,B,TIC 2,,2024\n"
        "C b,C,,4.0,2020\n"
    )
    planets = read_archive_planets(table)
    assert set(planets) == {"TIC 1", "TIC 2"}
    assert archive_match(planets["TIC 1"], 3.02) == "A b"  # within 1%
    assert archive_match(planets["TIC 1"], 7.5) == "A c"
    assert archive_match(planets["TIC 1"], 6.0) is None  # twice a planet's period is not it
    assert archive_match(planets["TIC 1"], 3.05) is None
    assert archive_match(planets["TIC 2"], 5.0) is None  # no period, no match
    assert archive_match(planets["TIC 1"], float("nan")) is None


def test_the_archive_check_asks_where_confirmed_planets_rank(trained):
    from sklearn.metrics import roc_auc_score

    dataset = _hosts(60, seed=4)
    targets = _open_targets(dataset)
    period = {t.target_id: t.reference.period for t in targets}
    # The archive confirms the six best-scored hosts at their TOI's period,
    # and lists another planet, at a different period, on a seventh.
    scores = trained.score(dataset.inputs(tuple(trained.feature_names)))
    ids = dataset.meta["target_id"].astype(str).to_numpy()
    order = ids[np.argsort(-scores)]
    archive = {i: [(f"{i} b", period[i] * 1.001)] for i in order[:6]}
    archive[str(order[-1])] = [(f"{order[-1]} c", period[str(order[-1])] * 2.0)]

    result = rank_open_hosts(
        dataset, targets, trained, sectors=[14], selection={}, resolved=[],
        archive=archive, archive_name="planets.csv",
    )
    named = [r["archive_planet"] for r in result.rows]
    assert named[:6] == [f"{i} b" for i in order[:6]] and named[6:] == [None] * 54

    check = result.archive_check(n_resamples=200)
    assert check["archive"] == "planets.csv"
    assert (check["n_confirmed"], check["n_rest"]) == (6, 54)
    y = np.array([n is not None for n in named], dtype=int)
    assert check["auc"]["model"] == pytest.approx(roc_auc_score(y, [r["model_score"] for r in result.rows]))
    assert check["auc"]["model"] == 1.0  # the top six by construction
    assert check["kept_confirmed"] == np.mean([r["kept"] for r in result.rows[:6]])
    assert 0.0 <= check["model_ahead_of_snr"] <= 1.0
    assert set(check["auc"]) == {"model", "toi_snr", "toi_depth_ppm", "brightness"}
    report = format_open_report(result, top=8)
    assert "Already confirmed in the literature (planets.csv)" in report
    assert "6 of the 60 are a confirmed planet" in report


def test_the_ranking_is_written_and_says_it_has_no_ap(trained, tmp_path):
    dataset = _hosts(12, seed=3)
    result = rank_open_hosts(
        dataset, _open_targets(dataset), trained, sectors=list(range(1, 27)),
        selection={"stars_in_table": 50, "not_open": 30, "not_in_sectors": 6, "selected": 14},
        resolved=[], n_without_curve=2, stitched=True, joined="27-102",
    )
    report = format_open_report(result, top=5)
    assert "OPEN TOIs OF SECTORS 1 TO 26, RANKED" in report
    assert "No average precision" in report
    assert "with every sector of 27-102" in report
    assert "no light curve at MAST: 2   scored: 12" in report

    written = write_open_ranking(result, tmp_path / "open_tois")
    assert sorted(p.name for p in written) == ["open_tois.csv", "open_tois.json", "open_tois.txt"]
    with (tmp_path / "open_tois" / "open_tois.csv").open() as handle:
        table = list(csv.DictReader(handle))
    assert list(table[0]) == list(OPEN_COLUMNS)
    assert [int(r["rank"]) for r in table] == list(range(1, 13))
    assert table[0]["centroid_status"] == ""  # not tested, not cleared
    payload = json.loads((tmp_path / "open_tois" / "open_tois.json").read_text())
    assert payload["n_stars"] == 12 and len(payload["stars"]) == 12
    assert payload["joined"] == "27-102"


def test_the_open_ranking_command_line():
    import run_pipeline

    table = "data/toi_benchmark/exofop_toi_2026-10-06.csv"
    args = run_pipeline.parse_args(
        ["--train-sectors", "1-26", "--rank-open-tois", "--benchmark-tois", table,
         "--stitch", "--centroids-every-sector", "--join-sectors", "27-102"]
    )
    assert args.results_dir.parts[-5:] == (
        "toi_trained", "stitched", "centroids_every_sector", "joined_27-102", "open_tois"
    )
    assert args.open_cache == args.train_cache.parent / "open_curves.npz"
    assert args.open_tpfs == args.train_cache.parent / "open_tpfs"
    assert args.open_centroids == 200
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--rank-open-tois", "--benchmark-tois", table])
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(
            ["--train-sectors", "1-26", "--rank-open-tois", "--benchmark-tois", table,
             "--pixel-features"]
        )
    with pytest.raises(SystemExit):
        # Without --rank-open-tois a training run still needs its benchmark.
        run_pipeline.parse_args(["--train-sectors", "1-26", "--benchmark-tois", table])
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(
            ["--train-sectors", "1-26", "--benchmark-tois", table, "--benchmark-sectors", "14-26",
             "--archive-planets", "planets.csv"]
        )
    assert args.archive_planets is None
