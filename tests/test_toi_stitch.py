"""TOI hosts searched on every sector they were observed in, joined, offline."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import transitml.benchmark as bench
from transitml.config import default_config
from transitml.data.base import LightCurve
from transitml.data.toi import TOI, BenchmarkTarget

from .test_toi_training import _hosts


def curve(tic: int, sector: int, seed: int = 0) -> LightCurve:
    """One 27-day, 30-minute sector of a quiet star, as MAST would serve it."""
    rng = np.random.default_rng(seed + 100 * sector)
    time = np.arange(27.4 * sector, 27.4 * sector + 26.0, 30.0 / 1440.0)
    level = 1.0 + 0.01 * sector  # each sector through its own aperture
    flux = level * (1.0 + rng.normal(0.0, 1e-3, time.size))
    return LightCurve(f"TIC {tic}", time, flux, np.full(time.size, 1e-3 * level),
                      meta={"kind": "real", "sector": sector})


def target(tic: int, sectors: tuple[int, ...], label: int = 1) -> BenchmarkTarget:
    toi = TOI(tic=tic, toi=f"{tic}.01", disposition="CP" if label else "FP", sectors=sectors)
    return BenchmarkTarget(tic=tic, label=label, sector=sectors[0], reference=toi, tois=(toi,))


def fake_mast(served: dict[tuple[str, int], LightCurve], asked: list[list[tuple[str, int]]]):
    def fetch(wanted, **_):
        batch = [(t.target_id, t.sector) for t in wanted]
        # One curve per star per fetch, as fetch_benchmark_curves returns them.
        assert len({target_id for target_id, _ in batch}) == len(batch)
        asked.append(batch)
        return [served[key] for key in batch if key in served]

    return fetch


def test_a_star_asked_for_in_several_sectors_gets_each_of_them(tmp_path, monkeypatch):
    served = {(f"TIC {tic}", s): curve(tic, s) for tic, s in [(1, 14), (1, 15), (1, 16), (2, 14)]}
    asked: list[list[tuple[str, int]]] = []
    monkeypatch.setattr(bench, "fetch_benchmark_curves", fake_mast(served, asked))
    wanted = [replace(target(1, (14,)), sector=s) for s in (14, 15, 16, 17)] + [target(2, (14,))]
    cache = tmp_path / "toi_curves.npz"

    got = bench.load_or_fetch_curves(wanted, cache)
    assert [(lc.target_id, lc.meta["sector"]) for lc in got] == [
        ("TIC 1", 14), ("TIC 1", 15), ("TIC 1", 16), ("TIC 2", 14)
    ]
    assert len(asked) == 4
    assert sorted(key for batch in asked for key in batch) == sorted(
        (t.target_id, t.sector) for t in wanted
    )
    np.testing.assert_array_equal(got[1].flux, served[("TIC 1", 15)].flux)

    again = bench.load_or_fetch_curves(wanted, cache)
    assert len(asked) == 4, "every sector is cached or known to be missing"
    assert [lc.meta["sector"] for lc in again] == [14, 15, 16, 14]


def test_each_star_is_joined_from_all_its_sectors_in_range(tmp_path, monkeypatch):
    served = {
        ("TIC 1", 15): curve(1, 15, seed=1),
        ("TIC 1", 16): curve(1, 16, seed=1),
        ("TIC 1", 30): curve(1, 30, seed=1),  # outside the sectors asked for
        ("TIC 2", 14): curve(2, 14, seed=2),
    }
    asked: list[list[tuple[str, int]]] = []
    monkeypatch.setattr(bench, "fetch_benchmark_curves", fake_mast(served, asked))
    targets = [target(1, (14, 15, 16, 30)), target(2, (14,), label=0), target(3, (14, 15))]

    moved, curves = bench.load_or_fetch_stitched(targets, range(14, 27), tmp_path / "c.npz")
    assert ("TIC 1", 30) not in {key for batch in asked for key in batch}
    # Star 1 has no curve in sector 14, so its pixels come from 15; star 3 has none.
    assert [t.sector for t in moved] == [15, 14, 14]
    assert [lc.target_id for lc in curves] == ["TIC 1", "TIC 2"]
    joined, single = curves
    assert joined.meta["stitched"] and sorted(joined.meta["sectors"]) == [15, 16]
    assert joined.meta["sector"] == 15 and joined.label == 1
    assert joined.n_cadences == served[("TIC 1", 15)].n_cadences + served[("TIC 1", 16)].n_cadences
    # Each sector is put on its own median before joining.
    in_15 = joined.time < 27.4 * 16
    assert np.median(joined.flux[in_15]) == pytest.approx(1.0)
    assert np.median(joined.flux[~in_15]) == pytest.approx(1.0)
    assert "stitched" not in single.meta and single.label == 0


def test_stitch_flag_and_its_defaults(tmp_path):
    import run_pipeline

    table = tmp_path / "toi.csv"
    base = ["--train-sectors", "1-13", "--benchmark-tois", str(table), "--benchmark-sectors", "14-26"]
    one = run_pipeline.parse_args(base)
    args = run_pipeline.parse_args([*base, "--stitch"])
    assert args.results_dir == run_pipeline.ROOT / "results" / "toi_trained" / "stitched"
    assert args.figures_dir == run_pipeline.ROOT / "figures" / "toi_trained" / "stitched"
    # Curves and pixel files are kept by star and sector, so one cache serves both.
    assert args.train_cache == args.benchmark_cache == one.train_cache
    assert args.train_tpfs == args.benchmark_tpfs == one.train_tpfs
    pixels = run_pipeline.parse_args([*base, "--stitch", "--pixel-features"])
    assert pixels.results_dir == args.results_dir / "pixels"
    assert not one.stitch
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--stitch"])


def test_run_pipeline_trains_and_scores_on_joined_sectors(tmp_path, monkeypatch):
    """``--train-sectors --stitch`` end to end: every star searched on all its sectors."""
    import run_pipeline
    import transitml.data.tic as tic_module
    from transitml.data.injection import save_curves

    base = default_config()
    config = replace(base, bls=replace(base.bls, n_periods=400))
    train_curves, train_tois = _hosts(config, 60, seed=21, tic0=6000, sector=13)
    test_curves, test_tois = _hosts(config, 24, seed=22, tic0=8000, sector=15)
    # Training star 0 was observed again in sector 14, and benchmark star 0 in 16.
    again = [replace(train_curves[0], time=train_curves[0].time + 27.4,
                     meta={**train_curves[0].meta, "sector": 14}),
             replace(test_curves[0], time=test_curves[0].time + 27.4,
                     meta={**test_curves[0].meta, "sector": 16})]
    tois = [replace(train_tois[0], sectors=(13, 14)), *train_tois[1:],
            replace(test_tois[0], sectors=(15, 16)), *test_tois[1:]]
    table = tmp_path / "exofop_toi.csv"
    table.write_text(
        "TIC ID,TOI,TFOPWG Disposition,Period (days),Planet SNR,Sectors\n"
        + "".join(
            f'{t.tic},{t.toi},{t.disposition},{t.period},{t.snr},"{",".join(map(str, t.sectors))}"\n'
            for t in tois
        )
    )
    results = tmp_path / "results"
    results.mkdir()
    save_curves(train_curves + test_curves + again, results / "toi_curves.npz")

    def no_network(*_, **__):
        raise AssertionError("everything is cached; MAST must not be queried")

    monkeypatch.setattr(bench, "fetch_benchmark_curves", no_network)
    monkeypatch.setattr(
        tic_module,
        "fetch_tic_stars",
        lambda ids: {tic: {"teff_k": 5800.0, "rho_star_cgs": 1.4} for tic in ids},
    )
    args = run_pipeline.parse_args(
        [
            "--train-sectors", "13-14",
            "--benchmark-tois", str(table),
            "--benchmark-sectors", "15-16",
            "--results-dir", str(results),
            "--no-figures",
            "--n-jobs", "2",
            "--stitch",
        ]
    )
    assert run_pipeline.run_toi_training(args, config, started=0.0) == 0

    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["dataset"]["n_curves"] == 60
    assert metrics["toi_benchmark"]["n_stars"] == 24
    # The joined star carries both sectors' cadences.
    saved = np.load(results / "dataset.npz", allow_pickle=True)
    meta = dict(zip(saved["meta_names"], saved["meta_values"].T))
    cadences = dict(zip(meta["target_id"], meta["n_cadences"]))
    assert cadences[f"TIC {train_tois[0].tic}"] == 2 * train_curves[0].n_cadences
    assert cadences[f"TIC {train_tois[1].tic}"] == train_curves[1].n_cadences
    scored = json.loads((results / "toi_benchmark.json").read_text())
    assert {s["target_id"] for s in scored["stars"]} == {lc.target_id for lc in test_curves}
    assert scored["stitched"] and metrics["training"]["stitched"]
    assert "each searched on every one of these sectors" in (results / "report.txt").read_text()
    assert "(every sector of each star, joined)" in (results / "toi_benchmark.txt").read_text()
