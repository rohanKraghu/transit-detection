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


def test_joined_sectors_add_data_to_a_star_but_never_a_star(tmp_path, monkeypatch):
    served = {
        ("TIC 1", 15): curve(1, 15, seed=1),
        ("TIC 1", 40): curve(1, 40, seed=1),
        ("TIC 1", 41): curve(1, 41, seed=1),
        ("TIC 2", 40): curve(2, 40, seed=2),
        ("TIC 3", 14): curve(3, 14, seed=3),
    }
    asked: list[list[tuple[str, int]]] = []
    monkeypatch.setattr(bench, "fetch_benchmark_curves", fake_mast(served, asked))
    targets = [target(1, (15, 40, 41)), target(2, (40,)), target(3, (14,), label=0)]

    moved, curves = bench.load_or_fetch_stitched(
        targets, range(14, 27), tmp_path / "c.npz", join=range(27, 103)
    )
    # Star 2 has nothing in 14-26, so its later sector is not even asked for.
    assert ("TIC 2", 40) not in {key for batch in asked for key in batch}
    assert [t.sector for t in moved] == [15, 40, 14]
    assert [lc.target_id for lc in curves] == ["TIC 1", "TIC 3"]
    joined, single = curves
    assert sorted(joined.meta["sectors"]) == [15, 40, 41]
    assert joined.meta["selection_sectors"] == [15] and joined.meta["sector"] == 15
    assert joined.n_cadences == sum(served[("TIC 1", s)].n_cadences for s in (15, 40, 41))
    assert single.meta["selection_sectors"] == [14] and single.meta["sector"] == 14
    # Without join, nothing changes.
    _, plain = bench.load_or_fetch_stitched(targets, range(14, 27), tmp_path / "c.npz")
    assert [lc.n_cadences for lc in plain] == [served[("TIC 1", 15)].n_cadences,
                                               served[("TIC 3", 14)].n_cadences]
    assert "selection_sectors" not in plain[0].meta


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
    every = run_pipeline.parse_args([*base, "--stitch", "--pixel-features", "--centroids-every-sector"])
    assert every.results_dir == args.results_dir / "centroids_every_sector" / "pixels"
    assert every.figures_dir == args.figures_dir / "centroids_every_sector" / "pixels"
    assert every.train_tpfs == one.train_tpfs
    veto = run_pipeline.parse_args(
        [*base, "--stitch", "--benchmark-centroids", "--centroids-every-sector"]
    )
    assert veto.results_dir == args.results_dir / "centroids_every_sector"
    sky = run_pipeline.parse_args(
        [*base, "--stitch", "--pixel-features", "--centroids-every-sector", "--sky-offsets"]
    )
    assert sky.results_dir == args.results_dir / "sky_offsets" / "pixels"
    assert sky.figures_dir == args.figures_dir / "sky_offsets" / "pixels"
    assert not veto.sky_offsets
    joined = run_pipeline.parse_args(
        [*base, "--stitch", "--pixel-features", "--centroids-every-sector", "--join-sectors", "27-102"]
    )
    assert joined.results_dir == args.results_dir / "centroids_every_sector" / "joined_27-102" / "pixels"
    assert joined.train_cache == one.train_cache and joined.train_tpfs == one.train_tpfs
    assert run_pipeline.apply_overrides(default_config(), joined).bls.max_search_baseline_days == (
        run_pipeline.JOINED_SEARCH_BASELINE_DAYS
    )
    assert run_pipeline.apply_overrides(default_config(), args).bls.max_search_baseline_days is None
    for wrong in (["--centroids-every-sector", "--benchmark-centroids"],  # nothing joined
                  ["--stitch", "--centroids-every-sector"],  # no centroid test to change
                  ["--stitch", "--benchmark-centroids", "--sky-offsets"],  # one sector each
                  ["--join-sectors", "27-102"],  # nothing to add them to
                  ["--stitch", "--join-sectors", "x"]):
        with pytest.raises(SystemExit):
            run_pipeline.parse_args([*base, *wrong])


def _joined_hosts(tmp_path, monkeypatch):
    """60 training hosts of sector 13 and 24 benchmark hosts of 15, one of each seen again.

    Everything is in the curve cache, so MAST must not be asked for a curve.
    Returns the config, the run's arguments without the flag under test, and
    both sets of curves and TOIs.
    """
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
    argv = [
        "--train-sectors", "13-14",
        "--benchmark-tois", str(table),
        "--benchmark-sectors", "15-16",
        "--results-dir", str(results),
        "--no-figures",
        "--n-jobs", "2",
        "--stitch",
    ]
    return config, argv, results, (train_curves, train_tois), (test_curves, test_tois)


def test_run_pipeline_trains_and_scores_on_joined_sectors(tmp_path, monkeypatch):
    """``--train-sectors --stitch`` end to end: every star searched on all its sectors."""
    import run_pipeline

    config, argv, results, (train_curves, train_tois), (test_curves, _) = _joined_hosts(
        tmp_path, monkeypatch
    )
    args = run_pipeline.parse_args(argv)
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


def test_run_pipeline_adds_later_sectors_as_data(tmp_path, monkeypatch):
    """``--join-sectors``: later sectors lengthen a star's curve and the search is windowed."""
    import run_pipeline
    from transitml.data.injection import load_curves, save_curves

    config, argv, results, (train_curves, train_tois), (test_curves, test_tois) = _joined_hosts(
        tmp_path, monkeypatch
    )
    # Training star 1 and benchmark star 1 were seen again years later, in sector 60.
    later = [replace(lc, time=lc.time + 1200.0, meta={**lc.meta, "sector": 60})
             for lc in (train_curves[1], test_curves[1])]
    cache = results / "toi_curves.npz"
    save_curves(load_curves(cache) + later, cache)
    table = tmp_path / "exofop_toi.csv"
    seen_again = {f"{train_tois[1].tic},", f"{test_tois[1].tic},"}
    table.write_text("".join(
        (line[:-1] + ',60"' if any(line.startswith(t) for t in seen_again) else line) + "\n"
        for line in table.read_text().splitlines()
    ))
    args = run_pipeline.parse_args([*argv, "--join-sectors", "60"])
    config = run_pipeline.apply_overrides(config, args)
    assert run_pipeline.run_toi_training(args, config, started=0.0) == 0

    saved = np.load(results / "dataset.npz", allow_pickle=True)
    meta = dict(zip(saved["meta_names"], saved["meta_values"].T))
    cadences = dict(zip(meta["target_id"], meta["n_cadences"]))
    assert cadences[f"TIC {train_tois[1].tic}"] == 2 * train_curves[1].n_cadences
    assert cadences[f"TIC {train_tois[2].tic}"] == train_curves[2].n_cadences
    scored = json.loads((results / "toi_benchmark.json").read_text())
    assert len(scored["stars"]) == 24 and scored["joined"] == "60"
    assert json.loads((results / "metrics.json").read_text())["training"]["joined"] == "60"
    assert "every sector of 60 it was observed in added" in (results / "report.txt").read_text()
    assert "with its sectors of 60 added" in (results / "toi_benchmark.txt").read_text()


def _scene_hosts(n: int, *, tic0: int, sector: int, again: int, cache):
    """Planets on their targets (CP) and blended binaries (FP), each a pixel scene.

    Each curve is its own stamp's aperture sum, as in the benchmark's centroid
    tests.  Star 0 is seen again in sector ``again``, its stamp turned and its
    eclipses kept on the ephemeris.  Returns the curves and TOIs.
    """
    from transitml.data.synthetic_tpf import blend_scenario
    from transitml.data.tpf import save_tpf

    curves, tois = [], []
    for i in range(n):
        tic, label = tic0 + i, int(i % 2 == 0)
        kind = "on_target" if label else "blend"
        seen = [(sector, (1.6, 0.8), 0.0)] + ([(again, (-0.8, 1.6), 27.9)] if i == 0 else [])
        for s, offset, shift in seen:
            tpf, truth = blend_scenario(kind, neighbour_offset=offset, seed=tic + s)
            tpf.time = tpf.time + shift  # nine periods of 3.1 days
            tpf.target_id = f"TIC {tic}"
            save_tpf(tpf, bench.tpf_cache_path(cache, tpf.target_id, s))
            curves.append(replace(tpf.to_light_curve(), label=label,
                                  meta={"kind": "real", "sector": s}))
        tois.append(TOI(tic=tic, toi=f"{tic}.01", disposition="CP" if label else "FP",
                        period=truth["period"], snr=20.0,
                        sectors=tuple(s for s, _, _ in seen)))
    return curves, tois


def test_run_pipeline_tests_the_pixels_of_every_joined_sector(tmp_path, monkeypatch):
    """``--centroids-every-sector``: one pixel file per sector a star was joined from."""
    import run_pipeline
    import transitml.data.tic as tic_module
    from transitml.data.injection import save_curves

    results = tmp_path / "results"
    tpfs = results / "toi_tpfs"
    train_curves, train_tois = _scene_hosts(40, tic0=6000, sector=13, again=14, cache=tpfs)
    test_curves, test_tois = _scene_hosts(16, tic0=8000, sector=15, again=16, cache=tpfs)
    table = tmp_path / "exofop_toi.csv"
    table.write_text(
        "TIC ID,TOI,TFOPWG Disposition,Period (days),Planet SNR,Sectors\n"
        + "".join(
            f'{t.tic},{t.toi},{t.disposition},{t.period},{t.snr},"{",".join(map(str, t.sectors))}"\n'
            for t in train_tois + test_tois
        )
    )
    save_curves(train_curves + test_curves, results / "toi_curves.npz")

    def no_network(*_, **__):
        raise AssertionError("everything is cached; MAST must not be queried")

    monkeypatch.setattr(bench, "fetch_benchmark_curves", no_network)
    monkeypatch.setattr(bench, "download_tpfs", no_network)
    monkeypatch.setattr(
        tic_module,
        "fetch_tic_stars",
        lambda ids: {tic: {"teff_k": 5800.0, "rho_star_cgs": 1.4} for tic in ids},
    )
    base = default_config()
    config = replace(base, bls=replace(base.bls, n_periods=400))
    args = run_pipeline.parse_args(
        [
            "--train-sectors", "13-14",
            "--benchmark-tois", str(table),
            "--benchmark-sectors", "15-16",
            "--results-dir", str(results),
            "--no-figures",
            "--n-jobs", "2",
            "--stitch",
            "--pixel-features",
            "--centroids-every-sector",
        ]
    )
    assert args.results_dir == results  # an explicit directory is kept as given
    assert run_pipeline.run_toi_training(args, config, started=0.0) == 0

    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["training"]["n_with_pixels"] == 40
    assert metrics["training"]["n_pixel_files"] == 41
    scored = json.loads((results / "toi_benchmark.json").read_text())
    assert scored["centroid_veto"]["n_with_pixels"] == 16
    assert scored["centroid_veto"]["n_pixel_files"] == 17
    stars = {s["target_id"]: s for s in scored["stars"]}
    joined = stars[f"TIC {test_tois[0].tic}"]["centroid"]
    assert (joined["n_sectors"], joined["n_sectors_placed"]) == (2, 2)
    assert stars[f"TIC {test_tois[1].tic}"]["centroid"]["n_sectors"] == 1
    # The blended binaries are flagged, the planets are not.
    flagged = {t: s["centroid"]["significant"] for t, s in stars.items()}
    assert sum(flagged[f"TIC {t.tic}"] for t in test_tois if t.disposition == "FP") >= 6
    assert not any(flagged[f"TIC {t.tic}"] for t in test_tois if t.disposition == "CP")
    assert "each tested in every sector it was joined from (41 pixel files)" in (
        results / "report.txt"
    ).read_text()
    assert "each tested in every sector it was joined from (17 pixel files)" in (
        results / "toi_benchmark.txt"
    ).read_text()
