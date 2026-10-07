#!/usr/bin/env python3
"""One command: generate data, detrend, search, train, evaluate, plot.

    python run_pipeline.py

Writes ``results/metrics.json``, ``results/report.txt``, ``results/dataset.npz``,
the fitted classifier as ``results/model.joblib`` (for scoring new light
curves) and four PNGs to ``figures/``.  Deterministic given ``--seed``; the default run
takes ~2 minutes on four cores.

Injection-recovery on real photometry (needs ``pip install lightkurve`` and
network access to MAST the first time; later runs reuse the cache)::

    python run_pipeline.py --inject-into targets.txt --exclude-tois toi.csv --sector 14

downloads one sector of real light curves for every target that is not a known
TOI host, injects the same planet and binary population as the synthetic run,
and writes everything to ``results/real_injection/`` and
``figures/real_injection/`` so the synthetic headline is never overwritten.

Structured spacecraft systematics in the synthetic sector (scattered light on
the 13.7-day orbit, camera-wide pointing jitter and momentum dumps, focus
drift)::

    python run_pipeline.py --systematics

writes to ``results/systematics/`` and ``figures/systematics/``.

Benchmark against real labels (either mode; needs ``lightkurve`` the first
time)::

    python run_pipeline.py ... --benchmark-tois exofop_toi.csv --benchmark-sectors 14-26

scores the trained model, unchanged, on TOI hosts whose follow-up disposition
is known (CP/KP planets, FP/FA false positives) and writes
``toi_benchmark.json`` and ``toi_benchmark.txt`` beside the metrics.  With
``--benchmark-centroids`` it also downloads each host's target pixel file and
scores the model again with the centroid test as a veto.

Training on real labels instead::

    python run_pipeline.py --train-sectors 1-13 \
        --benchmark-tois exofop_toi.csv --benchmark-sectors 14-26

trains on the labelled TOI hosts of sectors 1 to 13 and benchmarks the model on
those of sectors 14 to 26, none of them trained on; ``--pixel-features`` also
gives it the centroid test.  Writes to ``results/toi_trained/``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from transitml.config import Config, default_config
from transitml.data.base import LightCurve, LightCurveSource
from transitml.data.injection import (
    InjectionSource,
    exclude_known_hosts,
    load_curves,
    load_excluded_tic_ids,
    read_target_list,
    save_curves,
    tic_number,
)
from transitml.data.loader import Dataset, build_dataset
from transitml.data.synthetic import SYSTEMATIC_COMPONENTS, SyntheticTESSSource
from transitml.data.toi import (
    BenchmarkTarget,
    later_target,
    observed_in,
    parse_sector_spec,
    read_toi_table,
    select_benchmark_targets,
)
from transitml.evaluate import evaluate, format_report
from transitml.features import BLS_ENGINE_ENV, FEATURE_NAMES, bls_engine
from transitml.model import TrainedModel, make_split, save_model, train
from transitml.plots import plot_all, plot_sector_systematics

ROOT = Path(__file__).resolve().parent


def _display_path(path: Path) -> str:
    """Repository-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end exoplanet transit detection on synthetic TESS-like photometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--n-curves", type=int, default=None, help="Override the number of light curves."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Worker processes for the BLS search."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results", help="Where metrics land."
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ROOT / "figures", help="Where PNGs land."
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip plotting (useful in CI)."
    )
    parser.add_argument(
        "--systematics",
        action="store_true",
        help="Add structured spacecraft systematics shared across the sector "
        "(synthetic data only); writes to results/systematics/.",
    )
    parser.add_argument(
        "--systematics-scale",
        type=float,
        default=1.0,
        help="With --systematics: multiply every coupling (0 adds the sector's "
        "gap and dropped cadences but no signal).",
    )
    parser.add_argument(
        "--systematics-components",
        default=",".join(SYSTEMATIC_COMPONENTS),
        help="With --systematics: comma-separated components to add.",
    )
    search = parser.add_argument_group("periodic search")
    search.add_argument(
        "--search",
        choices=("bls", "tls"),
        default="bls",
        help="Box Least Squares, or Transit Least Squares (needs transitleastsquares; "
        "writes to results/tls/).",
    )
    search.add_argument(
        "--bls-engine",
        choices=("astropy", "cpu", "gpu"),
        default=None,
        help="Who computes the BLS periodogram; all give the same result. gpu needs CuPy. "
        f"None keeps ${BLS_ENGINE_ENV} or astropy.",
    )
    real = parser.add_argument_group(
        "injection-recovery on real photometry",
        "Inject the synthetic planet/binary population into real MAST light curves.",
    )
    real.add_argument(
        "--inject-into",
        type=Path,
        default=None,
        help="Target list (one TIC per line, or a CSV with a TIC column).",
    )
    real.add_argument(
        "--exclude-tois",
        type=Path,
        default=None,
        help="ExoFOP TOI CSV; listed TICs are dropped as known or candidate hosts.",
    )
    real.add_argument("--sector", type=int, default=None, help="TESS sector to use.")
    real.add_argument("--author", default="TESS-SPOC", help="Light-curve pipeline.")
    real.add_argument(
        "--exposure-time", type=int, default=1800, help="Cadence in seconds."
    )
    real.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Targets fetched from MAST concurrently.",
    )
    real.add_argument(
        "--curve-cache",
        type=Path,
        default=None,
        help="npz of downloaded curves; read if present, written after a download.",
    )
    bench = parser.add_argument_group(
        "benchmark against real labels",
        "Score the trained model on TOI hosts with a follow-up disposition.",
    )
    bench.add_argument(
        "--benchmark-tois",
        type=Path,
        default=None,
        help="ExoFOP TOI CSV (download_toi.php?output=csv); needs its Sectors column.",
    )
    bench.add_argument(
        "--benchmark-sectors",
        default=None,
        help="Sectors to score, in preference order, e.g. 14 or 14-26; "
        "None means --sector, else 14.",
    )
    bench.add_argument(
        "--benchmark-cache",
        type=Path,
        default=None,
        help="npz of downloaded TOI-host curves; None means toi_curves.npz in the results dir.",
    )
    bench.add_argument(
        "--benchmark-stars",
        type=Path,
        default=None,
        help="CSV of the hosts' TIC temperatures and densities, read if present; stars "
        "it lacks are looked up at MAST. None means tic_stars.csv beside the TOI table.",
    )
    bench.add_argument(
        "--benchmark-centroids",
        action="store_true",
        help="Also run the centroid test on each host's target pixel file and score the "
        "model with it as a veto.",
    )
    bench.add_argument(
        "--benchmark-tpfs",
        type=Path,
        default=None,
        help="Directory of downloaded target pixel files, one npz per star; None means "
        "toi_tpfs/ beside --benchmark-cache.",
    )
    bench.add_argument(
        "--benchmark-comments",
        type=Path,
        default=None,
        help="CSV of ExoFOP comments (TOI, Comments) used to sort the false positives by "
        "why they were retired; None means toi_comments.csv beside the TOI table.",
    )
    labels = parser.add_argument_group(
        "training on real labels",
        "Train on the labelled TOI hosts of some sectors instead of on synthetic or injected "
        "curves, then benchmark the model on the hosts of others.",
    )
    labels.add_argument(
        "--train-sectors",
        default=None,
        help="Sectors whose labelled TOI hosts (from --benchmark-tois) to train on, e.g. "
        "1-13; writes to results/toi_trained/.",
    )
    labels.add_argument(
        "--train-cache",
        type=Path,
        default=None,
        help="npz of the training hosts' curves; None means toi_curves.npz in the results "
        "dir, shared with the benchmark's (curves are kept by star and sector).",
    )
    labels.add_argument(
        "--train-tpfs",
        type=Path,
        default=None,
        help="Directory of their target pixel files; None means toi_tpfs/ beside --train-cache.",
    )
    labels.add_argument(
        "--pixel-features",
        action="store_true",
        help="With --train-sectors: also give the model the centroid test's offset, its "
        "significance and the difference-image SNR (downloads each host's target pixel "
        "file; implies --benchmark-centroids; writes to results/toi_trained/pixels/).",
    )
    args = parser.parse_args(argv)
    default_results = args.results_dir == ROOT / "results"
    default_figures = args.figures_dir == ROOT / "figures"
    components = tuple(c.strip() for c in args.systematics_components.split(",") if c.strip())
    unknown = set(components) - set(SYSTEMATIC_COMPONENTS)
    if unknown:
        parser.error(
            f"unknown --systematics-components {sorted(unknown)}; "
            f"choose from {', '.join(SYSTEMATIC_COMPONENTS)}"
        )
    args.systematics_components = components
    if args.systematics:
        if args.inject_into is not None:
            parser.error("--systematics is for synthetic data; real light curves have their own")
        if args.results_dir == ROOT / "results":
            args.results_dir = ROOT / "results" / "systematics"
        if args.figures_dir == ROOT / "figures":
            args.figures_dir = ROOT / "figures" / "systematics"
    if args.inject_into is not None:
        if args.results_dir == ROOT / "results":
            args.results_dir = ROOT / "results" / "real_injection"
        if args.figures_dir == ROOT / "figures":
            args.figures_dir = ROOT / "figures" / "real_injection"
        if args.curve_cache is None:
            args.curve_cache = args.results_dir / "base_curves.npz"
    if args.train_sectors is not None:
        if args.benchmark_tois is None or args.benchmark_sectors is None:
            parser.error(
                "--train-sectors reads the TOI table from --benchmark-tois and scores the "
                "model on --benchmark-sectors; give both"
            )
        if args.inject_into is not None or args.systematics:
            parser.error(
                "--train-sectors trains on TOI hosts; drop --inject-into and --systematics"
            )
        if default_results:
            args.results_dir = ROOT / "results" / "toi_trained"
        if default_figures:
            args.figures_dir = ROOT / "figures" / "toi_trained"
        # One cache for both sets of hosts, set before --pixel-features moves the
        # results: curves and pixel files are kept by star and sector, so a run
        # with pixel features reuses what the run without them downloaded.
        if args.train_cache is None:
            args.train_cache = args.results_dir / "toi_curves.npz"
        if args.benchmark_cache is None:
            args.benchmark_cache = args.results_dir / "toi_curves.npz"
        if args.train_tpfs is None:
            args.train_tpfs = Path(args.train_cache).parent / "toi_tpfs"
        if args.pixel_features:
            args.benchmark_centroids = True
            if default_results:
                args.results_dir = args.results_dir / "pixels"
            if default_figures:
                args.figures_dir = args.figures_dir / "pixels"
    elif args.pixel_features:
        parser.error("--pixel-features needs --train-sectors: only TOI hosts come with pixels")
    if args.search == "tls":
        # Beside the BLS run of the same data: results/tls/, results/systematics/tls/, ...
        if default_results:
            args.results_dir = args.results_dir / "tls"
        if default_figures:
            args.figures_dir = args.figures_dir / "tls"
    if args.benchmark_tois is not None and args.benchmark_cache is None:
        args.benchmark_cache = args.results_dir / "toi_curves.npz"
    if args.benchmark_tois is not None and args.benchmark_stars is None:
        args.benchmark_stars = args.benchmark_tois.parent / "tic_stars.csv"
    if args.benchmark_tois is not None and args.benchmark_tpfs is None:
        args.benchmark_tpfs = Path(args.benchmark_cache).parent / "toi_tpfs"
    if args.benchmark_tois is not None and args.benchmark_comments is None:
        args.benchmark_comments = args.benchmark_tois.parent / "toi_comments.csv"
    return args


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Return a config with the command-line overrides applied."""
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.n_curves is not None:
        config = replace(config, dataset=replace(config.dataset, n_curves=args.n_curves))
    if args.systematics:
        config = replace(
            config,
            systematics=replace(
                config.systematics,
                enabled=True,
                scale=args.systematics_scale,
                components=args.systematics_components,
            ),
        )
    if args.search != config.bls.search:
        config = replace(config, bls=replace(config.bls, search=args.search))
    return config


def build_source(config: Config, args: argparse.Namespace) -> LightCurveSource:
    """The synthetic population, or that population injected into real curves."""
    if args.inject_into is None:
        return SyntheticTESSSource(
            n_curves=config.dataset.n_curves,
            positive_rate=config.dataset.positive_rate,
            eclipsing_binary_rate=config.dataset.eclipsing_binary_rate,
            seed=config.seed,
            survey=config.survey,
            noise=config.noise,
            star=config.star,
            planet=config.planet,
            eb=config.eb,
            systematics=config.systematics,
        )

    cache = Path(args.curve_cache)
    if cache.exists():
        curves = load_curves(cache)
        if args.n_curves is not None:
            curves = curves[: args.n_curves]
        print(f"  loaded {len(curves)} real light curves from {_display_path(cache)}")
    else:
        from transitml.data.mast import MASTLightCurveSource

        targets = read_target_list(args.inject_into)
        if args.exclude_tois is not None:
            targets, dropped = exclude_known_hosts(
                targets, load_excluded_tic_ids(args.exclude_tois)
            )
            print(f"  excluded {len(dropped)} TOI hosts; {len(targets)} targets remain")
        if args.n_curves is not None:
            targets = targets[: args.n_curves]
        mast = MASTLightCurveSource(
            [(t, None) for t in targets],
            mission="TESS",
            author=args.author,
            exposure_time=args.exposure_time,
            sector=args.sector,
            n_workers=args.download_workers,
        )
        curves, seen = [], set()
        for lc in mast:
            # One sector per star, so a star cannot sit in both splits.
            if lc.target_id not in seen:
                seen.add(lc.target_id)
                curves.append(lc)
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_curves(curves, cache)
        print(f"  downloaded {len(curves)} real light curves; cached to {_display_path(cache)}")

    return InjectionSource(
        curves,
        config.dataset.positive_rate,
        config.dataset.eclipsing_binary_rate,
        seed=config.seed,
        star=config.star,
        planet=config.planet,
        eb=config.eb,
    )


def training_tic_ids(source: LightCurveSource) -> set[int]:
    """TIC numbers of the real stars the model was trained on (none if synthetic)."""
    if not isinstance(source, InjectionSource):
        return set()
    return {n for lc in source.base_curves if (n := tic_number(lc.target_id)) is not None}


@dataclass
class TOIHosts:
    """The labelled TOI hosts of some sectors: their curves searched, their pixels tested."""

    sectors: list[int]
    targets: list[BenchmarkTarget]
    selection: dict[str, int]
    dataset: Dataset
    n_without_curve: int
    #: One centroid test per dataset row, when asked for; the dataset then
    #: carries them as features too.
    centroids: list[dict[str, Any] | None] | None = None


def load_toi_hosts(
    args: argparse.Namespace,
    config: Config,
    spec: str,
    *,
    cache: Path,
    tpfs: Path,
    centroids: bool,
    exclude_tics: set[int] | None = None,
    training: bool = False,
) -> TOIHosts:
    """Select, fetch (or read from the caches) and search the labelled TOI hosts of ``spec``.

    ``exclude_tics`` names stars that must not be among them: for a benchmark,
    the stars the model was trained on; for a ``training`` set, the stars
    observed in the benchmark's sectors.  A benchmark star is scored on the
    first of the sectors it was observed in; a training star MAST has no light
    curve for there is taken from its next sector in ``spec`` instead, since a
    training set loses nothing by that.
    """
    from transitml.benchmark import (
        build_benchmark_dataset,
        centroid_tests,
        load_or_fetch_curves,
        load_or_fetch_tpfs,
        with_centroid_features,
        with_tic_stars,
    )

    sectors = parse_sector_spec(spec)
    targets, selection = select_benchmark_targets(
        read_toi_table(args.benchmark_tois), sectors, exclude_tics=exclude_tics or set()
    )
    if training:
        selection["in_benchmark_sectors"] = selection.pop("in_training_set")
    dropped = (
        f" ({selection['in_benchmark_sectors']} dropped as observed in the benchmark's sectors)"
        if training and selection["in_benchmark_sectors"]
        else f" ({selection['in_training_set']} dropped as training stars)"
        if not training
        else ""
    )
    print(
        f"\n{'Training set' if training else 'TOI benchmark'}: {selection['positives']} CP/KP "
        f"and {selection['negatives']} FP/FA hosts in sectors {spec}{dropped}"
    )

    def fetch(wanted: list[BenchmarkTarget]) -> list[LightCurve]:
        return load_or_fetch_curves(
            wanted,
            cache,
            author=args.author,
            exposure_time=args.exposure_time,
            n_workers=args.download_workers,
        )

    curves = fetch(targets)
    if training:
        found = {lc.target_id: lc for lc in curves}
        current = {t.target_id: t for t in targets}
        missing = [t for t in targets if t.target_id not in found]
        while missing := [m for t in missing if (m := later_target(t, sectors)) is not None]:
            current.update((t.target_id, t) for t in missing)
            found.update((lc.target_id, lc) for lc in fetch(missing))
            missing = [t for t in missing if t.target_id not in found]
        moved = [current[t.target_id] for t in targets if current[t.target_id] != t]
        selection["from_a_later_sector"] = sum(1 for t in moved if t.target_id in found)
        targets = [current[t.target_id] for t in targets]
        curves = [found[t.target_id] for t in targets if t.target_id in found]
    curves = with_tic_stars(curves, args.benchmark_stars)
    n_known = sum(1 for lc in curves if all(np.isfinite(lc.star)))
    print(
        f"  {len(curves)} have a light curve, {n_known} a TIC temperature or density; "
        "searching them ..."
    )
    dataset = build_benchmark_dataset(
        curves, preprocess=config.preprocess, bls=config.bls, n_jobs=args.n_jobs
    )
    tests = None
    if centroids:
        scored = {lc.target_id for lc in curves}
        paths = load_or_fetch_tpfs(
            [t for t in targets if t.target_id in scored],
            tpfs,
            author=args.author,
            exposure_time=args.exposure_time,
            n_workers=args.download_workers,
        )
        print(f"  {len(paths)} have a target pixel file; running the centroid test ...")
        tests = centroid_tests(dataset, paths, n_jobs=args.n_jobs)
        dataset = with_centroid_features(dataset, tests)
    return TOIHosts(
        sectors=sectors,
        targets=targets,
        selection=selection,
        dataset=dataset,
        n_without_curve=len(targets) - len(curves),
        centroids=tests,
    )


def run_toi_benchmark(
    args: argparse.Namespace,
    config: Config,
    trained: TrainedModel,
    exclude_tics: set[int],
    *,
    importance_repeats: int = 0,
) -> dict[str, Any]:
    """Score ``trained`` on labelled TOI hosts; write the report, JSON and figure.

    ``exclude_tics`` are the stars it was trained on, which are not scored.
    """
    from transitml.benchmark import benchmark, format_benchmark_report, plot_benchmark
    from transitml.data.toi import read_toi_comments

    spec = args.benchmark_sectors or str(args.sector if args.sector is not None else 14)
    hosts = load_toi_hosts(
        args,
        config,
        spec,
        cache=args.benchmark_cache,
        tpfs=args.benchmark_tpfs,
        # A model with pixel features cannot be scored without them.
        centroids=args.benchmark_centroids or tuple(trained.feature_names) != FEATURE_NAMES,
        exclude_tics=exclude_tics,
    )
    comments = (
        read_toi_comments(args.benchmark_comments)
        if Path(args.benchmark_comments).exists()
        else None
    )
    result = benchmark(
        hosts.dataset,
        hosts.targets,
        trained,
        sectors=hosts.sectors,
        selection=hosts.selection,
        n_without_curve=hosts.n_without_curve,
        top_k=config.evaluation.top_k,
        seed=config.seed,
        centroids=hosts.centroids,
        comments=comments,
        importance_repeats=importance_repeats,
    )
    report = format_benchmark_report(result)
    print()
    print(report)

    results_dir = Path(args.results_dir)
    payload = result.to_dict()
    payload["toi_table"] = _display_path(Path(args.benchmark_tois).resolve())
    if not args.no_figures:
        figure = plot_benchmark(result, Path(args.figures_dir) / "05_toi_benchmark.png")
        payload["figure"] = _display_path(figure)
    (results_dir / "toi_benchmark.json").write_text(json.dumps(payload, indent=2, default=str))
    (results_dir / "toi_benchmark.txt").write_text(report + "\n")
    return payload


def benchmark_headline(bench: dict[str, Any]) -> dict[str, Any]:
    """The few TOI benchmark numbers ``metrics.json`` repeats."""
    return {k: bench[k] for k in ("n_stars", "n_planets", "chance_average_precision")} | {
        "average_precision": bench["model"]["average_precision"],
        "planet_recall": bench["operating_point"]["planet_recall"],
        "false_positive_rejection": bench["operating_point"]["false_positive_rejection"],
    }


def run_toi_training(args: argparse.Namespace, config: Config, started: float) -> int:
    """Train on the TOI hosts of ``--train-sectors``, then benchmark on ``--benchmark-sectors``."""
    from transitml.benchmark import CENTROID_FEATURE_NAMES
    from transitml.toi_training import (
        format_training_report,
        scored_calibration,
        summarise_training,
        train_on_hosts,
    )

    # A star observed in the benchmark's sectors is never trained on, so the
    # benchmark scores the same stars whichever sectors the model learned from.
    benchmark_stars = observed_in(
        read_toi_table(args.benchmark_tois), parse_sector_spec(args.benchmark_sectors)
    )
    hosts = load_toi_hosts(
        args,
        config,
        args.train_sectors,
        cache=args.train_cache,
        tpfs=args.train_tpfs,
        centroids=args.pixel_features,
        exclude_tics=benchmark_stars,
        training=True,
    )
    names = FEATURE_NAMES + (CENTROID_FEATURE_NAMES if args.pixel_features else ())
    evaluation = config.evaluation
    trained, split = train_on_hosts(
        hosts.dataset,
        feature_names=names,
        n_folds=config.dataset.n_cv_folds,
        seed=config.seed,
        target_precision=evaluation.target_precision,
        precision_lcb_z=evaluation.precision_lcb_z,
    )
    print(f"trained; operating threshold {trained.threshold:.4f} ({trained.threshold_rule})")
    summary = summarise_training(
        hosts.dataset,
        split,
        trained,
        sectors=hosts.sectors,
        selection=hosts.selection,
        n_without_curve=hosts.n_without_curve,
        n_folds=config.dataset.n_cv_folds,
        target_precision=evaluation.target_precision,
        precision_lcb_z=evaluation.precision_lcb_z,
        seed=config.seed,
        centroids=hosts.centroids,
    )
    print()
    print(format_training_report(summary))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    source = f"TOI hosts, sectors {args.train_sectors}"
    payload: dict[str, Any] = {
        "config": config.to_dict(),
        "runtime_seconds": None,
        "dataset": {
            "source": source,
            "n_curves": len(hosts.dataset),
            "n_planets": int(hosts.dataset.y.sum()),
            "positive_rate": hosts.dataset.positive_rate,
            "n_train": len(split.y_train),
            "n_test": 0,
        },
        "training": summary.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    bench = run_toi_benchmark(
        args,
        config,
        trained,
        {t.tic for t in hosts.targets},
        importance_repeats=10,
    )
    calibration = scored_calibration(bench["stars"], trained.calibration)
    payload["toi_benchmark"] = benchmark_headline(bench) | {"calibration": calibration}
    report = format_training_report(summary, bench, calibration)

    hosts.dataset.save(results_dir / "dataset.npz")
    written = [results_dir / "metrics.json", results_dir / "report.txt"]
    if args.pixel_features:
        # vet computes the light-curve features only, so this model is not saved.
        payload["model"] = None
    else:
        model_path = save_model(
            trained,
            split,
            results_dir / "model.joblib",
            preprocess=config.preprocess,
            bls=config.bls,
            provenance={"seed": config.seed, "source": source},
        )
        payload["model"] = _display_path(model_path)
        written.append(model_path)
    payload["runtime_seconds"] = round(time.time() - started, 1)
    (results_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    (results_dir / "report.txt").write_text(report + "\n")
    print(f"\nwrote {', '.join(_display_path(p) for p in written)}")
    print(f"total runtime: {payload['runtime_seconds']}s")
    return 0


def pick_example_indices(dataset: Dataset, config: Config) -> list[int]:
    """One representative planet, eclipsing binary and variable star for the figure.

    The planet chosen is a median-SNR one rather than the strongest, so the
    figure is not quietly flattering.
    """
    kinds = dataset.meta["kind"].astype(str).to_numpy()
    snr = dataset.meta["true_snr"].astype(float).to_numpy()

    planets = np.flatnonzero(kinds == "planet")
    planet = int(planets[np.argsort(snr[planets])[len(planets) // 2]]) if planets.size else 0

    binaries = np.flatnonzero(kinds == "eclipsing_binary")
    binary = int(binaries[0]) if binaries.size else 1

    variability = dataset.meta["variability_amplitude"].astype(float).to_numpy()
    noise_rows = np.flatnonzero(kinds == "noise")
    quiet = (
        int(noise_rows[np.argmax(variability[noise_rows])]) if noise_rows.size else 2
    )
    return [planet, binary, quiet]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_overrides(default_config(), args)
    np.random.seed(config.seed)
    if args.bls_engine is not None:
        # Set before any worker starts, so every process inherits it.
        os.environ[BLS_ENGINE_ENV] = args.bls_engine

    started = time.time()
    print(
        f"transit-detection | seed={config.seed} | python {platform.python_version()} | "
        f"search {config.bls.search}"
        + (f" ({bls_engine()} BLS engine)" if config.bls.search == "bls" else "")
    )
    if args.train_sectors is not None:
        return run_toi_training(args, config, started)

    what = "real light curves (injected)" if args.inject_into else "light curves"
    source = build_source(config, args)
    print(
        f"generating and searching {len(source)} {what} "
        f"({config.dataset.positive_rate:.1%} planets, "
        f"{config.dataset.eclipsing_binary_rate:.1%} eclipsing binaries) ..."
    )

    dataset = build_dataset(
        source, preprocess=config.preprocess, bls=config.bls, n_jobs=args.n_jobs
    )
    t_data = time.time() - started
    print(
        f"  -> {len(dataset)} curves, {int(dataset.y.sum())} planets "
        f"({dataset.positive_rate:.2%} positive) in {t_data:.1f}s"
    )

    split = make_split(dataset, test_size=config.dataset.test_size, seed=config.seed)
    print(
        f"split: {len(split.y_train)} train ({split.n_train_positive} planets) / "
        f"{len(split.y_test)} test ({split.n_test_positive} planets)"
    )

    trained = train(
        split,
        n_folds=config.dataset.n_cv_folds,
        seed=config.seed,
        target_precision=config.evaluation.target_precision,
        precision_lcb_z=config.evaluation.precision_lcb_z,
    )
    print(
        f"trained; operating threshold {trained.threshold:.4f} "
        f"({trained.threshold_rule})"
    )

    result = evaluate(
        dataset, split, trained, top_k=config.evaluation.top_k, seed=config.seed
    )
    report = format_report(result)
    print()
    print(report)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": config.to_dict(),
        "runtime_seconds": None,
        "dataset": {
            "source": source.name,
            "n_curves": len(dataset),
            "n_planets": int(dataset.y.sum()),
            "positive_rate": dataset.positive_rate,
            "n_train": int(len(split.y_train)),
            "n_test": int(len(split.y_test)),
        },
        "results": result.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    if not args.no_figures:
        curves = [source.generate(i) for i in pick_example_indices(dataset, config)]
        paths = plot_all(
            dataset, split, trained, result, curves, config, Path(args.figures_dir)
        )
        sector = getattr(source, "sector", None)
        if sector is not None:
            paths.append(
                plot_sector_systematics(
                    sector, Path(args.figures_dir) / "05_sector_systematics.png"
                )
            )
        payload["figures"] = [_display_path(p) for p in paths]
        print("figures: " + ", ".join(_display_path(p) for p in paths))

    if args.benchmark_tois is not None:
        bench = run_toi_benchmark(args, config, trained, training_tic_ids(source))
        payload["toi_benchmark"] = benchmark_headline(bench)

    dataset.save(results_dir / "dataset.npz")
    model_path = save_model(
        trained,
        split,
        results_dir / "model.joblib",
        preprocess=config.preprocess,
        bls=config.bls,
        provenance={"seed": config.seed, "source": source.name},
    )
    payload["model"] = _display_path(model_path)
    payload["runtime_seconds"] = round(time.time() - started, 1)
    (results_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    (results_dir / "report.txt").write_text(report + "\n")

    print(
        f"\nwrote {results_dir / 'metrics.json'}, {results_dir / 'report.txt'} "
        f"and {model_path}"
    )
    print(f"total runtime: {payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
