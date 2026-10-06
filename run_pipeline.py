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

Benchmark against real labels (either mode; needs ``lightkurve`` the first
time)::

    python run_pipeline.py ... --benchmark-tois exofop_toi.csv --benchmark-sectors 14-26

scores the trained model, unchanged, on TOI hosts whose follow-up disposition
is known (CP/KP planets, FP/FA false positives) and writes
``toi_benchmark.json`` and ``toi_benchmark.txt`` beside the metrics.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from transitml.config import Config, default_config
from transitml.data.base import LightCurveSource
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
from transitml.data.synthetic import SyntheticTESSSource
from transitml.data.toi import parse_sector_spec, read_toi_table, select_benchmark_targets
from transitml.evaluate import evaluate, format_report
from transitml.model import TrainedModel, make_split, save_model, train
from transitml.plots import plot_all

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
    args = parser.parse_args(argv)
    if args.inject_into is not None:
        if args.results_dir == ROOT / "results":
            args.results_dir = ROOT / "results" / "real_injection"
        if args.figures_dir == ROOT / "figures":
            args.figures_dir = ROOT / "figures" / "real_injection"
        if args.curve_cache is None:
            args.curve_cache = args.results_dir / "base_curves.npz"
    if args.benchmark_tois is not None and args.benchmark_cache is None:
        args.benchmark_cache = args.results_dir / "toi_curves.npz"
    if args.benchmark_tois is not None and args.benchmark_stars is None:
        args.benchmark_stars = args.benchmark_tois.parent / "tic_stars.csv"
    return args


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Return a config with the command-line overrides applied."""
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.n_curves is not None:
        config = replace(config, dataset=replace(config.dataset, n_curves=args.n_curves))
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


def run_toi_benchmark(
    args: argparse.Namespace,
    config: Config,
    trained: TrainedModel,
    source: LightCurveSource,
) -> dict[str, Any]:
    """Score ``trained`` on labelled TOI hosts; write the report, JSON and figure."""
    from transitml.benchmark import (
        benchmark,
        build_benchmark_dataset,
        format_benchmark_report,
        load_or_fetch_curves,
        plot_benchmark,
        with_tic_stars,
    )

    spec = args.benchmark_sectors or str(args.sector if args.sector is not None else 14)
    sectors = parse_sector_spec(spec)
    targets, selection = select_benchmark_targets(
        read_toi_table(args.benchmark_tois), sectors, exclude_tics=training_tic_ids(source)
    )
    print(
        f"\nTOI benchmark: {selection['positives']} CP/KP and {selection['negatives']} FP/FA "
        f"hosts in sectors {spec} ({selection['in_training_set']} dropped as training stars)"
    )
    curves = load_or_fetch_curves(
        targets,
        args.benchmark_cache,
        author=args.author,
        exposure_time=args.exposure_time,
        n_workers=args.download_workers,
    )
    curves = with_tic_stars(curves, args.benchmark_stars)
    n_known = sum(1 for lc in curves if all(np.isfinite(lc.star)))
    print(
        f"  {len(curves)} have a light curve, {n_known} a TIC temperature or density; "
        "searching them ..."
    )
    dataset = build_benchmark_dataset(
        curves, preprocess=config.preprocess, bls=config.bls, n_jobs=args.n_jobs
    )
    result = benchmark(
        dataset,
        targets,
        trained,
        sectors=sectors,
        selection=selection,
        n_without_curve=len(targets) - len(curves),
        top_k=config.evaluation.top_k,
        seed=config.seed,
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

    started = time.time()
    print(f"transit-detection | seed={config.seed} | python {platform.python_version()}")
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
        payload["figures"] = [_display_path(p) for p in paths]
        print("figures: " + ", ".join(_display_path(p) for p in paths))

    if args.benchmark_tois is not None:
        bench = run_toi_benchmark(args, config, trained, source)
        payload["toi_benchmark"] = {
            k: bench[k] for k in ("n_stars", "n_planets", "chance_average_precision")
        } | {
            "average_precision": bench["model"]["average_precision"],
            "planet_recall": bench["operating_point"]["planet_recall"],
            "false_positive_rejection": bench["operating_point"]["false_positive_rejection"],
        }

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
