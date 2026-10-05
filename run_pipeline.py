#!/usr/bin/env python3
"""One command: generate data, detrend, search, train, evaluate, plot.

    python run_pipeline.py

Writes ``results/metrics.json``, ``results/report.txt``, ``results/dataset.npz``
and four PNGs to ``figures/``.  Deterministic given ``--seed``; the default run
takes ~2 minutes on four cores.
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
from transitml.data.loader import Dataset, build_default_dataset, sample_light_curves
from transitml.evaluate import evaluate, format_report
from transitml.model import make_split, train
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
    return parser.parse_args(argv)


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Return a config with the command-line overrides applied."""
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.n_curves is not None:
        config = replace(config, dataset=replace(config.dataset, n_curves=args.n_curves))
    return config


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
    print(
        f"generating and searching {config.dataset.n_curves} light curves "
        f"({config.dataset.positive_rate:.1%} planets, "
        f"{config.dataset.eclipsing_binary_rate:.1%} eclipsing binaries) ..."
    )

    dataset = build_default_dataset(config, n_jobs=args.n_jobs)
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
        curves = sample_light_curves(config, pick_example_indices(dataset, config))
        paths = plot_all(
            dataset, split, trained, result, curves, config, Path(args.figures_dir)
        )
        payload["figures"] = [_display_path(p) for p in paths]
        print("figures: " + ", ".join(_display_path(p) for p in paths))

    dataset.save(results_dir / "dataset.npz")
    payload["runtime_seconds"] = round(time.time() - started, 1)
    (results_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    (results_dir / "report.txt").write_text(report + "\n")

    print(f"\nwrote {results_dir / 'metrics.json'} and {results_dir / 'report.txt'}")
    print(f"total runtime: {payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
