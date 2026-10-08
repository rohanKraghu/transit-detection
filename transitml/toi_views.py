"""Folded views, the centroid test and every labelled TOI host, in one model.

    python -m transitml.toi_views --train-sectors 1-13,27-102 --test-sectors 14-26

Three results point here.  Boosting on the folded views of each TOI
(:mod:`transitml.tess_finetune`) beats boosting on the pipeline's 23 features
when both learn from the same TOI labels.  The centroid test, as three more
features, lifts the pipeline's model (:mod:`transitml.toi_training`).  And the
labelled hosts of later sectors help it too (``run_pipeline.py --train-sectors
1-13,27-102``).  This module asks whether the three add up.

What is trained
---------------
One row per labelled TOI on a training star (confirmed and known planets
against false positives and false alarms), folded into the five views at the
TOI's catalogue ephemeris exactly as :mod:`transitml.tess_transfer` does it.
Three sets of inputs, each given to the same boosting model:

* ``views``: the views with log period and duration, the ``tess_gbm`` model
  of :mod:`transitml.tess_finetune`.
* ``views_depth``: plus the depth the views were divided by and the
  out-of-transit scatter, which the views' common scale takes out.
* ``views_depth_pixels``: plus the centroid test, run at the same catalogue
  ephemeris on the star's target pixel file: the dip's offset from the target
  in sigma and in pixels, and the difference-image SNR (NaN where there is no
  file, and for the offsets where the difference image did not place the dip).

Each model is the mean of several fits whose early-stopping splits differ
(five by default): a single fit's average precision moves by about 0.007 with
its seed, as much as some of the differences being measured.

Stars
-----
Chosen as ``run_pipeline.py --train-sectors`` chooses them: a star observed in
any benchmark sector is never trained on, a training star with no light curve
in its first sector is taken from its next one, and a benchmark star is scored
in its first.  A benchmark star scores as its highest TOI, open candidates
included, so which TOI is scored never depends on the label, and every model
is compared on the stars all of them scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from .benchmark import _centroid_one, centroid_features, tpf_cache_path
from .centroid import CentroidConfig
from .data.toi import BenchmarkTarget
from .kepler_dr25 import TrainingSet, view_features
from .tess_finetune import compare_all, fit_gbm
from .tess_transfer import star_max, toi_ephemeris

#: The input sets, each one adding to the last.
INPUT_SETS: tuple[str, ...] = ("views", "views_depth", "views_depth_pixels")

#: What ``views_depth`` adds: the scale the views were divided by, and the noise.
DEPTH_NAMES: tuple[str, ...] = ("depth_scale", "scatter")

#: The model whose operating point the report gives.
HEADLINE: str = "views_depth_pixels"

_LABELS = {
    "views": "views",
    "views_depth": "views + depth and scatter",
    "views_depth_pixels": "views + depth and scatter + centroid test",
    "features": "pipeline features",
    "features_pixels": "pipeline features + centroid test",
}


def catalogue_centroid_tests(
    ts: TrainingSet,
    targets: Sequence[BenchmarkTarget],
    tpf_dir: str | Path,
    *,
    config: CentroidConfig | None = None,
    n_jobs: int = -1,
) -> list[dict[str, Any] | None]:
    """The centroid test of every row of ``ts``, at its TOI's catalogue ephemeris.

    Each row is tested on the pixel file of its star's own sector
    (:func:`~transitml.benchmark.tpf_cache_path` in ``tpf_dir``).  ``None``
    for a row whose star has no file there or whose TOI lacks an ephemeris.
    """
    by_tic = {t.tic: t for t in targets}
    jobs = []
    for i, (tic, toi_id) in enumerate(zip(ts.kepids.tolist(), ts.tce_ids.tolist())):
        target = by_tic.get(tic)
        if target is None:
            continue
        toi = next((t for t in target.tois or (target.reference,) if t.toi == toi_id), None)
        ephemeris = toi_ephemeris(toi) if toi is not None else None
        path = tpf_cache_path(tpf_dir, target.target_id, target.sector)
        if ephemeris is not None and path.exists():
            jobs.append((i, path, ephemeris))
    done = Parallel(n_jobs=n_jobs)(
        delayed(_centroid_one)(path, e.period, e.epoch, e.duration, config) for _, path, e in jobs
    )
    out: list[dict[str, Any] | None] = [None] * len(ts)
    for (i, _, _), result in zip(jobs, done):
        out[i] = result
    return out


def model_inputs(ts: TrainingSet, pixels: NDArray[np.float64], kind: str) -> NDArray[np.float64]:
    """The model input for one of :data:`INPUT_SETS`, one row per row of ``ts``.

    ``pixels`` holds :func:`~transitml.benchmark.centroid_features` of the
    same rows, and is read only by ``views_depth_pixels``.
    """
    if kind not in INPUT_SETS:
        raise ValueError(f"unknown input set {kind!r}; choose from {INPUT_SETS}")
    parts = [view_features(ts)]
    if kind != "views":
        parts.append(np.column_stack([ts.scalars[name] for name in DEPTH_NAMES]))
    if kind == "views_depth_pixels":
        parts.append(np.asarray(pixels, dtype=np.float64))
    return np.hstack(parts)


def mean_of_fits(
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    X_test: NDArray[np.float64],
    *,
    seed: int,
    n_fits: int,
) -> NDArray[np.float64]:
    """P(planet) on ``X_test``, averaged over ``n_fits`` boosting fits seeded from ``seed``."""
    return np.mean(
        [fit_gbm(X, y, seed + k).predict_proba(X_test)[:, 1] for k in range(n_fits)], axis=0
    )


@dataclass
class ViewsResult:
    """The view models on the benchmark stars, beside the models they are compared with."""

    train_sectors: str
    test_sectors: str
    n_train_tois: int
    n_train_stars: int
    n_train_planets: int
    n_train_from_later_sector: int
    n_test_tois: int
    centroid_placed: dict[str, float]
    n_fits: int
    seed: int
    n_stars: int
    n_planets: int
    chance: float
    average_precision: dict[str, float]
    recovered_subset: dict[str, Any]
    differences: dict[str, tuple[float, float, float]]
    operating_point: dict[str, float]
    stars: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["differences"] = {k: list(v) for k, v in self.differences.items()}
        return out


def usable_input_sets(train_pixels: NDArray[np.float64]) -> tuple[str, ...]:
    """:data:`INPUT_SETS`, less the pixel model if a centroid column is empty in training."""
    if len(train_pixels) and np.isfinite(train_pixels).any(axis=0).all():
        return INPUT_SETS
    return tuple(k for k in INPUT_SETS if k != "views_depth_pixels")


def comparison_pairs(compared: Sequence[str], other_runs: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """The differences the report bootstraps.

    Each input set against the one before it, each against every compared
    benchmark model, and each against the same inputs of every other run.
    """
    pairs = [(b, a) for a, b in pairwise(INPUT_SETS)]
    pairs += [(kind, name) for name in compared for kind in INPUT_SETS]
    pairs += [(kind, f"{prefix}_{kind}") for prefix in other_runs for kind in INPUT_SETS]
    return tuple(pairs)


def compare_views(
    train_set: TrainingSet,
    train_pixels: NDArray[np.float64],
    test_set: TrainingSet,
    test_pixels: NDArray[np.float64],
    tess_runs: dict[str, list[dict[str, Any]]],
    *,
    other_runs: dict[str, dict[str, dict[int, float]]] | None = None,
    seed: int = 0,
    n_fits: int = 5,
) -> dict[str, Any]:
    """Train every input set, score the test stars, and compare with the other models.

    ``tess_runs`` maps a benchmark model's name to the per-star rows of its
    ``toi_benchmark.json``; ``other_runs`` maps a prefix to another run's
    ``{input set: {tic: score}}``, compared as ``<prefix>_<input set>``.
    The pixel model is left out when a centroid column has no value on any
    training row (no pixel files), since nothing can be learned from it.
    """
    other_runs = other_runs or {}
    star_scores: dict[str, dict[int, float]] = {}
    for kind in usable_input_sets(train_pixels):
        scores = mean_of_fits(
            model_inputs(train_set, train_pixels, kind),
            train_set.labels,
            model_inputs(test_set, test_pixels, kind),
            seed=seed,
            n_fits=n_fits,
        )
        star_scores[kind] = star_max(test_set.kepids, scores)
    for prefix, run in other_runs.items():
        for kind, scores in run.items():
            if scores:
                star_scores[f"{prefix}_{kind}"] = scores
    return compare_all(
        star_scores,
        tess_runs,
        pairs=comparison_pairs(list(tess_runs), list(other_runs)),
        operating_model=HEADLINE,
        seed=seed,
    )


def placed_fraction(pixels: NDArray[np.float64]) -> float:
    """Fraction of rows whose centroid test placed the dip (a finite offset)."""
    return float(np.isfinite(pixels[:, 0]).mean()) if len(pixels) else float("nan")


def _label(name: str) -> str:
    """A model's name for the report: ``later_views_depth`` reads "views + depth and scatter (later)"."""
    if name in _LABELS:
        return _LABELS[name]
    for kind in sorted(INPUT_SETS, key=len, reverse=True):
        if name.endswith(f"_{kind}"):
            return f"{_LABELS[kind]} ({name[: -len(kind) - 1].replace('_', ' ')})"
    return name


def format_views_report(r: ViewsResult) -> str:
    sub = r.recovered_subset
    lines = [
        "TOI hosts as folded views, with depth and the centroid test",
        "=" * 72,
        "",
        (
            f"Trained on {r.n_train_tois} labelled TOIs on {r.n_train_stars} stars of sectors "
            f"{r.train_sectors} ({r.n_train_planets} planet TOIs"
        )
        + (
            f"; {r.n_train_from_later_sector} stars taken from a later sector than their first)."
            if r.n_train_from_later_sector
            else ")."
        ),
        (
            f"Scored on {r.n_stars} TOI hosts of sectors {r.test_sectors} ({r.n_planets} CP/KP "
            f"hosts, {r.n_stars - r.n_planets} FP/FA-only hosts; {r.n_test_tois} TOIs folded), "
            "no star shared with training."
        ),
        (
            "Centroid test at the catalogue ephemeris placed the dip for "
            f"{r.centroid_placed['train']:.0%} of training TOIs and "
            f"{r.centroid_placed['test']:.0%} of scored ones."
        ),
        (
            f"Each view model is the mean of {r.n_fits} boosting fits (seeds {r.seed} to "
            f"{r.seed + r.n_fits - 1}); a star scores as its highest TOI."
        ),
        "",
        f"Average precision, all stars (chance {r.chance:.3f})",
    ]
    for name, ap in sorted(r.average_precision.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {_label(name):<58} {ap:.3f}")
    lines += ["", "Paired bootstrap, difference in AP (95% interval)"]
    for pair, (d, lo, hi) in r.differences.items():
        a, b = pair.split(" - ")
        lines.append(f"  {_label(a)} minus {_label(b)}")
        lines.append(f"      {d:+.3f}  ({lo:+.3f} to {hi:+.3f})")
    lines += [
        "",
        (
            f"Stars where the TESS search found the TOI period ({sub['n_stars']}, "
            f"{sub['n_planets']} planets, chance {sub['chance']:.3f})"
        ),
    ]
    for name, ap in sorted(sub["average_precision"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {_label(name):<58} {ap:.3f}")
    if r.operating_point:
        op = r.operating_point
        lines += [
            "",
            f"{_label(HEADLINE)} at probability {op['probability']:.2f}:",
            f"  planet hosts kept      {op['planets_kept']:.1%}",
            f"  FP/FA hosts rejected   {op['false_positives_rejected']:.1%}",
        ]
    lines += [
        "",
        "The view models are given each TOI's catalogue ephemeris; the pipeline's",
        "models find their own period with BLS.  See transitml.toi_views.",
    ]
    return "\n".join(lines) + "\n"


def read_run(path: str | Path, kinds: Sequence[str] = INPUT_SETS) -> dict[str, dict[int, float]]:
    """Another run's per-star scores, ``{input set: {tic: score}}``, from its metrics.json."""
    stars = json.loads((Path(path) / "metrics.json").read_text())["stars"]
    return {kind: {int(s["tic"]): float(s[kind]) for s in stars if kind in s} for kind in kinds}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.toi_views",
        description="Boosting on folded views of TOIs, with depth and the centroid test, "
        "trained on one set of sectors' TOI labels and scored on another's hosts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--toi-table", type=Path, default=Path("data/toi_benchmark/exofop_toi_2026-10-06.csv"))
    parser.add_argument("--train-sectors", default="1-13")
    parser.add_argument("--test-sectors", default="14-26")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("results/toi_trained/toi_curves.npz"),
        help="light-curve cache, shared with run_pipeline.py --train-sectors",
    )
    parser.add_argument("--tpfs", type=Path, default=None, help="pixel files (default: toi_tpfs/ beside --cache)")
    parser.add_argument(
        "--compare",
        nargs=2,
        action="append",
        metavar=("NAME", "JSON"),
        help="a benchmark model's toi_benchmark.json, scored on the test sectors, to compare with "
        "(default: the TOI-trained models of results/toi_trained/); only its stars are scored",
    )
    parser.add_argument(
        "--compare-run",
        nargs=2,
        action="append",
        metavar=("PREFIX", "DIR"),
        help="another run of this module on the same test sectors, compared input set by input set",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/toi_views"))
    parser.add_argument("--fits", type=int, default=5, help="boosting fits averaged per model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--download-workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from .benchmark import fetch_with_fallback, load_or_fetch_curves
    from .data.toi import (
        observed_in,
        parse_sector_spec,
        read_toi_table,
        select_benchmark_targets,
    )
    from .tess_finetune import drop_stars, labelled
    from .tess_transfer import build_toi_set

    args = parse_args(argv)
    tpf_dir = args.tpfs or args.cache.parent / "toi_tpfs"
    compare = args.compare or [
        ["features", "results/toi_trained/toi_benchmark.json"],
        ["features_pixels", "results/toi_trained/pixels/toi_benchmark.json"],
    ]
    tess_runs = {name: json.loads(Path(path).read_text())["stars"] for name, path in compare}
    other_runs = {prefix: read_run(path) for prefix, path in args.compare_run or []}
    wanted = set.intersection(*({s["target_id"] for s in stars} for stars in tess_runs.values()))
    tois = read_toi_table(args.toi_table)
    test_sectors = parse_sector_spec(args.test_sectors)
    train_sectors = parse_sector_spec(args.train_sectors)

    def fetch(targets: list[BenchmarkTarget]):
        return load_or_fetch_curves(targets, args.cache, n_workers=args.download_workers)

    test_targets, _ = select_benchmark_targets(tois, test_sectors)
    test_targets = [t for t in test_targets if t.target_id in wanted]
    test_set = build_toi_set(fetch(test_targets), test_targets)

    first, _ = select_benchmark_targets(
        tois, train_sectors, exclude_tics=observed_in(tois, test_sectors)
    )
    train_targets, train_curves = fetch_with_fallback(first, train_sectors, fetch)
    have = {lc.target_id for lc in train_curves}
    n_moved = sum(1 for a, b in zip(first, train_targets) if b != a and b.target_id in have)
    train_set = labelled(drop_stars(build_toi_set(train_curves, train_targets), test_set.kepids))
    print(
        f"train: {len(train_set)} labelled TOIs on {np.unique(train_set.kepids).size} stars; "
        f"test: {len(test_set)} TOIs on {np.unique(test_set.kepids).size} stars; "
        "running the centroid tests ..."
    )
    train_pixels = centroid_features(
        catalogue_centroid_tests(train_set, train_targets, tpf_dir, n_jobs=args.n_jobs)
    ).to_numpy()
    test_pixels = centroid_features(
        catalogue_centroid_tests(test_set, test_targets, tpf_dir, n_jobs=args.n_jobs)
    ).to_numpy()
    if HEADLINE not in usable_input_sets(train_pixels):
        print(f"  no centroid results on the training TOIs (pixel files in {tpf_dir}?); "
              "leaving out the pixel model")

    cmp = compare_views(
        train_set,
        train_pixels,
        test_set,
        test_pixels,
        tess_runs,
        other_runs=other_runs,
        seed=args.seed,
        n_fits=args.fits,
    )
    result = ViewsResult(
        train_sectors=args.train_sectors,
        test_sectors=args.test_sectors,
        n_train_tois=len(train_set),
        n_train_stars=int(np.unique(train_set.kepids).size),
        n_train_planets=int(train_set.labels.sum()),
        n_train_from_later_sector=n_moved,
        n_test_tois=len(test_set),
        centroid_placed={"train": placed_fraction(train_pixels), "test": placed_fraction(test_pixels)},
        n_fits=args.fits,
        seed=args.seed,
        **cmp,
    )
    report = format_views_report(result)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "report.txt").write_text(report)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str) + "\n"
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
