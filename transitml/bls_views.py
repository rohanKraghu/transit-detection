"""Folded views at the pipeline's own BLS ephemeris: the view model as a vetter.

    python -m transitml.bls_views --train-sectors 1-13,27-102 --test-sectors 14-26

:mod:`transitml.toi_views` folds each TOI at its catalogue ephemeris, which a
vetter of a new star's candidates does not have, and about half of its lead
over the pipeline's features went when the stars whose period BLS missed were
left out.  This module folds every TOI host instead at the ephemeris the
pipeline's own search found on it, the BLS peak its 23 features and its
centroid test are measured at, so the views and the features see the same
signal and the comparison is between inputs alone.

What is trained
---------------
One row per TOI host, labelled as the pipeline labels it (a confirmed or known
planet on the star against false positives and false alarms only), with the
host's curve detrended, searched and featurised exactly as ``run_pipeline.py
--train-sectors`` does it.  Six input sets, each given to the same boosting
model as in :mod:`transitml.toi_views` (the mean of five fits):

* ``features``: the pipeline's 23 features, a control for the booster.
* ``features_pixels``: plus the centroid test at the BLS ephemeris.
* ``views``: the five views at the BLS ephemeris, with log period and
  duration, the curve detrended with only that signal's transits masked.
* ``views_depth``: plus the depth the views were divided by and the scatter.
* ``views_depth_pixels``: plus the centroid test.
* ``everything``: plus the pipeline's 23 features.

Stars are chosen as ``run_pipeline.py --train-sectors`` chooses them, and the
pipeline's own models (from their ``toi_benchmark.json``) are compared on the
same stars, so the first two sets separate what the booster adds from what
the views add.
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
from numpy.typing import NDArray

from .config import Config, default_config
from .data.base import LightCurve
from .data.toi import BenchmarkTarget
from .kepler_dr25 import SCALAR_NAMES, TrainingSet
from .tess_finetune import compare_all
from .toi_views import mean_of_fits, placed_fraction, read_run
from .toi_views import model_inputs as view_inputs
from .views import (
    VIEW_NAMES,
    Ephemeris,
    ViewConfig,
    detrend_masked,
    in_transit_mask,
    knot_spacing_for,
    make_views,
)

#: The input sets: two controls on the pipeline's features, then the views,
#: each adding to the last.
INPUT_SETS: tuple[str, ...] = (
    "features",
    "features_pixels",
    "views",
    "views_depth",
    "views_depth_pixels",
    "everything",
)

#: The sets that read the centroid test.
PIXEL_SETS: tuple[str, ...] = ("features_pixels", "views_depth_pixels", "everything")

#: The model whose operating point the report gives.
HEADLINE: str = "everything"

_LABELS = {
    "features": "pipeline features",
    "features_pixels": "pipeline features + centroid test",
    "views": "views",
    "views_depth": "views + depth and scatter",
    "views_depth_pixels": "views + depth and scatter + centroid test",
    "everything": "views, depth, centroid test and pipeline features",
    "pipeline": "the pipeline's model",
    "pipeline_pixels": "the pipeline's model + centroid test",
}


def host_views(
    lc: LightCurve, ephemeris: Ephemeris, config: ViewConfig
) -> tuple[dict[str, NDArray[np.float32]], dict[str, float]]:
    """The five views of one host at ``ephemeris``, and the scalars that go with them.

    The curve is detrended with that signal's transits masked, as
    :func:`~transitml.tess_transfer.star_views` does with a TOI's.  Raises
    ``ValueError`` when there is no data near transit.
    """
    lc = lc.finite()
    mask = in_transit_mask(lc.time, [ephemeris], config.mask_half_width)
    flux = detrend_masked(
        lc.time, lc.flux, mask, knot_spacing_for([ephemeris], config), config.gap_threshold_days
    )
    views = make_views(lc.time, flux, ephemeris, config)
    scalars = {
        "period_days": ephemeris.period,
        "duration_hours": 24.0 * ephemeris.duration,
        "depth_ppm": np.nan,
        "mes": np.nan,
        "depth_scale": views.depth_scale,
        "scatter": views.scatter,
        "local_empty_fraction": views.local_empty_fraction,
        "n_cadences": float(lc.n_cadences),
    }
    return views.as_dict(), scalars


@dataclass
class Hosts:
    """TOI hosts searched as the pipeline searches them, one row each."""

    #: Views at each host's BLS ephemeris; ``kepids`` holds the TIC ID and
    #: ``labels`` the host's label.
    views: TrainingSet
    #: The pipeline's features of the same rows.
    features: NDArray[np.float64]
    #: The centroid test at the same ephemeris
    #: (:func:`~transitml.benchmark.centroid_features`), NaN without a pixel file.
    pixels: NDArray[np.float64]
    #: Hosts searched but dropped for having no data near the BLS transit.
    n_without_views: int = 0

    def __len__(self) -> int:
        return len(self.features)


def search_hosts(
    curves: Sequence[LightCurve],
    targets: Sequence[BenchmarkTarget],
    tpf_dir: str | Path,
    *,
    stars: str | Path | None = None,
    config: Config | None = None,
    view_config: ViewConfig | None = None,
    n_jobs: int = -1,
) -> Hosts:
    """Search, featurise, fold and centroid-test every host, as the pipeline would.

    ``stars`` is the TIC table the pipeline reads (temperature and density
    size the secondary test's allowance for a planet's occultation); ``None``
    leaves the curves as they are.  Each host's pixel file is read from
    ``tpf_dir`` (:func:`~transitml.benchmark.tpf_cache_path`) and never
    downloaded.
    """
    from .benchmark import (
        build_benchmark_dataset,
        centroid_features,
        centroid_tests,
        tpf_cache_path,
        with_tic_stars,
    )

    config = config or default_config()
    view_config = view_config or ViewConfig()
    by_id = {t.target_id: t for t in targets}
    if stars is not None:
        curves = with_tic_stars(curves, stars)
    dataset = build_benchmark_dataset(
        curves, preprocess=config.preprocess, bls=config.bls, n_jobs=n_jobs
    )
    paths = {
        t.target_id: path
        for t in targets
        if (path := tpf_cache_path(tpf_dir, t.target_id, t.sector)).exists()
    }
    pixels = centroid_features(centroid_tests(dataset, paths, n_jobs=n_jobs)).to_numpy()
    ephemerides = dataset.meta[["search_period", "search_epoch", "search_duration"]].to_numpy(float)
    features = dataset.features.to_numpy(dtype=np.float64)

    kept, rows = [], []
    for i, lc in enumerate(curves):
        if dataset.meta["target_id"].iloc[i] != lc.target_id:
            raise RuntimeError("searched rows are out of step with the curves")
        try:
            views, scalars = host_views(lc, Ephemeris(*ephemerides[i]), view_config)
        except ValueError:
            continue
        kept.append(i)
        rows.append((by_id[lc.target_id], views, scalars))
    ts = TrainingSet(
        tce_ids=np.array([t.target_id for t, _, _ in rows], dtype=str),
        kepids=np.array([t.tic for t, _, _ in rows], dtype=np.int64),
        classes=np.array([t.reference.disposition or "" for t, _, _ in rows], dtype=str),
        labels=np.array([int(t.label) for t, _, _ in rows], dtype=np.int64),
        views={
            k: np.stack([v[k] for _, v, _ in rows])
            if rows
            else np.empty(
                (0, view_config.global_bins if k == "global" else view_config.local_bins), np.float32
            )
            for k in VIEW_NAMES
        },
        scalars={k: np.array([s[k] for _, _, s in rows], dtype=np.float64) for k in SCALAR_NAMES},
    )
    return Hosts(
        views=ts,
        features=features[kept],
        pixels=pixels[kept],
        n_without_views=len(curves) - len(kept),
    )


def model_inputs(hosts: Hosts, kind: str) -> NDArray[np.float64]:
    """The model input for one of :data:`INPUT_SETS`, one row per host."""
    if kind not in INPUT_SETS:
        raise ValueError(f"unknown input set {kind!r}; choose from {INPUT_SETS}")
    if kind == "features":
        return hosts.features
    if kind == "features_pixels":
        return np.hstack([hosts.features, hosts.pixels])
    if kind == "everything":
        return np.hstack([view_inputs(hosts.views, hosts.pixels, "views_depth_pixels"), hosts.features])
    return view_inputs(hosts.views, hosts.pixels, kind)


def usable_input_sets(train_pixels: NDArray[np.float64]) -> tuple[str, ...]:
    """:data:`INPUT_SETS`, less those reading a centroid column empty in training."""
    if len(train_pixels) and np.isfinite(train_pixels).any(axis=0).all():
        return INPUT_SETS
    return tuple(k for k in INPUT_SETS if k not in PIXEL_SETS)


def comparison_pairs(compared: Sequence[str], other_runs: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """The differences the report bootstraps.

    The booster against the pipeline's own models on the same inputs, the
    views against the features at each step, each view set against the one
    before it, and each set against the same set of every other run.
    """
    pairs = [(kind, name) for kind, name in zip(("features", "features_pixels"), compared)]
    pairs += [
        ("views", "features"),
        ("views_depth_pixels", "features_pixels"),
        ("everything", "features_pixels"),
    ]
    pairs += [(b, a) for a, b in pairwise(INPUT_SETS[2:])]
    pairs += [(kind, f"{prefix}_{kind}") for prefix in other_runs for kind in INPUT_SETS]
    return tuple(pairs)


def compare_hosts(
    train: Hosts,
    test: Hosts,
    pipeline_runs: dict[str, list[dict[str, Any]]],
    *,
    other_runs: dict[str, dict[str, dict[int, float]]] | None = None,
    seed: int = 0,
    n_fits: int = 5,
) -> dict[str, Any]:
    """Train every input set on ``train``, score ``test``, and compare with the other models.

    ``pipeline_runs`` maps a pipeline model's name to the per-star rows of
    its ``toi_benchmark.json``, the model without the centroid test first;
    ``other_runs`` maps a prefix to another run's ``{input set: {tic: score}}``.
    """
    other_runs = other_runs or {}
    star_scores: dict[str, dict[int, float]] = {}
    for kind in usable_input_sets(train.pixels):
        scores = mean_of_fits(
            model_inputs(train, kind),
            train.views.labels,
            model_inputs(test, kind),
            seed=seed,
            n_fits=n_fits,
        )
        star_scores[kind] = {int(tic): float(s) for tic, s in zip(test.views.kepids, scores)}
    for prefix, run in other_runs.items():
        for kind, scores in run.items():
            if scores:
                star_scores[f"{prefix}_{kind}"] = scores
    return compare_all(
        star_scores,
        pipeline_runs,
        pairs=comparison_pairs(list(pipeline_runs), list(other_runs)),
        operating_model=HEADLINE,
        seed=seed,
    )


@dataclass
class HostViewsResult:
    """The host models on the benchmark stars, beside the pipeline's."""

    train_sectors: str
    test_sectors: str
    n_train_hosts: int
    n_train_planets: int
    n_train_from_later_sector: int
    n_without_views: dict[str, int]
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


def _label(name: str) -> str:
    """A model's name for the report: ``one_year_views`` reads "views (one year)"."""
    if name in _LABELS:
        return _LABELS[name]
    for kind in sorted(INPUT_SETS, key=len, reverse=True):
        if name.endswith(f"_{kind}"):
            return f"{_LABELS[kind]} ({name[: -len(kind) - 1].replace('_', ' ')})"
    return name


def format_host_views_report(r: HostViewsResult) -> str:
    sub = r.recovered_subset
    lines = [
        "TOI hosts folded at the pipeline's own BLS ephemeris",
        "=" * 72,
        "",
        (
            f"Trained on {r.n_train_hosts} TOI hosts of sectors {r.train_sectors} "
            f"({r.n_train_planets} planet hosts"
        )
        + (
            f"; {r.n_train_from_later_sector} taken from a later sector than their first)."
            if r.n_train_from_later_sector
            else ")."
        ),
        (
            f"Scored on {r.n_stars} TOI hosts of sectors {r.test_sectors} ({r.n_planets} CP/KP "
            f"hosts, {r.n_stars - r.n_planets} FP/FA-only hosts), no star shared with training."
        ),
        (
            f"Hosts without data near their BLS transit, left out: {r.n_without_views['train']} "
            f"in training, {r.n_without_views['test']} scored."
        ),
        (
            "Centroid test at the BLS ephemeris placed the dip for "
            f"{r.centroid_placed['train']:.0%} of training hosts and "
            f"{r.centroid_placed['test']:.0%} of scored ones."
        ),
        (
            f"Each model here is the mean of {r.n_fits} boosting fits (seeds {r.seed} to "
            f"{r.seed + r.n_fits - 1})."
        ),
        "",
        f"Average precision, all stars (chance {r.chance:.3f})",
    ]
    for name, ap in sorted(r.average_precision.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {_label(name):<62} {ap:.3f}")
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
        lines.append(f"  {_label(name):<62} {ap:.3f}")
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
        "Every model here sees the ephemeris the pipeline's search found, as a",
        "vetter of a new star would.  See transitml.bls_views.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.bls_views",
        description="Boosting on views folded at the pipeline's own BLS ephemeris, beside the "
        "pipeline's features, trained on one set of sectors' TOI hosts and scored on another's.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--toi-table", type=Path, default=Path("data/toi_benchmark/exofop_toi_2026-10-06.csv"))
    parser.add_argument(
        "--stars",
        type=Path,
        default=None,
        help="TIC temperatures and densities (default: tic_stars.csv beside --toi-table)",
    )
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
        help="the pipeline's toi_benchmark.json on the test sectors, without the centroid test "
        "and then with it (default: the TOI-trained models of results/toi_trained/)",
    )
    parser.add_argument(
        "--compare-run",
        nargs=2,
        action="append",
        metavar=("PREFIX", "DIR"),
        help="another run of this module on the same test sectors, compared input set by input set",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/bls_views"))
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

    args = parse_args(argv)
    stars = args.stars or args.toi_table.parent / "tic_stars.csv"
    tpf_dir = args.tpfs or args.cache.parent / "toi_tpfs"
    compare = args.compare or [
        ["pipeline", "results/toi_trained/toi_benchmark.json"],
        ["pipeline_pixels", "results/toi_trained/pixels/toi_benchmark.json"],
    ]
    pipeline_runs = {name: json.loads(Path(path).read_text())["stars"] for name, path in compare}
    other_runs = {prefix: read_run(path, INPUT_SETS) for prefix, path in args.compare_run or []}
    tois = read_toi_table(args.toi_table)
    test_sectors = parse_sector_spec(args.test_sectors)
    train_sectors = parse_sector_spec(args.train_sectors)

    def fetch(targets: list[BenchmarkTarget]) -> list[LightCurve]:
        return load_or_fetch_curves(targets, args.cache, n_workers=args.download_workers)

    first, _ = select_benchmark_targets(
        tois, train_sectors, exclude_tics=observed_in(tois, test_sectors)
    )
    train_targets, train_curves = fetch_with_fallback(first, train_sectors, fetch)
    have = {lc.target_id for lc in train_curves}
    n_moved = sum(1 for a, b in zip(first, train_targets) if b != a and b.target_id in have)
    test_targets, _ = select_benchmark_targets(
        tois, test_sectors, exclude_tics={t.tic for t in train_targets}
    )
    test_curves = fetch(test_targets)
    print(
        f"searching {len(train_curves)} training hosts and {len(test_curves)} benchmark hosts "
        "as the pipeline does ..."
    )
    train = search_hosts(train_curves, train_targets, tpf_dir, stars=stars, n_jobs=args.n_jobs)
    test = search_hosts(test_curves, test_targets, tpf_dir, stars=stars, n_jobs=args.n_jobs)
    if "views_depth_pixels" not in usable_input_sets(train.pixels):
        print(f"  no centroid results on the training hosts (pixel files in {tpf_dir}?); "
              "leaving out the pixel models")

    cmp = compare_hosts(train, test, pipeline_runs, other_runs=other_runs, seed=args.seed, n_fits=args.fits)
    result = HostViewsResult(
        train_sectors=args.train_sectors,
        test_sectors=args.test_sectors,
        n_train_hosts=len(train),
        n_train_planets=int(train.views.labels.sum()),
        n_train_from_later_sector=n_moved,
        n_without_views={"train": train.n_without_views, "test": test.n_without_views},
        centroid_placed={"train": placed_fraction(train.pixels), "test": placed_fraction(test.pixels)},
        n_fits=args.fits,
        seed=args.seed,
        **cmp,
    )
    report = format_host_views_report(result)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "report.txt").write_text(report)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str) + "\n"
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
