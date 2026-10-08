"""Do 34,000 Kepler labels help vet TESS?  The DR25 models on the TOI benchmark.

    python -m transitml.tess_transfer

The models in :mod:`transitml.kepler_dr25` and :mod:`transitml.cnn` learned
transit shape from Kepler's Robovetter labels.  This module scores the TOI
benchmark's TESS stars with them, unchanged, and puts the result beside the
TESS pipeline's own models on the same stars.

How a TESS star is scored
-------------------------
The star's single benchmark sector (30-minute TESS-SPOC cadence, close to
Kepler's 29.4 minutes) is detrended with every TOI on the star masked and
folded into the same five views as a Kepler TCE, at each TOI's catalogue
ephemeris.  The star's score is the **highest** score over all of its TOIs,
open candidates included, so which TOI is scored never depends on the label.

What the comparison can and cannot say
--------------------------------------
* The DR25 models are given each TOI's catalogue period and epoch.  The TESS
  models (:mod:`transitml.benchmark`) find their own period with BLS and miss
  it on some stars.  The report therefore also compares them on the stars
  where the TESS search recovered the TOI period, where both see the same
  signal.
* The labels differ in kind.  Kepler's are Robovetter calls; TESS's are
  follow-up dispositions (confirmed or known planets against false positives
  and false alarms).  Every TOI already passed a vetting step, so the TESS
  negatives are the hard ones, mostly astrophysical false positives.
* One TESS sector holds a few transits; a DR25 TCE folds up to four years.
  The views are noisier here than anything the models were trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data.base import LightCurve
from .data.toi import TOI, BenchmarkTarget
from .evaluate import fast_average_precision
from .kepler_dr25 import SCALAR_NAMES, TrainingSet, _build_classifier, view_features
from .views import (
    VIEW_NAMES,
    Ephemeris,
    ViewConfig,
    detrend_masked,
    in_transit_mask,
    knot_spacing_for,
    make_views,
)

#: TESS times are BJD - 2457000 (BTJD); the TOI table gives full BJD.
BTJD_OFFSET = 2457000.0


def toi_ephemeris(toi: TOI) -> Ephemeris | None:
    """The TOI's ephemeris in BTJD days, or ``None`` if any part is missing."""
    period, epoch, hours = toi.period, toi.epoch_bjd, toi.duration_hours
    if not all(np.isfinite(v) and v > 0 for v in (period, epoch, hours)):
        return None
    if epoch > BTJD_OFFSET:
        epoch -= BTJD_OFFSET
    return Ephemeris(period, epoch, hours / 24.0)


def star_views(
    lc: LightCurve, tois: Sequence[TOI], config: ViewConfig
) -> list[tuple[TOI, dict[str, NDArray[np.float32]], dict[str, float]]]:
    """Views of every TOI on one star that has a usable ephemeris and data near transit."""
    usable = [(t, e) for t in tois if (e := toi_ephemeris(t)) is not None]
    if not usable:
        return []
    lc = lc.finite()
    ephs = [e for _, e in usable]
    mask = in_transit_mask(lc.time, ephs, config.mask_half_width)
    flux = detrend_masked(
        lc.time, lc.flux, mask, knot_spacing_for(ephs, config), config.gap_threshold_days
    )
    out = []
    for toi, eph in usable:
        try:
            views = make_views(lc.time, flux, eph, config)
        except ValueError:
            continue
        scalars = {
            "period_days": toi.period,
            "duration_hours": toi.duration_hours,
            "depth_ppm": toi.depth_ppm,
            "mes": toi.snr,
            "depth_scale": views.depth_scale,
            "scatter": views.scatter,
            "local_empty_fraction": views.local_empty_fraction,
            "n_cadences": float(lc.n_cadences),
        }
        out.append((toi, views.as_dict(), scalars))
    return out


def build_toi_set(
    curves: Sequence[LightCurve], targets: Sequence[BenchmarkTarget], config: ViewConfig | None = None
) -> TrainingSet:
    """One row per TOI with views, in a :class:`TrainingSet`.

    ``kepids`` holds the TIC ID, ``classes`` the TFOPWG disposition, and
    ``labels`` the TOI's own label (-1 for an open candidate).
    """
    config = config or ViewConfig()
    by_id = {t.target_id: t for t in targets}
    rows = []
    for lc in curves:
        target = by_id.get(lc.target_id)
        if target is None:
            continue
        for toi, views, scalars in star_views(lc, target.tois or (target.reference,), config):
            rows.append((target.tic, toi, views, scalars))
    label = lambda t: -1 if t.label is None else int(t.label)  # noqa: E731
    return TrainingSet(
        tce_ids=np.array([r[1].toi for r in rows], dtype=str),
        kepids=np.array([r[0] for r in rows], dtype=np.int64),
        classes=np.array([r[1].disposition or "" for r in rows], dtype=str),
        labels=np.array([label(r[1]) for r in rows], dtype=np.int64),
        views={
            k: np.stack([r[2][k] for r in rows])
            if rows
            else np.empty((0, config.global_bins if k == "global" else config.local_bins), np.float32)
            for k in VIEW_NAMES
        },
        scalars={k: np.array([r[3][k] for r in rows], dtype=np.float64) for k in SCALAR_NAMES},
    )


def star_max(tics: NDArray[np.int64], scores: NDArray[np.float64]) -> dict[int, float]:
    """Highest TOI score per star."""
    best: dict[int, float] = {}
    for tic, score in zip(tics.tolist(), scores.tolist()):
        best[tic] = max(best.get(tic, -np.inf), score)
    return best


def _paired(y, a, b, seed: int, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, y.size, y.size)
        if 0 < y[idx].sum() < y.size:
            diffs.append(fast_average_precision(y[idx], a[idx]) - fast_average_precision(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass
class TransferResult:
    n_stars: int
    n_planets: int
    n_tois_scored: int
    chance: float
    average_precision: dict[str, float]
    recovered_subset: dict[str, Any]
    cnn_minus_best_tess: tuple[float, float]
    best_tess: str
    cnn_threshold: float
    planets_kept: float
    false_positives_rejected: float
    stars: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["cnn_minus_best_tess"] = list(self.cnn_minus_best_tess)
        return out


def compare(
    star_scores: dict[str, dict[int, float]],
    tess_runs: dict[str, list[dict[str, Any]]],
    *,
    cnn_threshold: float,
    n_tois_scored: int,
    seed: int = 0,
) -> TransferResult:
    """AP of every scorer on the stars all of them scored, and on the recovered subset.

    ``star_scores`` maps a DR25 model name to ``{tic: score}``; ``tess_runs``
    maps a TESS model name to the per-star rows of its ``toi_benchmark.json``.
    """
    rows = {}
    for name, stars in tess_runs.items():
        for s in stars:
            tic = int(s["target_id"].split()[-1])
            row = rows.setdefault(tic, {"label": 1 if s["disposition"] in ("CP", "KP") else 0})
            row[name] = float(s["model_score"])
            row["recovered"] = bool(s.get("period_recovered"))
    names = list(star_scores) + list(tess_runs)
    common = sorted(
        tic for tic, row in rows.items()
        if all(n in row for n in tess_runs) and all(tic in star_scores[n] for n in star_scores)
    )
    for tic in common:
        for name, scores in star_scores.items():
            rows[tic][name] = scores[tic]

    y = np.array([rows[t]["label"] for t in common])
    recovered = np.array([rows[t]["recovered"] for t in common])
    score = {n: np.array([rows[t][n] for t in common]) for n in names}
    ap = {n: fast_average_precision(y, s) for n, s in score.items()}
    sub = {n: fast_average_precision(y[recovered], s[recovered]) for n, s in score.items()}
    best_tess = max(tess_runs, key=lambda n: ap[n])
    cnn = score["kepler_cnn"]
    flagged = cnn >= cnn_threshold
    return TransferResult(
        n_stars=len(common),
        n_planets=int(y.sum()),
        n_tois_scored=n_tois_scored,
        chance=float(y.mean()),
        average_precision=ap,
        recovered_subset={
            "n_stars": int(recovered.sum()),
            "n_planets": int(y[recovered].sum()),
            "chance": float(y[recovered].mean()) if recovered.any() else float("nan"),
            "average_precision": sub,
        },
        cnn_minus_best_tess=_paired(y, cnn, score[best_tess], seed),
        best_tess=best_tess,
        cnn_threshold=cnn_threshold,
        planets_kept=float(flagged[y == 1].mean()),
        false_positives_rejected=float(1 - flagged[y == 0].mean()),
        stars=[{"tic": t, **rows[t]} for t in common],
    )


_LABELS = {
    "kepler_cnn": "Kepler DR25 CNN",
    "kepler_gbm": "Kepler DR25 boosting on views",
    "tess_synthetic": "TESS model, trained on synthetic curves",
    "tess_injection": "TESS model, trained on injections into real curves",
}


def format_transfer_report(r: TransferResult) -> str:
    sub = r.recovered_subset
    lines = [
        "Kepler DR25 models on the TESS TOI benchmark",
        "=" * 44,
        "",
        f"{r.n_stars} TOI hosts scored by every model ({r.n_planets} CP/KP hosts, "
        f"{r.n_stars - r.n_planets} FP/FA-only hosts); {r.n_tois_scored} TOIs folded.",
        "",
        f"Average precision, all stars (chance {r.chance:.3f})",
    ]
    for name, ap in sorted(r.average_precision.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {_LABELS.get(name, name):<52} {ap:.3f}")
    lo, hi = r.cnn_minus_best_tess
    lines += [
        f"  Kepler CNN minus {_LABELS[r.best_tess].lower()}: 95% paired bootstrap {lo:+.3f} to {hi:+.3f}",
        "",
        f"Stars where the TESS search found the TOI period ({sub['n_stars']}, "
        f"{sub['n_planets']} planets, chance {sub['chance']:.3f})",
    ]
    for name, ap in sorted(sub["average_precision"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {_LABELS.get(name, name):<52} {ap:.3f}")
    lines += [
        "",
        f"Kepler CNN at its Kepler threshold ({r.cnn_threshold:.3f}, 90% PC recall on Kepler):",
        f"  TESS planet hosts kept      {r.planets_kept:.1%}",
        f"  TESS FP/FA hosts rejected   {r.false_positives_rejected:.1%}",
        "",
        "The DR25 models are given each TOI's catalogue ephemeris; the TESS models",
        "find their own period with BLS.  See transitml.tess_transfer.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.tess_transfer",
        description="Score the TOI benchmark stars with the Kepler DR25 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--toi-table", type=Path, default=Path("data/toi_benchmark/exofop_toi_2026-10-06.csv"))
    parser.add_argument("--sectors", default="14-26")
    parser.add_argument("--curve-cache", type=Path, default=Path("results/tess_transfer/toi_curves.npz"))
    parser.add_argument("--kepler-set", type=Path, default=Path("results/kepler_dr25/full/training_set.npz"))
    parser.add_argument("--cnn-results", type=Path, default=Path("results/kepler_dr25/full_cnn"))
    parser.add_argument(
        "--tess-benchmark",
        nargs=2,
        action="append",
        metavar=("NAME", "JSON"),
        help="a TESS model's toi_benchmark.json to compare with",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/tess_transfer"))
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from .benchmark import load_or_fetch_curves
    from .cnn import CNNConfig, _predict, _torch, build_network, view_tensors
    from .data.toi import parse_sector_spec, read_toi_table, select_benchmark_targets

    args = parse_args(argv)
    tess = args.tess_benchmark or [
        ["tess_synthetic", "results/toi_benchmark.json"],
        ["tess_injection", "results/real_injection/toi_benchmark.json"],
    ]
    tess_runs = {name: json.loads(Path(path).read_text())["stars"] for name, path in tess}
    wanted = {s["target_id"] for stars in tess_runs.values() for s in stars}

    targets, _ = select_benchmark_targets(
        read_toi_table(args.toi_table), parse_sector_spec(args.sectors)
    )
    targets = [t for t in targets if t.target_id in wanted]
    curves = load_or_fetch_curves(targets, args.curve_cache, n_workers=args.n_workers)
    toi_set = build_toi_set(curves, targets)
    print(f"{len(curves)} curves, {len(toi_set)} TOIs folded")

    torch = _torch()
    cnn_metrics = json.loads((args.cnn_results / "metrics.json").read_text())
    states = torch.load(args.cnn_results / "cnn_models.pt")
    inputs = view_tensors(toi_set)
    nets = []
    for state in states:
        net = build_network(CNNConfig(), inputs["global"].shape[2], inputs["local"].shape[2])
        net.load_state_dict(state)
        nets.append(net)
    cnn = np.mean([_predict(n, inputs) for n in nets], axis=0)

    kepler = TrainingSet.load(args.kepler_set)
    gbm = _build_classifier(args.seed).fit(view_features(kepler), kepler.labels)
    gbm_scores = gbm.predict_proba(view_features(toi_set))[:, 1]

    result = compare(
        {"kepler_cnn": star_max(toi_set.kepids, cnn), "kepler_gbm": star_max(toi_set.kepids, gbm_scores)},
        tess_runs,
        cnn_threshold=float(cnn_metrics["cnn_threshold"]),
        n_tois_scored=len(toi_set),
        seed=args.seed,
    )
    report = format_transfer_report(result)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "report.txt").write_text(report)
    (args.results_dir / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
