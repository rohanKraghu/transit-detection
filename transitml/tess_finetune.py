"""Kepler first, then TESS: fine-tune the DR25 models on real TOI labels.

    python -m transitml.tess_finetune

Two results meet here.  The Kepler DR25 models, scored unchanged on the TOI
benchmark (:mod:`transitml.tess_transfer`), beat every model trained on
synthetic or injected TESS signals.  The boosting model trained on real TOI
dispositions (:mod:`transitml.toi_training`) closes most of that gap from the
other side.  This module asks whether the two add up: start from what Kepler
taught, then learn from the labelled TOI hosts of sectors 1 to 13, and score
the TOI hosts of sectors 14 to 26, which share no star with the training set.

Models, all fixed before the test hemisphere was scored
-------------------------------------------------------
* ``finetuned_cnn``: each of the three Kepler CNNs, trained further on the
  TESS TOIs at a tenth of the original learning rate, early-stopped on 15% of
  the training stars, then averaged.
* ``tess_cnn``: the same network from random weights on the TESS TOIs alone,
  to show what the Kepler start is worth.
* ``combined_gbm``: boosting on the DR25 views of all 34,032 Kepler TCEs plus
  the TESS TOIs, with the TESS rows weighted so they count as much in total as
  all of Kepler.
* ``tess_gbm``: the same boosting model on the TESS TOIs alone.

The unchanged Kepler models, the TESS model trained on synthetic curves and
the boosting model trained on TOI dispositions (with and without pixel
features) are scored on the same stars for comparison.  A star scores as its
highest TOI, open candidates included, exactly as in the transfer test.

Training labels are per TOI: confirmed and known planets against false
positives and false alarms.  Open candidates are not trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .evaluate import fast_average_precision
from .kepler_dr25 import TrainingSet, _build_classifier, group_split, view_features
from .tess_transfer import _paired, build_toi_set, star_max

#: Fine-tuning steps at a tenth of the from-scratch rate, so the Kepler
#: weights move rather than get overwritten.
FINETUNE_LEARNING_RATE: float = 1e-4

#: Where a model trained on TOI labels keeps a star: at least as likely a
#: planet as not, at the training set's own mix (as in :mod:`transitml.toi_training`).
OPERATING_PROBABILITY: float = 0.5

#: Pairs whose difference in average precision gets a paired bootstrap.
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("finetuned_cnn", "kepler_cnn"),
    ("finetuned_cnn", "tess_cnn"),
    ("finetuned_cnn", "toi_trained"),
    ("combined_gbm", "tess_gbm"),
    ("tess_gbm", "toi_trained"),
    ("combined_gbm", "toi_trained"),
    ("finetuned_cnn", "toi_trained_pixels"),
)


def labelled(ts: TrainingSet) -> TrainingSet:
    """The TOIs with a disposition (planet or false positive); open candidates dropped."""
    return ts.subset(np.flatnonzero(ts.labels >= 0))


def drop_stars(ts: TrainingSet, tics: NDArray[np.int64]) -> TrainingSet:
    """``ts`` without any row on a star in ``tics``."""
    return ts.subset(np.flatnonzero(~np.isin(ts.kepids, tics)))


def tess_weight(n_kepler: int, n_tess: int) -> float:
    """Per-row weight that makes the TESS rows, together, weigh as much as all of Kepler."""
    return n_kepler / max(n_tess, 1)


def fit_gbm(
    X: NDArray[np.float64], y: NDArray[np.int_], seed: int, weight: NDArray[np.float64] | None = None
):
    return _build_classifier(seed).fit(X, y, sample_weight=weight)


def train_cnns(
    ts: TrainingSet,
    config: Any,
    seed: int,
    init_states: list[dict[str, Any]] | None = None,
) -> tuple[list[Any], list[int]]:
    """One network per Kepler state (or ``config.n_models`` from scratch), early-stopped on held-back stars.

    Returns the networks and the epochs each ran.
    """
    from .cnn import train_network, view_tensors

    inputs = view_tensors(ts)
    train, valid = group_split(ts.kepids, config.validation_size, seed)
    n = len(init_states) if init_states else config.n_models
    nets, epochs = [], []
    for i in range(n):
        net, history = train_network(
            inputs,
            ts.labels,
            train,
            valid,
            config,
            seed=seed + i,
            init_state=init_states[i] if init_states else None,
        )
        nets.append(net)
        epochs.append(len(history))
    return nets, epochs


def predict_cnns(nets: list[Any], ts: TrainingSet) -> NDArray[np.float64]:
    from .cnn import _predict, view_tensors

    inputs = view_tensors(ts)
    return np.mean([_predict(n, inputs) for n in nets], axis=0)


@dataclass
class FinetuneResult:
    n_train_tois: int
    n_train_stars: int
    n_train_planets: int
    n_stars: int
    n_planets: int
    chance: float
    average_precision: dict[str, float]
    recovered_subset: dict[str, Any]
    differences: dict[str, tuple[float, float, float]]
    operating_point: dict[str, float]
    epochs: dict[str, list[int]]
    tess_weight: float
    stars: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["differences"] = {k: list(v) for k, v in self.differences.items()}
        return out


def compare_all(
    star_scores: dict[str, dict[int, float]],
    tess_runs: dict[str, list[dict[str, Any]]],
    *,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
    operating_model: str = "finetuned_cnn",
    seed: int = 0,
) -> dict[str, Any]:
    """AP of every scorer on the stars all of them scored, with paired differences.

    ``star_scores`` maps a model name to ``{tic: score}``; ``tess_runs`` maps a
    benchmark model name to the per-star rows of its ``toi_benchmark.json``.
    """
    rows: dict[int, dict[str, Any]] = {}
    for name, stars in tess_runs.items():
        for s in stars:
            tic = int(s["target_id"].split()[-1])
            row = rows.setdefault(tic, {"label": 1 if s["disposition"] in ("CP", "KP") else 0})
            row[name] = float(s["model_score"])
            row["recovered"] = bool(s.get("period_recovered"))
    names = list(star_scores) + list(tess_runs)
    common = sorted(
        tic
        for tic, row in rows.items()
        if all(n in row for n in tess_runs) and all(tic in star_scores[n] for n in star_scores)
    )
    for tic in common:
        for name, scores in star_scores.items():
            rows[tic][name] = scores[tic]

    y = np.array([rows[t]["label"] for t in common])
    recovered = np.array([rows[t]["recovered"] for t in common], dtype=bool)
    score = {n: np.array([rows[t][n] for t in common]) for n in names}
    ap = {n: fast_average_precision(y, s) for n, s in score.items()}
    sub = {n: fast_average_precision(y[recovered], s[recovered]) for n, s in score.items()}
    differences = {}
    for a, b in pairs:
        if a in score and b in score:
            lo, hi = _paired(y, score[a], score[b], seed)
            differences[f"{a} - {b}"] = (ap[a] - ap[b], lo, hi)
    operating = {}
    if operating_model in score:
        kept = score[operating_model] >= OPERATING_PROBABILITY
        operating = {
            "probability": OPERATING_PROBABILITY,
            "planets_kept": float(kept[y == 1].mean()),
            "false_positives_rejected": float(1 - kept[y == 0].mean()),
        }
    return {
        "n_stars": len(common),
        "n_planets": int(y.sum()),
        "chance": float(y.mean()),
        "average_precision": ap,
        "recovered_subset": {
            "n_stars": int(recovered.sum()),
            "n_planets": int(y[recovered].sum()),
            "chance": float(y[recovered].mean()) if recovered.any() else float("nan"),
            "average_precision": sub,
        },
        "differences": differences,
        "operating_point": operating,
        "stars": [{"tic": t, **rows[t]} for t in common],
    }


_LABELS = {
    "finetuned_cnn": "Kepler CNN fine-tuned on TESS TOIs",
    "tess_cnn": "CNN trained on TESS TOIs only",
    "combined_gbm": "boosting on Kepler TCEs + TESS TOIs",
    "tess_gbm": "boosting on views, TESS TOIs only",
    "kepler_cnn": "Kepler CNN, unchanged",
    "kepler_gbm": "Kepler boosting on views, unchanged",
    "toi_trained": "boosting on TOI labels, light-curve features",
    "toi_trained_pixels": "boosting on TOI labels, with pixel features",
    "tess_synthetic": "TESS model, trained on synthetic curves",
}


def format_finetune_report(r: FinetuneResult) -> str:
    sub = r.recovered_subset
    lines = [
        "Kepler first, then TESS: fine-tuning on real TOI labels",
        "=" * 55,
        "",
        f"Trained on {r.n_train_tois} labelled TOIs on {r.n_train_stars} stars of sectors 1-13 "
        f"({r.n_train_planets} planet TOIs); TESS row weight in the combined boosting model "
        f"{r.tess_weight:.1f}.",
        f"Scored on {r.n_stars} TOI hosts of sectors 14-26 ({r.n_planets} CP/KP hosts, "
        f"{r.n_stars - r.n_planets} FP/FA-only hosts), no star shared with training.",
        "Epochs run: "
        + "; ".join(f"{_LABELS.get(k, k)} {', '.join(map(str, v))}" for k, v in r.epochs.items()),
        "",
        f"Average precision, all stars (chance {r.chance:.3f})",
    ]
    for name, ap in sorted(r.average_precision.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {_LABELS.get(name, name):<50} {ap:.3f}")
    lines += ["", "Paired bootstrap, difference in AP (95% interval)"]
    for pair, (d, lo, hi) in r.differences.items():
        a, b = pair.split(" - ")
        lines.append(f"  {_LABELS.get(a, a)} minus {_LABELS.get(b, b).lower()}")
        lines.append(f"      {d:+.3f}  ({lo:+.3f} to {hi:+.3f})")
    lines += [
        "",
        f"Stars where the TESS search found the TOI period ({sub['n_stars']}, "
        f"{sub['n_planets']} planets, chance {sub['chance']:.3f})",
    ]
    for name, ap in sorted(sub["average_precision"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {_LABELS.get(name, name):<50} {ap:.3f}")
    if r.operating_point:
        op = r.operating_point
        lines += [
            "",
            f"Fine-tuned CNN at probability {op['probability']:.2f}:",
            f"  TESS planet hosts kept      {op['planets_kept']:.1%}",
            f"  TESS FP/FA hosts rejected   {op['false_positives_rejected']:.1%}",
        ]
    lines += [
        "",
        "The view-based models are given each TOI's catalogue ephemeris; the",
        "benchmark models find their own period with BLS.  See transitml.tess_finetune.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.tess_finetune",
        description="Fine-tune the Kepler DR25 models on TOI labels from one set of sectors, score another.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--toi-table", type=Path, default=Path("data/toi_benchmark/exofop_toi_2026-10-06.csv"))
    parser.add_argument("--train-sectors", default="1-13")
    parser.add_argument("--test-sectors", default="14-26")
    parser.add_argument("--train-cache", type=Path, default=Path("results/tess_finetune/train_curves.npz"))
    parser.add_argument("--test-cache", type=Path, default=Path("results/tess_transfer/toi_curves.npz"))
    parser.add_argument("--kepler-set", type=Path, default=Path("results/kepler_dr25/full/training_set.npz"))
    parser.add_argument("--cnn-results", type=Path, default=Path("results/kepler_dr25/full_cnn"))
    parser.add_argument(
        "--tess-benchmark",
        nargs=2,
        action="append",
        metavar=("NAME", "JSON"),
        help="a benchmark model's toi_benchmark.json (scored on the test sectors) to compare with",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/tess_finetune"))
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from .benchmark import load_or_fetch_curves
    from .cnn import CNNConfig, _torch, build_network
    from .data.toi import parse_sector_spec, read_toi_table, select_benchmark_targets

    args = parse_args(argv)
    tess = args.tess_benchmark or [
        ["tess_synthetic", "results/toi_benchmark.json"],
        ["toi_trained", "results/toi_trained/toi_benchmark.json"],
        ["toi_trained_pixels", "results/toi_trained/pixels/toi_benchmark.json"],
    ]
    tess_runs = {name: json.loads(Path(path).read_text())["stars"] for name, path in tess}
    wanted = {s["target_id"] for stars in tess_runs.values() for s in stars}
    tois = read_toi_table(args.toi_table)

    test_targets, _ = select_benchmark_targets(tois, parse_sector_spec(args.test_sectors))
    test_targets = [t for t in test_targets if t.target_id in wanted]
    test_set = build_toi_set(load_or_fetch_curves(test_targets, args.test_cache, n_workers=args.n_workers), test_targets)

    train_targets, _ = select_benchmark_targets(tois, parse_sector_spec(args.train_sectors))
    train_curves = load_or_fetch_curves(train_targets, args.train_cache, n_workers=args.n_workers)
    train_set = labelled(drop_stars(build_toi_set(train_curves, train_targets), test_set.kepids))
    print(f"train: {len(train_set)} labelled TOIs on {np.unique(train_set.kepids).size} stars; "
          f"test: {len(test_set)} TOIs folded")

    torch = _torch()
    states = torch.load(args.cnn_results / "cnn_models.pt")
    kepler_nets = []
    for state in states:
        net = build_network(CNNConfig(), test_set.views["global"].shape[1], test_set.views["local"].shape[1])
        net.load_state_dict(state)
        kepler_nets.append(net)

    fine_config = CNNConfig(learning_rate=FINETUNE_LEARNING_RATE)
    fine_nets, fine_epochs = train_cnns(train_set, fine_config, args.seed, init_states=states)
    scratch_nets, scratch_epochs = train_cnns(train_set, CNNConfig(), args.seed)

    kepler = TrainingSet.load(args.kepler_set)
    X_kep, X_tess, X_test = view_features(kepler), view_features(train_set), view_features(test_set)
    w = tess_weight(len(kepler), len(train_set))
    combined = fit_gbm(
        np.vstack([X_kep, X_tess]),
        np.concatenate([kepler.labels, train_set.labels]),
        args.seed,
        weight=np.concatenate([np.ones(len(kepler)), np.full(len(train_set), w)]),
    )
    kepler_gbm = fit_gbm(X_kep, kepler.labels, args.seed)
    tess_gbm = fit_gbm(X_tess, train_set.labels, args.seed)

    per_toi = {
        "finetuned_cnn": predict_cnns(fine_nets, test_set),
        "tess_cnn": predict_cnns(scratch_nets, test_set),
        "kepler_cnn": predict_cnns(kepler_nets, test_set),
        "combined_gbm": combined.predict_proba(X_test)[:, 1],
        "tess_gbm": tess_gbm.predict_proba(X_test)[:, 1],
        "kepler_gbm": kepler_gbm.predict_proba(X_test)[:, 1],
    }
    cmp = compare_all({k: star_max(test_set.kepids, v) for k, v in per_toi.items()}, tess_runs, seed=args.seed)
    result = FinetuneResult(
        n_train_tois=len(train_set),
        n_train_stars=int(np.unique(train_set.kepids).size),
        n_train_planets=int(train_set.labels.sum()),
        epochs={"finetuned_cnn": fine_epochs, "tess_cnn": scratch_epochs},
        tess_weight=w,
        **cmp,
    )
    report = format_finetune_report(result)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "report.txt").write_text(report)
    (args.results_dir / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n")
    torch.save([n.state_dict() for n in fine_nets], args.results_dir / "finetuned_cnn_models.pt")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
