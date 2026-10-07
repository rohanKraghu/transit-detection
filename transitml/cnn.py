"""A 1D convolutional network on the Kepler DR25 views, against gradient boosting.

    pip install torch            # optional; nothing else in the project needs it
    python -m transitml.kepler_dr25 --per-class 3025 --no-train \\
        --results-dir results/kepler_dr25/large --cache-dir results/kepler_dr25/curves
    python -m transitml.cnn --training-set results/kepler_dr25/large/training_set.npz

The architecture follows AstroNet (Shallue & Vanderburg 2018), slimmed for a
CPU and for a few thousand training examples:

* a **global column** of four convolution blocks over the 2001-bin global
  view (8 to 64 filters, kernel 5, max-pool 5 stride 2), which sees
  secondary eclipses and phase curves;
* a **local column** of two blocks over the four local views stacked as
  channels (transit, odd, even, secondary; 16 and 32 filters, kernel 5,
  max-pool 7 stride 2), which sees the transit shape and the two binary
  tests side by side;
* the two flattened, plus log period and log duration, into two dense layers
  of 256 with dropout, and one logit.

Training is Adam on binary cross-entropy, with AstroNet's one augmentation
(each view reversed in time with probability one half; a transit is
symmetric, so a reversed planet is still a planet) and early stopping on a
validation slice of the **training** stars.  Several networks with different
seeds are averaged, as AstroNet averaged ten.

The comparison is held to one split: the held-out stars are exactly those
:func:`transitml.kepler_dr25.evaluate_training_set` scores the gradient
boosting on, so the two are compared on the same TCEs, with a paired
bootstrap of the difference in average precision.  The CNN's operating
threshold is frozen on the validation stars (it never sees the test split),
the boosting's on out-of-fold training scores.
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

from .data.kepler import CLASSES
from .evaluate import fast_average_precision
from .kepler_dr25 import (
    DEFAULT_FIGURES,
    TARGET_RECALL,
    TrainingSet,
    _pass_rates,
    catalogue_precision,
    evaluate_training_set,
    group_split,
    threshold_for_recall,
)

#: DR25 class counts, for the catalogue-mix precision.
CATALOGUE_COUNTS = {"PC": 4034, "AFP": 3025, "NTP": 26973}

#: Views are divided by the transit depth, so a TCE with almost no dip has
#: huge values everywhere; clipping keeps one such TCE from dominating a batch.
VIEW_CLIP = 10.0

_TORCH_HINT = (
    "transitml.cnn needs PyTorch, which is deliberately not in requirements.txt. "
    "Install the CPU build with: pip install torch"
)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_TORCH_HINT) from exc
    return torch


@dataclass(frozen=True)
class CNNConfig:
    global_filters: tuple[int, ...] = (8, 16, 32, 64)
    local_filters: tuple[int, ...] = (16, 32)
    kernel: int = 5
    dense: int = 256
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 40
    #: Stop when validation average precision has not improved for this many epochs.
    patience: int = 6
    #: Fraction of training stars held back for early stopping and the threshold.
    validation_size: float = 0.15
    n_models: int = 3


def view_tensors(ts: TrainingSet) -> dict[str, NDArray[np.float32]]:
    """Network inputs: global ``(N, 1, 2001)``, local ``(N, 4, 201)``, scalars ``(N, 2)``."""
    clip = lambda a: np.clip(np.nan_to_num(a, nan=0.0), -VIEW_CLIP, VIEW_CLIP)  # noqa: E731
    local = np.stack([clip(ts.views[k]) for k in ("local", "odd", "even", "secondary")], axis=1)
    scalars = np.stack(
        [np.log10(ts.scalars["period_days"]), np.log10(ts.scalars["duration_hours"])], axis=1
    )
    return {
        "global": clip(ts.views["global"])[:, None, :].astype(np.float32),
        "local": local.astype(np.float32),
        "scalars": np.nan_to_num(scalars).astype(np.float32),
    }


def build_network(config: CNNConfig, global_bins: int, local_bins: int):
    """The two-column network.  Returns a ``torch.nn.Module`` producing one logit."""
    torch = _torch()
    nn = torch.nn

    def column(channels_in: int, filters: tuple[int, ...], pool: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        c = channels_in
        for f in filters:
            layers += [
                nn.Conv1d(c, f, config.kernel, padding=config.kernel // 2),
                nn.ReLU(),
                nn.Conv1d(f, f, config.kernel, padding=config.kernel // 2),
                nn.ReLU(),
                nn.MaxPool1d(pool, stride=2),
            ]
            c = f
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    class TwoColumn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.global_column = column(1, config.global_filters, 5)
            self.local_column = column(4, config.local_filters, 7)
            with torch.no_grad():
                n_global = self.global_column(torch.zeros(1, 1, global_bins)).shape[1]
                n_local = self.local_column(torch.zeros(1, 4, local_bins)).shape[1]
            self.head = nn.Sequential(
                nn.Linear(n_global + n_local + 2, config.dense),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dense, config.dense),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dense, 1),
            )

        def forward(self, global_view, local_views, scalars):
            features = torch.cat(
                [self.global_column(global_view), self.local_column(local_views), scalars], dim=1
            )
            return self.head(features).squeeze(1)

    return TwoColumn()


def _predict(model, inputs: dict[str, NDArray[np.float32]], batch: int = 512) -> NDArray[np.float64]:
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, inputs["global"].shape[0], batch):
            sl = slice(start, start + batch)
            logits = model(
                torch.from_numpy(inputs["global"][sl]),
                torch.from_numpy(inputs["local"][sl]),
                torch.from_numpy(inputs["scalars"][sl]),
            )
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out).astype(np.float64) if out else np.empty(0)


def _subset(inputs: dict[str, NDArray[np.float32]], index: NDArray[np.int_]):
    return {k: v[index] for k, v in inputs.items()}


def train_network(
    inputs: dict[str, NDArray[np.float32]],
    labels: NDArray[np.int_],
    train: NDArray[np.int_],
    valid: NDArray[np.int_],
    config: CNNConfig,
    seed: int,
) -> tuple[Any, list[float]]:
    """Fit one network; returns it at its best validation epoch, and the AP history."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_network(config, inputs["global"].shape[2], inputs["local"].shape[2])
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    valid_inputs = _subset(inputs, valid)

    best_ap, best_state, history, stale = -1.0, None, [], 0
    for _ in range(config.max_epochs):
        model.train()
        order = rng.permutation(train)
        for start in range(0, order.size, config.batch_size):
            idx = order[start : start + config.batch_size]
            g, loc, s = inputs["global"][idx], inputs["local"][idx], inputs["scalars"][idx]
            flip = rng.random(idx.size) < 0.5
            g, loc = g.copy(), loc.copy()
            g[flip] = g[flip][..., ::-1]
            loc[flip] = loc[flip][..., ::-1]
            optimiser.zero_grad()
            logits = model(
                torch.from_numpy(np.ascontiguousarray(g)),
                torch.from_numpy(np.ascontiguousarray(loc)),
                torch.from_numpy(s),
            )
            loss = loss_fn(logits, torch.from_numpy(labels[idx].astype(np.float32)))
            loss.backward()
            optimiser.step()

        ap = fast_average_precision(labels[valid], _predict(model, valid_inputs))
        history.append(float(ap))
        if ap > best_ap + 1e-4:
            best_ap, stale = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _paired_bootstrap(y, a, b, seed: int, n: int = 2000) -> tuple[float, float, float]:
    """95% interval of AP(a) - AP(b) on shared resamples, and the share above zero."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, y.size, y.size)
        if y[idx].any():
            diffs.append(fast_average_precision(y[idx], a[idx]) - fast_average_precision(y[idx], b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float((diffs > 0).mean())


@dataclass
class CNNResult:
    n_train: int
    n_valid: int
    n_test: int
    test_class_counts: dict[str, int]
    cnn_average_precision: float
    gbm_average_precision: float
    mes_average_precision: float
    chance_average_precision: float
    ensemble_average_precision: float
    difference_ci: tuple[float, float]
    difference_share_positive: float
    cnn_threshold: float
    cnn_pass_rates: dict[str, float]
    gbm_pass_rates: dict[str, float]
    cnn_catalogue_precision: float
    gbm_catalogue_precision: float
    epochs: list[int]
    target_recall: float = TARGET_RECALL
    curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "curves"}
        out["difference_ci"] = list(self.difference_ci)
        return out


def evaluate_cnn(
    ts: TrainingSet,
    *,
    config: CNNConfig | None = None,
    seed: int = 0,
    test_size: float = 0.3,
    target_recall: float = TARGET_RECALL,
    catalogue_counts: dict[str, int] | None = None,
) -> tuple[CNNResult, list[Any]]:
    """Train the ensemble and score it beside gradient boosting on the same test stars."""
    from sklearn.metrics import precision_recall_curve

    config = config or CNNConfig()
    counts = catalogue_counts or CATALOGUE_COUNTS
    torch = _torch()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train_all, test = group_split(ts.kepids, test_size, seed)
    # Validation stars come out of the training stars only.
    inner_train, inner_valid = group_split(ts.kepids[train_all], config.validation_size, seed + 1)
    train, valid = train_all[inner_train], train_all[inner_valid]

    inputs = view_tensors(ts)
    y = ts.labels
    models, epochs = [], []
    for k in range(config.n_models):
        model, history = train_network(inputs, y, train, valid, config, seed=seed + 100 + k)
        models.append(model)
        epochs.append(len(history))

    test_inputs, valid_inputs = _subset(inputs, test), _subset(inputs, valid)
    cnn = np.mean([_predict(m, test_inputs) for m in models], axis=0)
    cnn_valid = np.mean([_predict(m, valid_inputs) for m in models], axis=0)
    threshold = threshold_for_recall(y[valid], cnn_valid, target_recall)

    gbm_result = evaluate_training_set(ts, counts, seed=seed, test_size=test_size, target_recall=target_recall)
    gbm = np.asarray(gbm_result.test_scores)
    mes = np.nan_to_num(ts.scalars["mes"], nan=0.0)[test]

    yt, ct = y[test], ts.classes[test]
    cnn_rates = _pass_rates(ct, cnn >= threshold)
    # Rank-average, so neither model's calibration dominates.
    ensemble = (np.argsort(np.argsort(cnn)) + np.argsort(np.argsort(gbm))) / (2.0 * max(cnn.size, 1))
    lo, hi, share = _paired_bootstrap(yt, cnn, gbm, seed)

    curves = {}
    for name, s in (("cnn", cnn), ("gbm", gbm), ("mes", mes)):
        p, r, _ = precision_recall_curve(yt, s)
        curves[name] = {"precision": p.tolist(), "recall": r.tolist()}

    result = CNNResult(
        n_train=int(train.size),
        n_valid=int(valid.size),
        n_test=int(test.size),
        test_class_counts={c: int((ct == c).sum()) for c in CLASSES},
        cnn_average_precision=fast_average_precision(yt, cnn),
        gbm_average_precision=gbm_result.average_precision,
        mes_average_precision=fast_average_precision(yt, mes),
        chance_average_precision=float(yt.mean()),
        ensemble_average_precision=fast_average_precision(yt, ensemble),
        difference_ci=(lo, hi),
        difference_share_positive=share,
        cnn_threshold=threshold,
        cnn_pass_rates=cnn_rates,
        gbm_pass_rates=gbm_result.test_pass_rates,
        cnn_catalogue_precision=catalogue_precision(cnn_rates, counts),
        gbm_catalogue_precision=gbm_result.catalogue_precision,
        epochs=epochs,
        target_recall=target_recall,
        curves=curves,
    )
    return result, models


def format_cnn_report(result: CNNResult) -> str:
    def pct(x: float) -> str:
        return f"{x:.1%}" if np.isfinite(x) else "n/a"

    t = result.test_class_counts
    c, g = result.cnn_pass_rates, result.gbm_pass_rates
    lines = [
        "Kepler DR25: CNN on the views against gradient boosting",
        "=" * 56,
        "",
        f"Split by star: {result.n_train:,} train, {result.n_valid:,} validation, "
        f"{result.n_test:,} test ({t['PC']} PC, {t['AFP']} AFP, {t['NTP']} NTP).",
        f"Networks averaged: {len(result.epochs)} (epochs run: {', '.join(map(str, result.epochs))}).",
        "",
        "Average precision on the test stars",
        f"  CNN                         {result.cnn_average_precision:.3f}",
        f"  gradient boosting on views  {result.gbm_average_precision:.3f}",
        f"  rank average of the two     {result.ensemble_average_precision:.3f}",
        f"  Kepler MES alone            {result.mes_average_precision:.3f}",
        f"  chance (PC fraction)        {result.chance_average_precision:.3f}",
        f"  CNN minus boosting: 95% paired bootstrap {result.difference_ci[0]:+.3f} to "
        f"{result.difference_ci[1]:+.3f}; CNN ahead in {result.difference_share_positive:.0%} of resamples",
        "",
        f"At thresholds frozen for {result.target_recall:.0%} PC recall (CNN on validation stars,",
        "boosting on out-of-fold training scores)",
        "                       CNN   boosting",
        f"  PC kept         {pct(c['PC']):>9}  {pct(g['PC']):>9}",
        f"  AFP rejected    {pct(1 - c['AFP']):>9}  {pct(1 - g['AFP']):>9}",
        f"  NTP rejected    {pct(1 - c['NTP']):>9}  {pct(1 - g['NTP']):>9}",
        f"  precision at the catalogue's class mix: CNN {pct(result.cnn_catalogue_precision)}, "
        f"boosting {pct(result.gbm_catalogue_precision)}",
        "",
        "Labels are the DR25 Robovetter's calls; see transitml.data.kepler.",
    ]
    return "\n".join(lines) + "\n"


def plot_cnn_pr(result: CNNResult, path: Path) -> Path:
    from .plots import NEUTRAL, SERIES, _save, _style, plt

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for key, colour, label, ap in (
        ("cnn", SERIES[0], "CNN on views", result.cnn_average_precision),
        ("gbm", SERIES[2], "gradient boosting on views", result.gbm_average_precision),
        ("mes", SERIES[1], "baseline: Kepler MES", result.mes_average_precision),
    ):
        curve = result.curves[key]
        ax.plot(curve["recall"][:-1], curve["precision"][:-1], lw=2.0, color=colour, label=f"{label}  (AP = {ap:.3f})")
    ax.axhline(result.chance_average_precision, lw=1.4, ls="--", color=NEUTRAL)
    ax.text(0.015, result.chance_average_precision + 0.02, f"random ranking (AP = {result.chance_average_precision:.3f})", fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall on planet candidates")
    ax.set_ylabel("precision")
    ax.set_title(f"Held-out DR25 stars: {result.n_test:,} TCEs, class-balanced sample", loc="left")
    ax.legend(loc="center left")
    fig.tight_layout()
    return _save(fig, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.cnn",
        description="Train a CNN on a DR25 training set and compare it with gradient boosting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-set", type=Path, default=Path("results/kepler_dr25/training_set.npz"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/kepler_dr25/cnn"))
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-models", type=int, default=CNNConfig.n_models)
    parser.add_argument("--max-epochs", type=int, default=CNNConfig.max_epochs)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = _torch()
    ts = TrainingSet.load(args.training_set)
    config = CNNConfig(n_models=args.n_models, max_epochs=args.max_epochs)
    result, models = evaluate_cnn(ts, config=config, seed=args.seed)
    report = format_cnn_report(result)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "report.txt").write_text(report)
    (args.results_dir / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    torch.save([m.state_dict() for m in models], args.results_dir / "cnn_models.pt")
    print(report)
    if not args.no_figures:
        plot_cnn_pr(result, args.figures_dir / "03_cnn_precision_recall.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
