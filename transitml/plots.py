"""Figures written to disk with the Agg backend (no display required).

Four figures, each answering one question:

``light_curves``     What does the detrending actually do, and to what?
``pr_curve``         Does the model beat the baselines where it matters?
``diagnostics``      Where does it fail, and what is it adding over BLS SNR?
``feature_importance``  Which vetting statistics are carrying the decision?
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # must precede pyplot; there is no display in CI or in a container

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from .config import Config  # noqa: E402
from .data.base import LightCurve  # noqa: E402
from .data.loader import Dataset  # noqa: E402
from .evaluate import EvaluationResult  # noqa: E402
from .model import Split, TrainedModel  # noqa: E402
from .preprocess import flatten  # noqa: E402

# --- Design tokens ---------------------------------------------------------
# Categorical slots 1-3 of a CVD-validated palette (worst all-pairs deuteranope
# Delta E 9.2, normal-vision 24.0).  Assigned by entity, in fixed order, never
# cycled; every series is also legended or directly labelled so identity is
# never carried by colour alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
NEUTRAL = "#8d8b84"

_CLASS_COLOR = {"planet": SERIES[0], "eclipsing_binary": SERIES[1], "noise": SERIES[2]}
_CLASS_LABEL = {
    "planet": "planet (label 1)",
    "eclipsing_binary": "eclipsing binary (label 0)",
    "noise": "variable star, no companion (label 0)",
}


def _style() -> None:
    """Recessive chrome: hairline solid axes and grid, generous padding."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9.5,
            "figure.dpi": 130,
            "axes.axisbelow": True,
        }
    )
    for spine in ("top", "right"):
        plt.rcParams[f"axes.spines.{spine}"] = False


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def plot_light_curves(
    curves: Sequence[LightCurve], config: Config, path: Path
) -> Path:
    """Raw photometry with the fitted trend, beside the phase-folded result.

    One row per population.  The right-hand panels are folded on the period the
    **pipeline recovered**, not on the injected truth, so the figure shows what
    the classifier actually sees.  Binned means are drawn on top of the
    individual cadences: a 1000 ppm transit is invisible in 400 ppm per-cadence
    noise until ~20 cadences are folded together, and that folding gain is the
    entire reason a periodic search works.
    """
    from .features import run_bls  # local import: keeps module import cheap

    _style()
    fig, axes = plt.subplots(len(curves), 2, figsize=(13.0, 2.7 * len(curves)))
    axes = np.atleast_2d(axes)

    for row, lc in enumerate(curves):
        flat = flatten(lc, config.preprocess)
        bls = run_bls(flat, config.bls)
        kind = str(lc.meta.get("kind", "noise"))
        colour = _CLASS_COLOR.get(kind, NEUTRAL)
        ax_raw, ax_fold = axes[row, 0], axes[row, 1]

        # -- left: raw flux and the trend that was removed ------------------
        ax_raw.plot(
            lc.time, (lc.flux - 1) * 1e3, ".", ms=1.8, color=NEUTRAL, alpha=0.7,
            label="raw flux",
        )
        ax_raw.plot(
            flat.time, (flat.trend - 1) * 1e3, "-", lw=1.8, color=SERIES[0],
            label="fitted trend (robust spline + rotation harmonics)",
        )
        ax_raw.set_ylim(*_robust_limits((lc.flux - 1) * 1e3))
        ax_raw.set_ylabel("flux - 1 (ppt)")
        ax_raw.set_title(_CLASS_LABEL.get(kind, kind), loc="left", fontsize=9.5,
                         color=colour, fontweight="bold")
        ax_raw.set_title(_curve_caption(lc), loc="right", fontsize=8.5, color=INK_SOFT)
        if row == 0:
            ax_raw.legend(loc="upper right", ncols=1)

        # -- right: folded on the recovered period --------------------------
        period, duration = bls["period"], bls["duration"]
        phase_days = (flat.time - bls["transit_time"] + 0.5 * period) % period - 0.5 * period
        hours = phase_days * 24.0
        depth_ppt = (flat.flux - 1) * 1e3
        # Never fold past half a period, or the window wraps onto itself.
        half_window = min(4.0 * duration, 0.5 * period) * 24.0

        window = np.abs(hours) <= half_window
        ax_fold.plot(hours[window], depth_ppt[window], ".", ms=2.4, color=NEUTRAL,
                     alpha=0.45, label="folded cadences")
        centres, means = _bin_means(
            hours[window], depth_ppt[window], bin_width=max(duration * 24.0 / 4.0, 1e-3)
        )
        ax_fold.plot(centres, means, "o", ms=4.2, color=colour, mec=SURFACE, mew=0.5,
                     label=f"binned to {duration * 360:.0f} min")
        ax_fold.axhline(0.0, lw=0.9, color=NEUTRAL)
        ax_fold.axvspan(-0.5 * duration * 24.0, 0.5 * duration * 24.0,
                        color=colour, alpha=0.14, lw=0)
        ax_fold.set_xlim(-half_window, half_window)
        ax_fold.set_ylim(*_robust_limits(means, pad=0.45))
        ax_fold.set_ylabel("flux - 1 (ppt)")
        truth = lc.meta.get("period")
        ax_fold.set_title(
            f"recovered P = {period:.3f} d", loc="left", fontsize=9, color=INK_SOFT
        )
        ax_fold.set_title(
            f"injected P = {truth:.3f} d" if truth else "nothing injected",
            loc="right", fontsize=8.5, color=INK_SOFT,
        )
        if row == 0:
            ax_fold.legend(loc="lower right", ncols=2, markerscale=1.5)

    axes[-1, 0].set_xlabel("time (days)")
    axes[-1, 1].set_xlabel("hours from mid-transit (folded on the recovered period)")
    fig.suptitle(
        "Detrending removes the stellar variability and leaves the transit standing",
        x=0.008, ha="left", fontsize=12.5, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return _save(fig, path)


def _robust_limits(
    values: NDArray[np.float64], pad: float = 0.25
) -> tuple[float, float]:
    """Percentile-based y-limits so one flare does not set the scale."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [0.5, 99.5])
    span = max(hi - lo, 1e-6)
    return float(lo - pad * span), float(hi + pad * span)


def _bin_means(
    x: NDArray[np.float64], y: NDArray[np.float64], bin_width: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean of ``y`` in equal-width bins of ``x``; empty bins become NaN."""
    if x.size == 0:
        return np.array([]), np.array([])
    n_bins = max(int(np.ceil((float(np.max(x)) - float(np.min(x))) / bin_width)), 1)
    edges = np.linspace(float(np.min(x)), float(np.max(x)) + 1e-9, n_bins + 1)
    which = np.digitize(x, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = which == b
        if sel.any():
            means[b] = float(np.mean(y[sel]))
    return centres, means


def _curve_caption(lc: LightCurve) -> str:
    kind = lc.meta.get("kind")
    if kind == "planet":
        return (
            f"depth {lc.meta['depth'] * 1e6:.0f} ppm, T14 = {lc.meta['duration_t14'] * 24:.1f} h, "
            f"b = {lc.meta['impact_parameter']:.2f}, injected SNR {lc.meta['true_snr']:.0f}"
        )
    if kind == "eclipsing_binary":
        shape = "grazing / V-shaped" if lc.meta.get("grazing") else "flat-bottomed"
        secondary = lc.meta.get("secondary_depth", 0.0) or 0.0
        return (
            f"depth {lc.meta['depth'] * 1e6:.0f} ppm, {shape}, "
            f"secondary {secondary * 1e6:.0f} ppm"
        )
    return (
        f"rotational modulation {lc.meta['variability_amplitude'] * 1e6:.0f} ppm at "
        f"P = {lc.meta['variability_period']:.2f} d"
    )


# --------------------------------------------------------------------------
def plot_pr_curve(result: EvaluationResult, path: Path) -> Path:
    """Precision-recall curves for the model and both baselines.

    Precision-recall, not ROC: with ~2300 negatives in the test set the ROC
    false-positive-rate axis compresses the entire interesting range into its
    first two percent.
    """
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    entries = [(result.model, SERIES[0], "gradient boosting")]
    for curve, colour in zip(result.baselines, SERIES[1:]):
        entries.append((curve, colour, f"baseline: {curve.name}"))

    for curve, colour, label in entries:
        ax.plot(
            curve.recall, curve.precision, lw=2.0, color=colour,
            label=f"{label}  (AP = {curve.average_precision:.3f})",
        )

    chance = result.chance_average_precision
    ax.axhline(chance, lw=1.4, ls="--", color=NEUTRAL)
    ax.text(
        0.015, chance + 0.02,
        f"random ranking (AP = {chance:.3f})",
        ha="left", va="bottom", color=INK_SOFT, fontsize=8.5,
    )

    # Direct-label the operating point; the aqua slot sits below 3:1 contrast on
    # this surface, so every series also gets a visible legend entry.
    ax.plot(
        [result.test_recall], [result.test_precision], "o", ms=9,
        color=SERIES[0], mec=SURFACE, mew=2.0, zorder=5,
    )
    ax.annotate(
        f"operating point\nprecision {result.test_precision:.2f}, recall {result.test_recall:.2f}",
        xy=(result.test_recall, result.test_precision),
        xytext=(12, 16), textcoords="offset points",
        color=INK, fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": NEUTRAL, "lw": 0.9},
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall (fraction of real planets recovered)")
    ax.set_ylabel("precision (fraction of flagged candidates that are real)")
    ax.set_title(
        f"Precision-recall on {result.n_test} held-out light curves "
        f"({result.n_test_positive} planets, {result.positive_rate:.1%} positive)",
        loc="left",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
def plot_diagnostics(
    dataset: Dataset,
    split: Split,
    trained: TrainedModel,
    result: EvaluationResult,
    path: Path,
) -> Path:
    """Two panels: completeness versus injected SNR, and score versus BLS SNR.

    The right-hand panel is the whole argument for the model in one picture:
    points far to the right are strong BLS detections, and the vertical spread
    at fixed BLS SNR is exactly the information the SNR baseline cannot see.
    """
    _style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # -- completeness vs injected SNR --------------------------------------
    rows = [r for r in result.recall_by_snr if r["n_planets"] > 0]
    labels, values, counts = [], [], []
    for r in rows:
        hi = "+" if r["snr_high"] > 1e8 else f"-{r['snr_high']:.0f}"
        labels.append(f"{r['snr_low']:.0f}{hi}")
        values.append(r["recall"])
        counts.append(r["n_planets"])
    positions = np.arange(len(labels))
    ax_left.bar(positions, values, width=0.62, color=SERIES[0])
    for x, v, n in zip(positions, values, counts):
        ax_left.text(
            x, v + 0.03, f"{v:.0%}\nn={n}", ha="center", va="bottom",
            color=INK_SOFT, fontsize=8.5,
        )
    ax_left.set_xticks(positions, labels)
    ax_left.set_ylim(0, 1.18)
    ax_left.set_yticks(np.linspace(0, 1, 6))
    ax_left.set_xlabel("injected transit signal-to-noise")
    ax_left.set_ylabel("recall at the operating threshold")
    ax_left.set_title("Completeness is set by injected SNR", loc="left")
    ax_left.grid(axis="x", visible=False)

    # -- model score vs BLS depth SNR --------------------------------------
    from .features import FEATURE_NAMES  # local import keeps the module import light

    snr_column = FEATURE_NAMES.index("bls_depth_snr")
    scores = trained.score(split.X_test)
    bls_snr = split.X_test[:, snr_column]
    kinds = dataset.meta.iloc[split.test_index]["kind"].astype(str).to_numpy()

    for kind in ("noise", "eclipsing_binary", "planet"):
        sel = kinds == kind
        if not sel.any():
            continue
        ax_right.scatter(
            bls_snr[sel], scores[sel], s=26 if kind == "planet" else 14,
            color=_CLASS_COLOR[kind], alpha=0.85 if kind == "planet" else 0.5,
            linewidths=0.6 if kind == "planet" else 0.0,
            edgecolors=SURFACE, label=_CLASS_LABEL[kind], zorder=3 if kind == "planet" else 2,
        )
    ax_right.set_ylim(-0.04, 1.2)  # headroom so the legend clears the points
    ax_right.axhline(trained.threshold, lw=1.4, ls="--", color=NEUTRAL)
    ax_right.text(
        0.99, trained.threshold + 0.02, f"operating threshold = {trained.threshold:.3f}",
        transform=ax_right.get_yaxis_transform(), ha="right", va="bottom",
        color=INK_SOFT, fontsize=8.5,
    )
    positive = bls_snr[np.isfinite(bls_snr) & (bls_snr > 0)]
    ax_right.set_xscale("log")
    if positive.size:
        ax_right.set_xlim(float(positive.min()) * 0.85, float(positive.max()) * 1.3)
    ax_right.set_xlabel("BLS depth signal-to-noise (the baseline's only input)")
    ax_right.set_ylabel("model score")
    ax_right.set_title("What the model adds over ranking by BLS SNR", loc="left")
    ax_right.legend(loc="upper left", markerscale=1.6)

    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
def plot_feature_importance(
    result: EvaluationResult, path: Path, top_n: int = 14
) -> Path:
    """Permutation importance, measured as loss of average precision."""
    _style()
    rows = result.feature_importance[:top_n][::-1]
    names = [r["feature"] for r in rows]
    values = np.array([r["importance"] for r in rows])
    errors = np.array([r["std"] for r in rows])

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.8))
    positions = np.arange(len(rows))
    ax.barh(positions, values, height=0.62, color=SERIES[0], xerr=errors,
            error_kw={"ecolor": NEUTRAL, "elinewidth": 1.0, "capsize": 2.5})
    for y, v, e in zip(positions, values, errors):
        ax.text(
            max(v, 0) + e + 0.004, y, f"{v:+.3f}", va="center", ha="left",
            color=INK_SOFT, fontsize=8.5,
        )
    ax.set_yticks(positions, names)
    ax.set_xlabel("drop in average precision when the feature is shuffled")
    ax.set_title(
        "Which vetting statistics carry the decision (held-out set)", loc="left"
    )
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=min(0.0, float((values - errors).min()) * 1.15))
    ax.margins(x=0.18)
    fig.tight_layout()
    return _save(fig, path)


def plot_all(
    dataset: Dataset,
    split: Split,
    trained: TrainedModel,
    result: EvaluationResult,
    curves: Sequence[LightCurve],
    config: Config,
    figure_dir: Path,
) -> list[Path]:
    """Write every figure and return the paths, in README order."""
    figure_dir = Path(figure_dir)
    return [
        plot_light_curves(curves, config, figure_dir / "01_light_curves.png"),
        plot_pr_curve(result, figure_dir / "02_precision_recall.png"),
        plot_diagnostics(dataset, split, trained, result, figure_dir / "03_diagnostics.png"),
        plot_feature_importance(result, figure_dir / "04_feature_importance.png"),
    ]
