"""Vet one star: detrend, search, featurise, score, and explain the score.

Usage::

    python run_pipeline.py                      # trains and saves results/model.joblib
    python -m transitml.vet star.csv            # a local time,flux[,flux_err] CSV
    python -m transitml.vet cache.npz --target-id "TIC 123"
    python -m transitml.vet "TIC 307210830" --sector 14   # needs lightkurve + network
    python -m transitml.vet "TIC 307210830" --stitch      # every sector, joined

Writes ``vet_<target>.png`` and ``vet_<target>.json`` to ``--out-dir``.

This is the single-target counterpart of ``run_pipeline.py``.  It takes one
light curve and a model trained by ``run_pipeline.py`` (``results/model.joblib``)
and runs exactly the same detrending and feature extraction on it, with the
preprocess and BLS settings stored in the model file, so the score means what
it meant in training.

The classifier sees only the primary (strongest) BLS signal, as in training.
The iterative search in :mod:`transitml.search` additionally lists every
significant signal, so a second planet is reported even though it is not
scored.

The "top reasons" are an approximation and are labelled as one: for each
feature, the change in score when that feature alone is replaced by its median
over the training split.  They are not additive and do not sum to the score;
they show which measured values the model reacted to most for this star.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import MultiPlanetConfig
from .data.base import LightCurve, stitch_light_curves
from .data.files import read_light_curves
from .data.mast import MASTLightCurveSource
from .features import FEATURE_NAMES, extract_features, run_bls
from .model import SavedModel, load_model
from .preprocess import FlattenedLightCurve, flatten
from .search import CandidateSignal, iterative_search

#: Features shown in the report table, in display order.
KEY_FEATURES: tuple[str, ...] = (
    "bls_sde",
    "bls_depth_snr",
    "log_depth",
    "bls_duration",
    "log_period",
    "n_transits",
    "odd_even_sigma",
    "secondary_sigma",
    "flat_bottom_fraction",
    "log_duration_ratio",
    "red_noise_beta",
    "max_single_event_fraction",
)

#: How many of the largest score changes are reported as reasons.
N_REASONS = 5

CONTRIBUTION_METHOD = (
    "approximate: change in score when this feature alone is replaced by its "
    "training-split median; not additive"
)


@dataclass
class VetResult:
    """Everything the vetting report shows, as plain numbers."""

    target_id: str
    score: float
    threshold: float
    features: dict[str, float]
    primary: dict[str, float]
    candidates: list[CandidateSignal]
    contributions: list[dict[str, float | str]]
    n_cadences: int
    baseline_days: float
    model_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def above_threshold(self) -> bool:
        return self.score >= self.threshold

    @property
    def verdict(self) -> str:
        if self.above_threshold:
            return "planet candidate (score at or above the operating threshold)"
        return "not a candidate (score below the operating threshold)"

    @property
    def reasons(self) -> list[dict[str, float | str]]:
        return self.contributions[:N_REASONS]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "target_id": self.target_id,
                "score": self.score,
                "threshold": self.threshold,
                "above_threshold": self.above_threshold,
                "verdict": self.verdict,
                "primary_signal": self.primary,
                "candidates": [c.to_dict() for c in self.candidates],
                "features": self.features,
                "top_reasons": self.reasons,
                "contribution_method": CONTRIBUTION_METHOD,
                "n_cadences": self.n_cadences,
                "baseline_days": self.baseline_days,
                "model": self.model_provenance,
            }
        )


def _json_safe(value: Any) -> Any:
    """NaN and infinities become ``None`` so the output is strict JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def feature_contributions(model: SavedModel, x: np.ndarray) -> list[dict[str, float | str]]:
    """Score change from replacing each feature, one at a time, by its training median.

    ``delta = score(x) - score(x with feature j set to its median)``: positive
    means this star's value of feature ``j`` pushed the score up relative to a
    typical training curve.  Features whose median is undefined are skipped.
    Sorted by ``|delta|``, largest first.
    """
    x = np.asarray(x, dtype=float).reshape(1, -1)
    base = float(model.score(x)[0])
    usable = [j for j in range(x.shape[1]) if np.isfinite(model.train_medians[j])]
    if not usable:
        return []
    probes = np.repeat(x, len(usable), axis=0)
    for row, j in enumerate(usable):
        probes[row, j] = model.train_medians[j]
    replaced = model.score(probes)
    rows = [
        {
            "feature": model.feature_names[j],
            "value": float(x[0, j]),
            "training_median": float(model.train_medians[j]),
            "delta_score": base - float(replaced[row]),
        }
        for row, j in enumerate(usable)
    ]
    rows.sort(key=lambda r: abs(float(r["delta_score"])), reverse=True)
    return rows


def vet_light_curve(
    lc: LightCurve,
    model: SavedModel,
    multi: MultiPlanetConfig | None = None,
) -> tuple[VetResult, FlattenedLightCurve]:
    """Detrend, search, featurise and score one light curve."""
    lc = lc.finite()
    flat = flatten(lc, model.preprocess)
    features = extract_features(flat, model.bls)
    primary = run_bls(flat, model.bls)
    candidates = iterative_search(flat, model.bls, multi)

    x = np.array([features[name] for name in FEATURE_NAMES], dtype=float)
    score = float(model.score(x)[0])
    result = VetResult(
        target_id=lc.target_id,
        score=score,
        threshold=model.threshold,
        features=features,
        primary={
            "period": primary["period"],
            "epoch": primary["transit_time"],
            "duration": primary["duration"],
            "depth": primary["depth"],
            "depth_snr": primary["depth_snr"],
            "sde": features["bls_sde"],
        },
        candidates=candidates,
        contributions=feature_contributions(model, x),
        n_cadences=int(flat.time.size),
        baseline_days=float(flat.baseline_days),
        model_provenance={"threshold_rule": model.threshold_rule, **model.provenance},
    )
    return result, flat


def write_report(
    lc: LightCurve,
    flat: FlattenedLightCurve,
    result: VetResult,
    out_dir: str | Path,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Write ``<stem>.png`` (the one-page report) and ``<stem>.json``.  Returns both paths."""
    from .plots import plot_vetting_report  # matplotlib only when a figure is drawn

    out_dir = Path(out_dir)
    stem = stem or "vet_" + "".join(ch if ch.isalnum() else "_" for ch in result.target_id)
    png = plot_vetting_report(lc.finite(), flat, result, out_dir / f"{stem}.png")
    return png, write_json(result, out_dir / f"{stem}.json")


def write_json(result: VetResult, path: str | Path) -> Path:
    """The report's numbers as strict JSON (NaN written as ``null``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n")
    return path


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
_TIC_PATTERN = re.compile(r"^\s*(?:TIC)?\s*(\d+)\s*$", re.IGNORECASE)


def tic_id(text: str) -> str | None:
    """``"TIC 123"`` for ``"TIC 123"``, ``"tic123"`` or ``"123"``; ``None`` otherwise."""
    match = _TIC_PATTERN.match(text)
    return f"TIC {int(match.group(1))}" if match else None


def load_target_curves(target: str, args: argparse.Namespace) -> list[LightCurve]:
    """Light curves for ``target``: a local file if one exists, else a TIC from MAST."""
    path = Path(target)
    if path.exists():
        return read_light_curves(path, args.target_id)
    tic = tic_id(target)
    if tic is None:
        raise SystemExit(f"{target!r} is neither an existing file nor a TIC ID")
    source = MASTLightCurveSource(
        [(tic, None)],
        mission="TESS",
        author=args.author,
        exposure_time=args.exposure_time,
        sector=args.sector,
        stitch_sectors=args.stitch,
    )
    curves = list(source)
    if not curves:
        raise SystemExit(f"{tic}: no light curve found on MAST (or the download failed)")
    return curves


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.vet",
        description="Score one light curve with a model trained by run_pipeline.py "
        "and write a one-page vetting report (PNG) plus its numbers (JSON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target", help="A .csv or .npz light-curve file, or a TIC ID (downloaded from MAST)."
    )
    parser.add_argument(
        "--model", type=Path, default=Path("results/model.joblib"),
        help="Model file written by run_pipeline.py.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/vet"), help="Where the report lands."
    )
    parser.add_argument(
        "--target-id", default=None, help="Which star to take from a multi-star npz cache."
    )
    parser.add_argument("--sector", type=int, default=None, help="TESS sector (TIC input).")
    parser.add_argument("--author", default="TESS-SPOC", help="Light-curve pipeline (TIC input).")
    parser.add_argument(
        "--exposure-time", type=int, default=1800, help="Cadence in seconds (TIC input)."
    )
    parser.add_argument(
        "--stitch", action="store_true",
        help="Join every sector of the star into one curve before vetting.",
    )
    parser.add_argument(
        "--max-signals", type=int, default=MultiPlanetConfig.max_signals,
        help="Most signals the iterative search reports.",
    )
    parser.add_argument(
        "--min-sde", type=float, default=MultiPlanetConfig.min_sde,
        help="Significance a signal needs to be listed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    curves = load_target_curves(args.target, args)
    if args.stitch and len(curves) > 1:
        try:
            curves = [stitch_light_curves(curves)]
        except ValueError as exc:
            raise SystemExit(f"cannot stitch: {exc}; pick one star with --target-id") from exc
    if len(curves) != 1:
        found = ", ".join(
            f"{lc.target_id} (sector {lc.meta.get('sector', '?')})" for lc in curves
        )
        raise SystemExit(
            f"{len(curves)} light curves found ({found}); pick one with --target-id "
            "or --sector, or join one star's sectors with --stitch"
        )
    lc = curves[0]
    model = load_model(args.model)
    multi = MultiPlanetConfig(max_signals=args.max_signals, min_sde=args.min_sde)
    result, flat = vet_light_curve(lc, model, multi)
    png, js = write_report(lc, flat, result, args.out_dir)

    print(f"{result.target_id}: score {result.score:.3f}, threshold {result.threshold:.3f}")
    print(f"  verdict: {result.verdict}")
    for cand in result.candidates:
        print(
            f"  signal {cand.rank}: P = {cand.period:.4f} d, depth {cand.depth * 1e6:.0f} ppm, "
            f"duration {cand.duration * 24:.2f} h, SDE {cand.sde:.1f}, SNR {cand.depth_snr:.1f}"
        )
    if not result.candidates:
        print(f"  no signal reached SDE {multi.min_sde:g}")
    print(f"  wrote {png} and {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
