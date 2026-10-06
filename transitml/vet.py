"""Vet one star: detrend, search, featurise, score, and explain the score.

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

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import MultiPlanetConfig
from .data.base import LightCurve
from .features import FEATURE_NAMES, extract_features, run_bls
from .model import SavedModel
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
