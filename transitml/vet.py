"""Vet one star: detrend, search, featurise, score, and explain the score.

Usage::

    python run_pipeline.py                      # trains and saves results/model.joblib
    python -m transitml.vet star.csv            # a local time,flux[,flux_err] CSV
    python -m transitml.vet cache.npz --target-id "TIC 123"
    python -m transitml.vet "TIC 307210830" --sector 14   # needs lightkurve + network
    python -m transitml.vet "TIC 307210830" --stitch      # every sector, joined
    python -m transitml.vet star.csv --tpf star_tpf.npz   # plus a centroid test
    python -m transitml.vet "TIC 307210830" --sector 14 --centroids
    python -m transitml.vet star.csv --fit --stellar-density 1.4 --stellar-radius 1.0

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

Beside the score, the report gives a calibrated probability that the star
hosts a planet (:mod:`transitml.calibration`), at the planet rate of the
training population unless ``--planet-rate`` names another, and the "top
reasons": SHAP values (:mod:`transitml.treeshap`) in calibrated log-odds.
They are exact and additive: the base log-odds plus every feature's value
is the star's log-odds, so a reason of +1.0 multiplies the odds by e.

With ``--tpf`` (a saved :class:`~transitml.data.tpf.TargetPixelData`) or, for
a TIC, ``--centroids`` (download the target pixel files), the primary signal's
ephemeris is also run through the centroid tests in :mod:`transitml.centroid`.
That result is a separate vetting test reported beside the score: the model
was not trained on centroid features and its score does not change.

With ``--fit`` the primary signal is also fitted with a limb-darkened
transit model and sampled with MCMC (:mod:`transitml.fit`): radius ratio,
impact parameter, duration, depth and the stellar density the transit shape
implies, each with an interval, in ``vet_<target>_fit.png`` and a ``fit``
section of the JSON.  Given the star's density, the fitted one is checked
against it.  Like the centroid test, the fit does not change the score.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .centroid import CentroidConfig, CentroidResult, centroid_test
from .config import MultiPlanetConfig
from .data.base import LightCurve, stitch_light_curves
from .data.files import read_light_curves
from .data.mast import MASTLightCurveSource
from .data.tpf import TargetPixelData, download_tpfs, load_tpf
from .features import FEATURE_NAMES, detrend_and_search, extract_features
from .fit import (
    FitConfig,
    FitError,
    FitResult,
    default_exposure_minutes,
    fit_transit,
    stellar_priors_from_meta,
)
from .model import SavedModel, load_model
from .preprocess import FlattenedLightCurve
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

CENTROID_NOTE = (
    "separate vetting test reported beside the score; the classifier was not "
    "trained on centroid features and its score does not use them"
)

CONTRIBUTION_METHOD = (
    "SHAP values (exact path-dependent TreeSHAP) in calibrated log-odds; "
    "base_log_odds plus the sum over every feature equals log_odds"
)

PROBABILITY_NOTE = (
    "calibrated on out-of-fold training scores; the chance this star hosts a planet "
    "if a fraction planet_rate of stars like it do"
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
    probability: float = float("nan")
    planet_rate: float = float("nan")
    log_odds: float = float("nan")
    base_log_odds: float = float("nan")
    model_provenance: dict[str, Any] = field(default_factory=dict)
    #: ``None`` when no centroid test was asked for; else one per target pixel file.
    centroids: list[CentroidResult] | None = None
    centroid_note: str = ""
    #: The transit fit of the primary signal, when one was asked for and ran.
    fit: FitResult | None = None
    fit_error: str | None = None

    @property
    def centroid_offset(self) -> bool | None:
        """True if any centroid test flags an offset; ``None`` if none was run."""
        if not self.centroids:
            return None
        return any(c.significant for c in self.centroids)

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
        out = {
            "target_id": self.target_id,
            "score": self.score,
            "threshold": self.threshold,
            "above_threshold": self.above_threshold,
            "verdict": self.verdict,
            "probability": self.probability,
            "planet_rate": self.planet_rate,
            "probability_note": PROBABILITY_NOTE,
            "primary_signal": self.primary,
            "candidates": [c.to_dict() for c in self.candidates],
            "features": self.features,
            "log_odds": self.log_odds,
            "base_log_odds": self.base_log_odds,
            "top_reasons": self.reasons,
            "contributions": self.contributions,
            "contribution_method": CONTRIBUTION_METHOD,
            "n_cadences": self.n_cadences,
            "baseline_days": self.baseline_days,
            "model": self.model_provenance,
        }
        if self.centroids is not None:
            out["centroid"] = {
                "offset_flag": self.centroid_offset,
                "note": self.centroid_note or CENTROID_NOTE,
                "tests": [c.to_dict() for c in self.centroids],
            }
        if self.fit is not None:
            out["fit"] = self.fit.to_dict()
        elif self.fit_error is not None:
            out["fit"] = {"error": self.fit_error}
        return _json_safe(out)


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


def feature_contributions(
    model: SavedModel, x: np.ndarray
) -> list[dict[str, float | str]]:
    """Every feature's SHAP value for one star, largest in size first.

    ``shap`` is in calibrated log-odds: positive pushed this star towards
    "planet", and ``model.base_log_odds`` plus the sum over all rows is the
    star's log-odds.  ``odds_factor = exp(shap)`` is the same number as a
    multiplier on the odds.  The training-split median is given beside each
    value for context; it plays no part in the calculation.
    """
    x = np.asarray(x, dtype=float).reshape(1, -1)
    values = model.explain(x)[0]
    rows = [
        {
            "feature": model.feature_names[j],
            "value": float(x[0, j]),
            "training_median": float(model.train_medians[j]),
            "shap": float(values[j]),
            "odds_factor": float(np.exp(values[j])),
        }
        for j in range(x.shape[1])
    ]
    rows.sort(key=lambda r: abs(float(r["shap"])), reverse=True)
    return rows


def vet_light_curve(
    lc: LightCurve,
    model: SavedModel,
    multi: MultiPlanetConfig | None = None,
    tpfs: list[TargetPixelData] | None = None,
    centroid_config: CentroidConfig | None = None,
    planet_rate: float | None = None,
) -> tuple[VetResult, FlattenedLightCurve]:
    """Detrend, search, featurise and score one light curve.

    ``planet_rate`` re-targets the calibrated probability to a population in
    which that fraction of stars host a planet; ``None`` keeps the training
    population's rate.  It moves the probability only, never the score or
    the verdict.

    With ``tpfs`` (an empty list counts: it records that none was found), the
    primary signal's ephemeris is also run through :func:`centroid_test` on
    each target pixel file.  The score is computed before and without it.
    """
    lc = lc.finite()
    flat, primary = detrend_and_search(lc, model.preprocess, model.bls)
    features = extract_features(flat, model.bls, search=primary)
    candidates = iterative_search(flat, model.bls, multi)

    x = np.array([features[name] for name in FEATURE_NAMES], dtype=float)
    score = float(model.score(x)[0])
    rate = model.calibration.train_positive_rate if planet_rate is None else planet_rate
    log_odds = float(model.log_odds(x, planet_rate)[0])
    # A different planet rate moves every star's log-odds by the same amount,
    # so it shifts the base value and leaves each SHAP value as it is.
    shift = log_odds - float(model.log_odds(x)[0])
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
        probability=float(model.probability(x, planet_rate)[0]),
        planet_rate=float(rate),
        log_odds=log_odds,
        base_log_odds=model.base_log_odds + shift,
    )
    if tpfs is not None:
        result.centroids = [
            centroid_test(
                tpf,
                primary["period"],
                primary["transit_time"],
                primary["duration"],
                centroid_config,
            )
            for tpf in tpfs
        ]
        if not tpfs:
            result.centroid_note = (
                "no target pixel file was available; " + CENTROID_NOTE
            )
    return result, flat


def fit_primary(
    result: VetResult,
    flat: FlattenedLightCurve,
    lc: LightCurve,
    config: FitConfig | None = None,
    *,
    stellar_density: tuple[float, float] | None = None,
    stellar_radius: float | None = None,
) -> FitResult | None:
    """Fit a transit model to the primary signal and attach it to ``result``.

    The stellar density and radius default to those the light curve carries
    (synthetic and injected curves do); survey curves need them given.  So
    does the exposure, unless the config sets one (see
    :func:`~transitml.fit.default_exposure_minutes`).  A signal too poorly
    sampled to fit leaves ``result.fit_error`` set instead.
    """
    meta_density, meta_radius = stellar_priors_from_meta(lc.meta)
    density = stellar_density if stellar_density is not None else meta_density
    radius = stellar_radius if stellar_radius is not None else meta_radius
    config = config or FitConfig()
    if config.exposure_minutes is None:
        config = replace(config, exposure_minutes=default_exposure_minutes(lc.meta))
    p = result.primary
    try:
        result.fit = fit_transit(
            flat, p["period"], p["epoch"], p["duration"], p["depth"], config,
            stellar_density=density, stellar_radius=radius,
        )
    except FitError as exc:
        result.fit_error = str(exc)
        return None
    if not result.candidates:
        result.fit.warnings.append(
            "no signal reached the search's significance threshold; this fits the "
            "strongest peak, which may be noise"
        )
    return result.fit


def write_fit_plot(result: VetResult, out_dir: str | Path, stem: str | None = None) -> Path | None:
    """Write ``<stem>_fit.png`` when the result carries a fit."""
    if result.fit is None:
        return None
    from .plots import plot_fit

    stem = stem or "vet_" + "".join(ch if ch.isalnum() else "_" for ch in result.target_id)
    return plot_fit(result.fit, Path(out_dir) / f"{stem}_fit.png")


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
    stem = stem or "vet_" + "".join(
        ch if ch.isalnum() else "_" for ch in result.target_id
    )
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
        raise SystemExit(
            f"{tic}: no light curve found on MAST (or the download failed)"
        )
    return curves


def load_target_pixels(
    target: str, args: argparse.Namespace
) -> list[TargetPixelData] | None:
    """Target pixel files for the centroid test: ``--tpf`` files, or a download.

    ``None`` when neither ``--tpf`` nor ``--centroids`` was given.
    """
    if args.tpf:
        return [load_tpf(path) for path in args.tpf]
    if not args.centroids:
        return None
    tic = tic_id(target)
    if Path(target).exists() or tic is None:
        raise SystemExit(
            "--centroids downloads target pixel files for a TIC ID; "
            "for a local light curve pass --tpf FILE.npz"
        )
    tpfs = download_tpfs(
        tic, author=args.author, exposure_time=args.exposure_time, sector=args.sector
    )
    if not tpfs:
        print(f"  {tic}: no target pixel file found on MAST; centroid test skipped")
    return tpfs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.vet",
        description="Score one light curve with a model trained by run_pipeline.py "
        "and write a one-page vetting report (PNG) plus its numbers (JSON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        help="A .csv or .npz light-curve file, or a TIC ID (downloaded from MAST).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results/model.joblib"),
        help="Model file written by run_pipeline.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/vet"),
        help="Where the report lands.",
    )
    parser.add_argument(
        "--target-id",
        default=None,
        help="Which star to take from a multi-star npz cache.",
    )
    parser.add_argument(
        "--sector", type=int, default=None, help="TESS sector (TIC input)."
    )
    parser.add_argument(
        "--author", default="TESS-SPOC", help="Light-curve pipeline (TIC input)."
    )
    parser.add_argument(
        "--exposure-time",
        type=int,
        default=1800,
        help="Cadence in seconds (TIC input).",
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Join every sector of the star into one curve before vetting.",
    )
    parser.add_argument(
        "--tpf",
        type=Path,
        action="append",
        default=None,
        metavar="FILE.npz",
        help="Target pixel file saved by transitml.data.tpf.save_tpf; runs the centroid "
        "test on the primary signal. Repeat for several sectors.",
    )
    parser.add_argument(
        "--centroids",
        action="store_true",
        help="For a TIC: download its target pixel files (same author, cadence and "
        "sector as the light curve) and run the centroid test.",
    )
    parser.add_argument(
        "--planet-rate",
        type=float,
        default=None,
        help="Fraction of stars like this one that host a planet, for the calibrated "
        "probability; None means the training population's rate. The score and verdict "
        "do not depend on it.",
    )
    parser.add_argument(
        "--max-signals",
        type=int,
        default=MultiPlanetConfig.max_signals,
        help="Most signals the iterative search reports.",
    )
    fit = parser.add_argument_group("transit fit (--fit)")
    fit.add_argument(
        "--fit",
        action="store_true",
        help="Fit the primary signal with batman and sample it with emcee.",
    )
    fit.add_argument(
        "--stellar-density",
        type=float,
        default=None,
        help="Host density in g/cm^3, to check the fitted one against (synthetic and "
        "injected curves carry it).",
    )
    fit.add_argument(
        "--stellar-density-err",
        type=float,
        default=None,
        help="Its uncertainty; default 10%% of the density.",
    )
    fit.add_argument(
        "--stellar-radius",
        type=float,
        default=None,
        help="Host radius in solar radii, for the planet radius in Earth radii.",
    )
    fit.add_argument(
        "--exposure-minutes",
        type=float,
        default=None,
        help="Exposure the model integrates over; default the median cadence, 0 for none.",
    )
    fit.add_argument(
        "--fit-max-steps",
        type=int,
        default=FitConfig.max_steps,
        help="Longest chain before giving up on convergence (a shorter one also lowers "
        "the minimum, for a quick look).",
    )
    parser.add_argument(
        "--min-sde",
        type=float,
        default=MultiPlanetConfig.min_sde,
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
            raise SystemExit(
                f"cannot stitch: {exc}; pick one star with --target-id"
            ) from exc
    if len(curves) != 1:
        found = ", ".join(
            f"{lc.target_id} (sector {lc.meta.get('sector', '?')})" for lc in curves
        )
        raise SystemExit(
            f"{len(curves)} light curves found ({found}); pick one with --target-id "
            "or --sector, or join one star's sectors with --stitch"
        )
    lc = curves[0]
    tpfs = load_target_pixels(args.target, args)
    model = load_model(args.model)
    multi = MultiPlanetConfig(max_signals=args.max_signals, min_sde=args.min_sde)
    if args.planet_rate is not None and not 0.0 < args.planet_rate < 1.0:
        raise SystemExit(f"--planet-rate must lie between 0 and 1, got {args.planet_rate}")
    result, flat = vet_light_curve(lc, model, multi, tpfs, planet_rate=args.planet_rate)
    if args.fit:
        density = None
        if args.stellar_density is not None:
            sd = args.stellar_density_err or 0.1 * args.stellar_density
            density = (args.stellar_density, sd)
        config = FitConfig(
            exposure_minutes=args.exposure_minutes,
            min_steps=min(FitConfig.min_steps, args.fit_max_steps),
            max_steps=args.fit_max_steps,
        )
        fit_primary(
            result, flat, lc, config, stellar_density=density, stellar_radius=args.stellar_radius
        )
    png, js = write_report(lc, flat, result, args.out_dir)
    fit_png = write_fit_plot(result, args.out_dir)

    print(
        f"{result.target_id}: score {result.score:.3f}, threshold {result.threshold:.3f}"
    )
    print(f"  verdict: {result.verdict}")
    print(
        f"  P(planet) = {result.probability:.3f} at a {result.planet_rate:.1%} planet rate "
        "(calibrated)"
    )
    reasons = ", ".join(f"{r['feature']} {float(r['shap']):+.2f}" for r in result.reasons[:3])
    print(f"  top reasons (SHAP, log-odds from a base of {result.base_log_odds:+.2f}): {reasons}")
    for cand in result.candidates:
        print(
            f"  signal {cand.rank}: P = {cand.period:.4f} d, depth {cand.depth * 1e6:.0f} ppm, "
            f"duration {cand.duration * 24:.2f} h, SDE {cand.sde:.1f}, SNR {cand.depth_snr:.1f}"
        )
    if not result.candidates:
        print(f"  no signal reached SDE {multi.min_sde:g}")
    for test in result.centroids or []:
        sector = test.meta.get("sector")
        label = f"sector {sector}" if sector is not None else test.target_id
        print(f"  centroid ({label}): {test.verdict}")
    if result.centroid_offset:
        print(
            "  FLAG: significant centroid offset; the signal is likely on a neighbour "
            "(the score above does not include this test)"
        )
    if result.fit is not None:
        for line in fit_summary_lines(result.fit):
            print("  " + line)
    elif result.fit_error is not None:
        print(f"  fit: not run ({result.fit_error})")
    print(f"  wrote {png} and {js}" + (f", and {fit_png}" if fit_png else ""))
    return 0


def fit_summary_lines(fit: FitResult) -> list[str]:
    """The fit in a few printable lines."""
    p = fit.parameters

    def show(name: str, digits: int) -> str:
        s = p[name]
        return (
            f"{s['median']:.{digits}f} +{s['upper'] - s['median']:.{digits}f} "
            f"-{s['median'] - s['lower']:.{digits}f}"
        )

    lines = [
        f"fit: Rp/R* = {show('k', 4)}, b = {show('b', 2)}, T14 = {show('t14_hours', 2)} h, "
        f"depth = {show('depth_ppm', 0)} ppm",
        f"     stellar density from the transit = {show('rho_star', 2)} g/cm^3"
        + (f", Rp = {show('rp_earth', 2)} Earth radii" if "rp_earth" in p else ""),
    ]
    check = fit.density_check
    if check is not None:
        verdict = "consistent with" if check["consistent"] else "INCONSISTENT with"
        lines.append(
            f"     {verdict} the star's {check['stellar_density']:.2f} g/cm^3 "
            f"(ratio {check['ratio']['median']:.2f})"
        )
    if not fit.converged:
        lines.append("     the chain did not converge; treat the intervals as rough")
    lines += [f"     warning: {w}" for w in fit.warnings if "autocorrelation" not in w]
    return lines


if __name__ == "__main__":
    sys.exit(main())
