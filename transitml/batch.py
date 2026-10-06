"""Batch mode: vet every star in a sector, cache the work, rank the candidates.

    python run_pipeline.py                                   # trains results/model.joblib
    python -m transitml.batch --synthetic 2000               # an offline demo sector
    python -m transitml.batch --targets s14.txt --sector 14  # real stars, from MAST
    python -m transitml.batch curves.npz more_curves/        # light-curve files

Each star goes through exactly what ``python -m transitml.vet`` does to one:
detrend, search, featurise, score with the trained model, calibrate, explain.
The batch writes, to ``--out-dir``:

``candidates.csv``  every star, ranked by score, with P(planet), the primary
                    signal and the three SHAP reasons that moved it most;
``summary.json``    counts, the model used, how much came from the cache and,
                    when the stars carry ground truth (synthetic or injected),
                    how the candidate list scores against it;
``dashboard.html``  one self-contained page: sortable, filterable candidate
                    table with a folded light curve per row (no network, no
                    server, open it from disk);
``reports/``        the full one-page ``vet`` report for the top candidates.

With ``--fit N`` the ``N`` best-ranked flagged stars are also fitted with a
limb-darkened transit model and MCMC (:mod:`transitml.fit`); the radius
ratio, impact parameter, duration, depth and implied stellar density, with
intervals, go into the CSV, the summary and the dashboard.  Fits are cached
in ``cache/fits.jsonl`` the same way, keyed by the star and the fit settings.

**Caching.**  Two layers, so a sector can be stopped and resumed and a rerun
does only what changed.  Downloaded light curves are saved in chunks as they
arrive (``curves/``), with the targets MAST had nothing for remembered, so an
interrupted download restarts where it stopped.  Each star's result is
appended to ``cache/results.jsonl`` the moment it is computed, keyed by a hash
of its light curve, the model file and the search settings; a rerun skips
every star whose key is already there, and a new model or a changed curve
invalidates exactly the stars it touches.  ``--planet-rate`` is applied at
output time, so changing it costs nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit, logit

from .config import MultiPlanetConfig, PreprocessConfig, default_config
from .data.base import LightCurve
from .data.files import read_light_curves
from .data.injection import load_curves, read_target_list, save_curves
from .fit import FitConfig
from .model import SavedModel, load_model

#: Bumped when a cached row's layout changes, so old rows are recomputed.
BATCH_FORMAT_VERSION = 1
#: The same for cached fits.
FIT_FORMAT_VERSION = 1
#: Fitted quantities kept per star, as (median, lower, upper) of the 68% interval.
FIT_KEPT: tuple[str, ...] = (
    "period", "t0", "k", "b", "rho_star", "t14_hours", "depth_ppm", "rp_earth",
)

#: Bins in the folded light curve kept per star for the dashboard thumbnail.
FOLD_BINS = 40

#: SHAP reasons kept per star.
N_BATCH_REASONS = 3


# --------------------------------------------------------------------------
# Fingerprints: what a cached result depends on
# --------------------------------------------------------------------------
def curve_fingerprint(lc: LightCurve) -> str:
    """Hash of the light curve's arrays: any change to the data changes it."""
    digest = hashlib.sha256()
    for array in (lc.time, lc.flux, lc.flux_err):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()[:20]


def model_fingerprint(path: str | Path) -> str:
    """Hash of the model file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:20]


def result_key(lc: LightCurve, model_hash: str, multi: MultiPlanetConfig) -> str:
    """Cache key for one star's result."""
    settings = f"v{BATCH_FORMAT_VERSION}|{multi.max_signals}|{multi.min_sde:g}"
    sector = lc.meta.get("sector", "")
    return "|".join((lc.target_id, str(sector), curve_fingerprint(lc), model_hash, settings))


# --------------------------------------------------------------------------
# One star
# --------------------------------------------------------------------------
def _finite_or_none(value: Any, digits: int = 6) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


def folded_profile(flat, period: float, epoch: float, duration: float) -> dict[str, Any]:
    """The detrended flux folded on the primary signal, binned for a thumbnail.

    ``half_window_hours`` is the half-width shown; ``ppt`` holds the binned
    mean flux minus one in parts per thousand, ``None`` where a bin is empty.
    """
    if not (np.isfinite(period) and np.isfinite(epoch) and np.isfinite(duration)):
        return {"half_window_hours": None, "duration_hours": None, "ppt": []}
    half = min(4.0 * duration, 0.5 * period) * 24.0
    hours = ((flat.time - epoch + 0.5 * period) % period - 0.5 * period) * 24.0
    ppt = (flat.flux - 1.0) * 1e3
    edges = np.linspace(-half, half, FOLD_BINS + 1)
    index = np.digitize(hours, edges) - 1
    keep = (index >= 0) & (index < FOLD_BINS)
    sums = np.bincount(index[keep], weights=ppt[keep], minlength=FOLD_BINS)
    counts = np.bincount(index[keep], minlength=FOLD_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return {
        "half_window_hours": _finite_or_none(half, 4),
        "duration_hours": _finite_or_none(duration * 24.0, 4),
        "ppt": [_finite_or_none(v, 4) for v in means],
    }


def vet_row(lc: LightCurve, model: SavedModel, multi: MultiPlanetConfig) -> dict[str, Any]:
    """Vet one star and return its result as a plain, JSON-ready row.

    Never raises for a bad light curve: the row comes back with ``status``
    ``"error"`` and the message, so one broken star cannot stop a sector.
    """
    from .vet import vet_light_curve  # vet imports matplotlib-free modules only

    row: dict[str, Any] = {
        "target_id": lc.target_id,
        "sector": lc.meta.get("sector"),
        "status": "ok",
    }
    truth = _truth(lc)
    if truth:
        row["truth"] = truth
    started = time.perf_counter()
    try:
        result, flat = vet_light_curve(lc, model, multi)
    except Exception as exc:  # noqa: BLE001 - any failure is recorded per star
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = round(time.perf_counter() - started, 3)
        return row
    primary = result.primary
    row.update(
        n_cadences=result.n_cadences,
        baseline_days=_finite_or_none(result.baseline_days, 5),
        score=_finite_or_none(result.score, 8),
        log_odds=_finite_or_none(result.log_odds, 8),
        period_days=_finite_or_none(primary["period"], 8),
        epoch=_finite_or_none(primary["epoch"], 9),
        duration_hours=_finite_or_none(primary["duration"] * 24.0, 5),
        depth_ppm=_finite_or_none(primary["depth"] * 1e6, 5),
        depth_snr=_finite_or_none(primary["depth_snr"], 5),
        sde=_finite_or_none(primary["sde"], 5),
        signals=[
            {
                "period_days": _finite_or_none(c.period, 8),
                "depth_ppm": _finite_or_none(c.depth * 1e6, 5),
                "duration_hours": _finite_or_none(c.duration * 24.0, 5),
                "sde": _finite_or_none(c.sde, 5),
                "depth_snr": _finite_or_none(c.depth_snr, 5),
            }
            for c in result.candidates
        ],
        reasons=[
            {
                "feature": str(r["feature"]),
                "value": _finite_or_none(r["value"], 5),
                "shap": _finite_or_none(r["shap"], 5),
            }
            for r in result.contributions[:N_BATCH_REASONS]
        ],
        fold=folded_profile(flat, primary["period"], primary["epoch"], primary["duration"]),
    )
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


def _truth(lc: LightCurve) -> dict[str, Any] | None:
    """Ground truth carried by synthetic or injected curves; ``None`` for survey data."""
    if lc.label is None:
        return None
    out: dict[str, Any] = {"label": int(lc.label)}
    if "kind" in lc.meta:
        out["kind"] = str(lc.meta["kind"])
    for key in ("period", "true_snr"):
        if key in lc.meta:
            out[key] = _finite_or_none(lc.meta[key], 6)
    return out


@lru_cache(maxsize=2)
def _cached_model(path: str, fingerprint: str) -> SavedModel:
    """One model load per worker process (the fingerprint keys a changed file)."""
    return load_model(path)


def _work(lc: LightCurve, key: str, model_path: str, fingerprint: str, multi) -> dict[str, Any]:
    row = vet_row(lc, _cached_model(model_path, fingerprint), multi)
    row["key"] = key
    return row


# --------------------------------------------------------------------------
# Transit fits for the top candidates
# --------------------------------------------------------------------------
def fit_key(star_key: str, config: FitConfig) -> str:
    """Cache key of one star's fit: the star's result key plus every fit setting."""
    settings = {"format": FIT_FORMAT_VERSION, **{k: v for k, v in vars(config).items()}}
    blob = star_key + json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def fit_row(
    lc: LightCurve, row: dict[str, Any], preprocess: PreprocessConfig, config: FitConfig
) -> dict[str, Any]:
    """Fit one star's primary signal; a failure is recorded, never raised."""
    from .fit import default_exposure_minutes, fit_transit, stellar_priors_from_meta
    from .preprocess import flatten

    out: dict[str, Any] = {"key": fit_key(row["key"], config), "star_key": row["key"], "status": "ok"}
    started = time.perf_counter()
    try:
        if config.exposure_minutes is None:
            config = replace(config, exposure_minutes=default_exposure_minutes(lc.meta))
        density, radius = stellar_priors_from_meta(lc.meta)
        fit = fit_transit(
            flatten(lc.finite(), preprocess),
            row["period_days"], row["epoch"], row["duration_hours"] / 24.0,
            row["depth_ppm"] / 1e6, config,
            stellar_density=density, stellar_radius=radius,
        )
    except Exception as exc:  # noqa: BLE001 - one star's failure stays with that star
        out.update(status="error", error=f"{type(exc).__name__}: {exc}")
        out["seconds"] = round(time.perf_counter() - started, 2)
        return out
    out["parameters"] = {
        name: [_finite_or_none(fit.parameters[name][q], 7) for q in ("median", "lower", "upper")]
        for name in FIT_KEPT
        if name in fit.parameters
    }
    out["converged"] = fit.converged
    out["autocorr_time"] = _finite_or_none(fit.sampler["autocorr_time_max"], 4)
    out["beta"] = _finite_or_none(fit.noise["beta"], 4)
    out["warnings"] = list(fit.warnings)
    check = fit.density_check
    if check is not None:
        out["density_ratio"] = [
            _finite_or_none(check["ratio"][q], 5) for q in ("median", "lower", "upper")
        ]
        out["density_consistent"] = bool(check["consistent"])
    out["seconds"] = round(time.perf_counter() - started, 2)
    return out


def _fit_candidates(
    rows: list[dict[str, Any]],
    curves: Sequence[LightCurve],
    keys: Sequence[str],
    model: SavedModel,
    out_dir: Path,
    n_fits: int,
    config: FitConfig,
    n_jobs: int,
    progress: bool,
) -> dict[str, Any] | None:
    """Fit the ``n_fits`` best-ranked flagged stars, reusing cached fits; sets ``row["fit"]``."""
    if n_fits <= 0:
        return None
    targets = [r for r in rows if r["status"] == "ok" and r["flagged"]][:n_fits]
    by_key = dict(zip(keys, curves))
    cache = ResultCache(out_dir / "cache" / "fits.jsonl")
    wanted = {r["key"]: fit_key(r["key"], config) for r in targets}
    todo = [r for r in targets if wanted[r["key"]] not in cache]
    if progress and targets:
        print(f"  fitting {len(targets)} candidates: {len(targets) - len(todo)} from the cache, "
              f"{len(todo)} to fit")
    try:
        if todo:
            jobs = (delayed(fit_row)(by_key[r["key"]], r, model.preprocess, config) for r in todo)
            stream = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs)
            for done, fitted in enumerate(stream, start=1):
                cache.add(fitted)
                if progress and (done % 5 == 0 or done == len(todo)):
                    print(f"  fitted {done}/{len(todo)}")
        current = set(keys)
        cache.compact([k for k, row in cache.rows.items() if row.get("star_key") in current])
    finally:
        cache.close()
    for row in targets:
        row["fit"] = cache.get(wanted[row["key"]])
    fits = [r["fit"] for r in targets]
    ok = [f for f in fits if f["status"] == "ok"]
    by_star = {r["key"]: r["id"] for r in targets}
    return {
        "requested": n_fits,
        "fitted": len(ok),
        "errors": len(fits) - len(ok),
        "computed": len(todo),
        "from_cache": len(targets) - len(todo),
        "converged": sum(bool(f.get("converged")) for f in ok),
        "density_inconsistent": [
            by_star[f["star_key"]] for f in ok if f.get("density_consistent") is False
        ],
        "settings": {
            "max_steps": config.max_steps,
            "n_walkers": config.n_walkers,
            "exposure_minutes": config.exposure_minutes,
        },
    }


# --------------------------------------------------------------------------
# The result cache
# --------------------------------------------------------------------------
class ResultCache:
    """Append-only JSON-lines file of per-star results, keyed by :func:`result_key`.

    Rows are appended and flushed as they arrive, so whatever finished before
    an interruption is kept.  A torn last line (the process killed mid-write)
    is skipped on load.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and "key" in row:
                    self.rows[row["key"]] = row
        self._handle = None

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def get(self, key: str) -> dict[str, Any]:
        return self.rows[key]

    def add(self, row: dict[str, Any]) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "a")
        self._handle.write(json.dumps(row, allow_nan=False) + "\n")
        self._handle.flush()
        self.rows[row["key"]] = row

    def compact(self, keep: Iterable[str]) -> None:
        """Rewrite the file with only ``keep``: drops rows for old models or curves."""
        self.close()
        keep = [k for k in keep if k in self.rows]
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as handle:
            for key in keep:
                handle.write(json.dumps(self.rows[key], allow_nan=False) + "\n")
        os.replace(tmp, self.path)
        self.rows = {k: self.rows[k] for k in keep}

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# --------------------------------------------------------------------------
# A whole sector
# --------------------------------------------------------------------------
@dataclass
class BatchResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    out_dir: Path


def _unique_ids(curves: Sequence[LightCurve]) -> list[str]:
    """Display ids: the target id, plus the sector when one star appears twice."""
    counts: dict[str, int] = {}
    for lc in curves:
        counts[lc.target_id] = counts.get(lc.target_id, 0) + 1
    out = []
    for lc in curves:
        sector = lc.meta.get("sector")
        if counts[lc.target_id] > 1 and sector is not None:
            out.append(f"{lc.target_id} s{sector}")
        else:
            out.append(lc.target_id)
    return out


def run_batch(
    curves: Sequence[LightCurve],
    model_path: str | Path,
    out_dir: str | Path,
    *,
    source: str,
    n_jobs: int = -1,
    multi: MultiPlanetConfig | None = None,
    planet_rate: float | None = None,
    n_reports: int = 10,
    force: bool = False,
    progress: bool = True,
    n_fits: int = 0,
    fit_config: FitConfig | None = None,
) -> BatchResult:
    """Vet ``curves``, reusing cached results, and write every output."""
    started = time.time()
    multi = multi or MultiPlanetConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_path)
    model = load_model(model_path)
    fingerprint = model_fingerprint(model_path)
    if planet_rate is not None and not 0.0 < planet_rate < 1.0:
        raise ValueError(f"planet_rate must lie in (0, 1), got {planet_rate}")

    keys = [result_key(lc, fingerprint, multi) for lc in curves]
    if len(set(keys)) != len(keys):
        raise ValueError("the same light curve appears twice in the batch")
    cache = ResultCache(out_dir / "cache" / "results.jsonl")
    todo = [i for i, key in enumerate(keys) if force or key not in cache]
    n_cached = len(curves) - len(todo)
    if progress:
        print(f"  {len(curves)} stars: {n_cached} from the cache, {len(todo)} to vet")

    try:
        if todo:
            jobs = (
                delayed(_work)(curves[i], keys[i], model_path, fingerprint, multi) for i in todo
            )
            stream = Parallel(n_jobs=n_jobs, return_as="generator_unordered", batch_size=4)(jobs)
            for done, row in enumerate(stream, start=1):
                cache.add(row)
                if progress and (done % 200 == 0 or done == len(todo)):
                    print(f"  vetted {done}/{len(todo)}")
        cache.compact(keys)
    finally:
        cache.close()

    ids = _unique_ids(curves)
    rows = []
    for key, uid in zip(keys, ids):
        row = dict(cache.get(key))
        row["id"] = uid
        rows.append(row)
    rows = rank_rows(rows, model, planet_rate)
    fits = _fit_candidates(
        rows, curves, keys, model, out_dir, n_fits, fit_config or FitConfig(), n_jobs, progress
    )

    report_paths = _write_reports(rows, curves, keys, model, multi, out_dir, n_reports)
    summary = summarise(
        rows,
        model,
        planet_rate=planet_rate,
        source=source,
        model_path=model_path,
        fingerprint=fingerprint,
        multi=multi,
        computed=len(todo),
        cached=n_cached,
        runtime=time.time() - started,
        reports=report_paths,
    )
    if fits is not None:
        summary["fits"] = fits
    write_candidates_csv(rows, out_dir / "candidates.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    from .dashboard import write_dashboard

    write_dashboard(rows, summary, out_dir / "dashboard.html")
    return BatchResult(rows=rows, summary=summary, out_dir=out_dir)


def rank_rows(
    rows: list[dict[str, Any]], model: SavedModel, planet_rate: float | None
) -> list[dict[str, Any]]:
    """Add ``p_planet``, ``flagged`` and ``rank``; sort by score, errors last."""
    train_rate = model.calibration.train_positive_rate
    shift = 0.0 if planet_rate is None else float(logit(planet_rate) - logit(train_rate))
    for row in rows:
        if row["status"] == "ok" and row.get("score") is not None:
            row["p_planet"] = _finite_or_none(expit(row["log_odds"] + shift), 6)
            row["flagged"] = bool(row["score"] >= model.threshold)
        else:
            row["p_planet"] = None
            row["flagged"] = False
    ok = [r for r in rows if r["p_planet"] is not None]
    bad = [r for r in rows if r["p_planet"] is None]
    ok.sort(key=lambda r: (-r["score"], r["id"]))
    for rank, row in enumerate(ok, start=1):
        row["rank"] = rank
    for row in bad:
        row["rank"] = None
    return ok + bad


def summarise(
    rows: list[dict[str, Any]],
    model: SavedModel,
    *,
    planet_rate: float | None,
    source: str,
    model_path: str,
    fingerprint: str,
    multi: MultiPlanetConfig,
    computed: int,
    cached: int,
    runtime: float,
    reports: list[str],
) -> dict[str, Any]:
    """The numbers at the top of the dashboard, as JSON."""
    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] != "ok"]
    flagged = [r for r in ok if r["flagged"]]
    rate = model.calibration.train_positive_rate if planet_rate is None else planet_rate
    threshold_p = float(
        expit(
            model.calibration.log_odds(np.array([logit(model.threshold)]), planet_rate)[0]
        )
    )
    summary: dict[str, Any] = {
        "format_version": BATCH_FORMAT_VERSION,
        "source": source,
        "n_stars": len(rows),
        "n_vetted": len(ok),
        "n_errors": len(errors),
        "errors": [{"id": r["id"], "error": r.get("error", "")} for r in errors[:20]],
        "n_flagged": len(flagged),
        "expected_planets_flagged": round(sum(r["p_planet"] for r in flagged), 3),
        "expected_planets_all": round(sum(r["p_planet"] for r in ok), 3),
        "threshold": model.threshold,
        "threshold_probability": threshold_p,
        "planet_rate": rate,
        "training_planet_rate": model.calibration.train_positive_rate,
        "model": {"path": model_path, "fingerprint": fingerprint, **model.provenance},
        "settings": {"max_signals": multi.max_signals, "min_sde": multi.min_sde},
        "computed": computed,
        "from_cache": cached,
        "runtime_seconds": round(runtime, 1),
        "reports": reports,
    }
    truth = truth_summary(ok)
    if truth is not None:
        summary["truth"] = truth
    return summary


def truth_summary(rows: list[dict[str, Any]], k: int = 20) -> dict[str, Any] | None:
    """How the ranked list does against ground truth, when every star has it."""
    if not rows or any("truth" not in r for r in rows):
        return None
    from sklearn.metrics import average_precision_score

    y = np.array([r["truth"]["label"] for r in rows])
    score = np.array([r["score"] for r in rows])
    flagged = np.array([r["flagged"] for r in rows])
    p = np.array([r["p_planet"] for r in rows])
    n_pos = int(y.sum())
    kinds: dict[str, int] = {}
    for r, f in zip(rows, flagged):
        if f and r["truth"]["label"] == 0:
            kind = r["truth"].get("kind", "unknown")
            kinds[kind] = kinds.get(kind, 0) + 1
    order = np.argsort(-score, kind="stable")
    return {
        "n_planets": n_pos,
        "planet_rate": float(y.mean()),
        "flagged_planets": int((y & flagged).sum()),
        "precision": float(y[flagged].mean()) if flagged.any() else None,
        "recall": float((y & flagged).sum() / n_pos) if n_pos else None,
        "average_precision": float(average_precision_score(y, score)) if 0 < n_pos < y.size else None,
        f"precision_at_{k}": float(y[order[:k]].mean()),
        "false_positives_by_kind": kinds,
        "expected_planets_all": float(p.sum()),
    }


def write_candidates_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """One line per star, ranked: the list a follow-up programme would start from."""
    columns = [
        "rank", "id", "target_id", "sector", "p_planet", "score", "flagged",
        "period_days", "epoch", "duration_hours", "depth_ppm", "depth_snr", "sde",
        "n_signals", "reason_1", "reason_2", "reason_3", "status", "error",
    ]
    has_truth = any("truth" in r for r in rows)
    if has_truth:
        columns += ["truth_label", "truth_kind"]
    has_fit = any("fit" in r for r in rows)
    if has_fit:
        for name in _FIT_COLUMNS:
            columns += [f"fit_{_FIT_COLUMNS[name]}", f"fit_{_FIT_COLUMNS[name]}_err"]
        columns += ["fit_density_ratio", "fit_density_consistent", "fit_converged", "fit_status"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            reasons = [
                f"{r['feature']} {r['shap']:+.2f}" for r in row.get("reasons", []) if r["shap"] is not None
            ]
            reasons += [""] * (3 - len(reasons))
            values = {
                **row,
                "n_signals": len(row.get("signals", [])) if row["status"] == "ok" else "",
                "reason_1": reasons[0],
                "reason_2": reasons[1],
                "reason_3": reasons[2],
            }
            if has_truth:
                truth = row.get("truth", {})
                values["truth_label"] = truth.get("label", "")
                values["truth_kind"] = truth.get("kind", "")
            if has_fit and "fit" in row:
                values.update(_fit_csv_values(row["fit"]))
            writer.writerow(["" if values.get(c) is None else values.get(c, "") for c in columns])
    return path


#: Fitted quantities in candidates.csv, and their column names.
_FIT_COLUMNS = {
    "k": "rp_rs", "b": "b", "rho_star": "rho_star", "t14_hours": "t14_hours",
    "depth_ppm": "depth_ppm", "rp_earth": "rp_earth",
}


def _fit_csv_values(fit: dict[str, Any]) -> dict[str, Any]:
    """A fit as CSV cells: the median and half the 68% interval of each quantity."""
    out: dict[str, Any] = {"fit_status": fit["status"]}
    if fit["status"] != "ok":
        return out
    for name, column in _FIT_COLUMNS.items():
        value = fit["parameters"].get(name)
        if value is None or None in value:
            continue
        median, lower, upper = value
        out[f"fit_{column}"] = median
        out[f"fit_{column}_err"] = float(f"{(upper - lower) / 2.0:.4g}")
    if "density_ratio" in fit:
        out["fit_density_ratio"] = fit["density_ratio"][0]
        out["fit_density_consistent"] = fit["density_consistent"]
    out["fit_converged"] = fit["converged"]
    return out


def _write_reports(
    rows: list[dict[str, Any]],
    curves: Sequence[LightCurve],
    keys: Sequence[str],
    model: SavedModel,
    multi: MultiPlanetConfig,
    out_dir: Path,
    n_reports: int,
) -> list[str]:
    """Full one-page ``vet`` reports for the ``n_reports`` best-ranked flagged stars."""
    if n_reports <= 0:
        return []
    from .vet import vet_light_curve, write_report

    by_key = dict(zip(keys, curves))
    written = []
    for row in [r for r in rows if r["flagged"]][:n_reports]:
        lc = by_key[row["key"]]
        result, flat = vet_light_curve(lc, model, multi)
        stem = "vet_" + "".join(ch if ch.isalnum() else "_" for ch in row["id"])
        png, _ = write_report(lc, flat, result, out_dir / "reports", stem)
        row["report"] = f"reports/{png.name}"
        written.append(row["report"])
    return written


# --------------------------------------------------------------------------
# Where the light curves come from
# --------------------------------------------------------------------------
def synthetic_sector(n: int, seed: int) -> list[LightCurve]:
    """A sector of synthetic stars from the training generator, with its default rates.

    Use a seed other than the one the model was trained on (42 by default)
    and these are stars the model has never seen, with ground truth.
    """
    from .data.synthetic import SyntheticTESSSource

    config = default_config()
    source = SyntheticTESSSource(
        n,
        config.dataset.positive_rate,
        config.dataset.eclipsing_binary_rate,
        seed=seed,
        survey=config.survey,
        noise=config.noise,
        star=config.star,
        planet=config.planet,
        eb=config.eb,
    )
    return [source.generate(i) for i in range(n)]


def read_curve_inputs(paths: Sequence[str | Path]) -> list[LightCurve]:
    """Light curves from ``.csv`` and ``.npz`` files, or directories of them."""
    curves: list[LightCurve] = []
    for path in map(Path, paths):
        if path.is_dir():
            files = sorted(
                p for p in path.iterdir() if p.suffix.lower() in (".csv", ".npz") and p.is_file()
            )
            if not files:
                raise ValueError(f"{path}: no .csv or .npz light-curve files")
            for file in files:
                curves.extend(read_light_curves(file))
        elif path.exists():
            curves.extend(read_light_curves(path))
        else:
            raise ValueError(f"{path}: no such file or directory")
    return curves


def fetch_sector_curves(
    targets: Sequence[str],
    sector: int | None,
    cache_dir: str | Path,
    *,
    author: str = "TESS-SPOC",
    exposure_time: int | None = 1800,
    n_workers: int = 8,
    chunk_size: int = 200,
    progress: bool = True,
) -> list[LightCurve]:
    """One light curve per target from MAST, cached in chunks as they arrive.

    Every ``chunk_size`` targets the downloaded curves are written to
    ``cache_dir/chunk_NNNN.npz`` and the targets tried are added to
    ``cache_dir/tried.json``, so a download that dies part way resumes at the
    next untried target, and targets MAST has nothing for are not asked for
    again.  Keyed by target and sector, so one cache can serve several sectors.
    """
    from .data.mast import MASTLightCurveSource

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tried_path = cache_dir / "tried.json"
    tried = set(json.loads(tried_path.read_text())) if tried_path.exists() else set()

    def key(target_id: str) -> str:
        return f"{target_id}:{sector}"

    have: dict[str, LightCurve] = {}
    for chunk in sorted(cache_dir.glob("chunk_*.npz")):
        for lc in load_curves(chunk):
            have.setdefault(f"{lc.target_id}:{lc.meta.get('requested_sector')}", lc)
    missing = [t for t in dict.fromkeys(targets) if key(t) not in tried and key(t) not in have]
    n_chunks = len(list(cache_dir.glob("chunk_*.npz")))
    for start in range(0, len(missing), chunk_size):
        batch = missing[start : start + chunk_size]
        source = MASTLightCurveSource(
            [(t, None) for t in batch],
            mission="TESS",
            author=author,
            exposure_time=exposure_time,
            sector=sector,
            n_workers=n_workers,
        )
        fetched: dict[str, LightCurve] = {}
        for lc in source:
            if lc.target_id not in fetched:  # one curve per star and sector
                meta = {**lc.meta, "requested_sector": sector}
                fetched[lc.target_id] = replace(lc, meta=meta)
        if fetched:
            save_curves(list(fetched.values()), cache_dir / f"chunk_{n_chunks:04d}.npz")
            n_chunks += 1
        for lc in fetched.values():
            have[key(lc.target_id)] = lc
        tried |= {key(t) for t in batch}
        tried_path.write_text(json.dumps(sorted(tried)))
        if progress:
            print(f"  downloaded {start + len(batch)}/{len(missing)} targets ({len(fetched)} found)")
    return [have[key(t)] for t in dict.fromkeys(targets) if key(t) in have]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.batch",
        description="Vet every star in a sector with a model trained by run_pipeline.py, "
        "caching the work, and write a ranked candidate list and a dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="*", type=Path, help="Light-curve .csv/.npz files or directories of them."
    )
    source = parser.add_argument_group("other sources (instead of files)")
    source.add_argument(
        "--targets", type=Path, default=None,
        help="Target list (one TIC per line, or a CSV with a TIC column) to download from MAST.",
    )
    source.add_argument("--sector", type=int, default=None, help="TESS sector for --targets.")
    source.add_argument(
        "--synthetic", type=int, default=None, metavar="N",
        help="Vet N synthetic stars (an offline demo sector, with ground truth).",
    )
    source.add_argument(
        "--seed", type=int, default=7,
        help="Seed for --synthetic; the default model was trained on seed 42.",
    )
    parser.add_argument(
        "--model", type=Path, default=Path("results/model.joblib"),
        help="Model file written by run_pipeline.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: results/batch/<source>.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Worker processes.")
    parser.add_argument(
        "--planet-rate", type=float, default=None,
        help="Planet rate the probabilities are stated for; None means the training rate.",
    )
    parser.add_argument(
        "--reports", type=int, default=10, help="Full vet reports for this many top candidates."
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached results.")
    parser.add_argument(
        "--fit", type=int, default=0, metavar="N",
        help="Fit a transit model (batman + emcee) to the N best-ranked flagged stars.",
    )
    parser.add_argument(
        "--fit-max-steps", type=int, default=FitConfig.max_steps,
        help="Longest MCMC chain per fit before giving up on convergence.",
    )
    parser.add_argument(
        "--max-signals", type=int, default=MultiPlanetConfig.max_signals,
        help="Most signals the iterative search reports per star.",
    )
    parser.add_argument(
        "--min-sde", type=float, default=MultiPlanetConfig.min_sde,
        help="Significance a signal needs to be listed.",
    )
    mast = parser.add_argument_group("MAST download (--targets)")
    mast.add_argument("--author", default="TESS-SPOC")
    mast.add_argument("--exposure-time", type=int, default=1800)
    mast.add_argument("--download-workers", type=int, default=8)
    mast.add_argument(
        "--curve-cache", type=Path, default=None, help="Default: <out-dir>/curves."
    )
    args = parser.parse_args(argv)
    chosen = sum([bool(args.inputs), args.targets is not None, args.synthetic is not None])
    if chosen != 1:
        parser.error("give light-curve files, or --targets with --sector, or --synthetic N")
    if args.planet_rate is not None and not 0.0 < args.planet_rate < 1.0:
        parser.error(f"--planet-rate must lie between 0 and 1, got {args.planet_rate}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"{args.model}: no model file; run `python run_pipeline.py` first")
    if args.synthetic is not None:
        name = f"synthetic_seed{args.seed}"
        description = f"{args.synthetic} synthetic stars, seed {args.seed}"
    elif args.targets is not None:
        name = f"sector{args.sector}" if args.sector is not None else args.targets.stem
        description = f"{args.targets.name}, TESS sector {args.sector}"
    else:
        name = args.inputs[0].stem
        description = ", ".join(p.name for p in args.inputs)
    out_dir = args.out_dir or Path("results") / "batch" / name
    print(f"batch: {description} -> {out_dir}")

    if args.synthetic is not None:
        curves = synthetic_sector(args.synthetic, args.seed)
    elif args.targets is not None:
        targets = read_target_list(args.targets)
        curves = fetch_sector_curves(
            targets,
            args.sector,
            args.curve_cache or out_dir / "curves",
            author=args.author,
            exposure_time=args.exposure_time,
            n_workers=args.download_workers,
        )
        print(f"  {len(curves)} of {len(targets)} targets have a light curve")
    else:
        curves = read_curve_inputs(args.inputs)
    if not curves:
        raise SystemExit("no light curves to vet")

    result = run_batch(
        curves,
        args.model,
        out_dir,
        source=description,
        n_jobs=args.n_jobs,
        multi=MultiPlanetConfig(max_signals=args.max_signals, min_sde=args.min_sde),
        planet_rate=args.planet_rate,
        n_reports=args.reports,
        force=args.force,
        n_fits=args.fit,
        fit_config=FitConfig(
            min_steps=min(FitConfig.min_steps, args.fit_max_steps), max_steps=args.fit_max_steps
        ),
    )
    s = result.summary
    print(
        f"  {s['n_vetted']} vetted, {s['n_errors']} failed, {s['n_flagged']} flagged "
        f"(expected planets among them: {s['expected_planets_flagged']:.1f}) "
        f"in {s['runtime_seconds']:.0f}s"
    )
    if "truth" in s:
        t = s["truth"]
        precision = "n/a" if t["precision"] is None else f"{t['precision']:.2f}"
        recall = "n/a" if t["recall"] is None else f"{t['recall']:.2f}"
        print(
            f"  against ground truth: {t['flagged_planets']} of {s['n_flagged']} flagged are "
            f"planets (precision {precision}, recall {recall})"
        )
    if "fits" in s:
        f = s["fits"]
        print(
            f"  fitted {f['fitted']} candidates ({f['converged']} converged, {f['errors']} failed)"
            + (
                f"; density inconsistent with the star: {', '.join(f['density_inconsistent'])}"
                if f["density_inconsistent"] else ""
            )
        )
    for name in ("candidates.csv", "summary.json", "dashboard.html"):
        print(f"  wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    # Run through the importable module, not ``__main__``, so the worker
    # processes can unpickle the per-star task by name.
    from transitml.batch import main as _main

    sys.exit(_main())
