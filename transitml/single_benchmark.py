"""How often does the single-event search find a lone transit, and how often does it invent one?

    python -m transitml.single_benchmark            # 1200 curves, ~5 minutes on 4 cores

Half of the variable stars from the synthetic generator get one long-period
planet, drawn from the same physics as the main population but with periods
from 14 to 400 days and a transit somewhere in the sector, so the sector holds
two transits or one (or none, when it falls in the gap).  The other half are
left alone.  Every curve is detrended exactly as in the pipeline and
searched with :func:`transitml.single.search_single_events`.

* **Recovery** is the fraction of planets with at least one transit in the
  data whose transit is found (an event within half a transit duration of a
  true mid-transit time).  It is binned by the SNR the strongest transit has
  against that star's own noise: its mean injected depth over the error of a
  box of its true duration at its true time, with the detrended light
  curve's scatter and its red-noise factor at that duration.  That is the
  best the box search can do on that star.  The ideal white-noise SNR
  (depth over the white-noise level times the square root of the in-transit
  cadences) is recorded too, and is typically twice as high, because the
  synthetic stars also carry red noise.
* **False alarms** are events on the untouched stars, which have no transit.
  This is the number that sets ``min_snr`` and ``ramp_delta_chi2``.
* **Duos** are planets with two transits in the data: how many have both
  found, how many of those pair into a duo, and whether the true period is
  among the periods the search leaves allowed.

Writes ``results/single_transit/report.txt`` and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from .config import PlanetConfig, PreprocessConfig, SingleEventConfig, default_config
from .data.base import LightCurve
from .data.synthetic import SyntheticTESSSource, planet_signal, trapezoid_transit
from .preprocess import FlattenedLightCurve, flatten
from .single import _BoxStats, binned_noise_factor, search_single_events

SNR_BINS: tuple[float, ...] = (0.0, 7.0, 10.0, 15.0, 25.0, np.inf)


def inject_long_period(
    lc: LightCurve, rng: np.random.Generator, planet: PlanetConfig
) -> tuple[LightCurve, dict[str, Any]]:
    """Subtract one long-period planet's transits from ``lc``; return its truth.

    The planet is drawn as in the main population, then its epoch is moved to
    a uniform time inside the window, so the benchmark measures planets that
    do transit during the sector rather than mostly ones that do not.  A
    transit can still land in the gap.
    """
    _, truth = planet_signal(lc.time, rng, float(lc.meta["rho_star_cgs"]), planet)
    truth["epoch"] = float(rng.uniform(lc.time[0], lc.time[-1]))
    dip = trapezoid_transit(
        lc.time,
        truth["period"],
        truth["epoch"],
        truth["depth"],
        truth["duration_t14"],
        truth["duration_t23"],
    )
    k = np.arange(
        np.floor((lc.time[0] - truth["epoch"]) / truth["period"]),
        np.ceil((lc.time[-1] - truth["epoch"]) / truth["period"]) + 1,
    )
    mids = truth["epoch"] + k * truth["period"]
    half = 0.5 * truth["duration_t14"]
    seen = [
        float(m) for m in mids
        if np.count_nonzero(np.abs(lc.time - m) < half) >= 2
    ]
    sigma = float(lc.meta["sigma_white"])
    n_in = [int(np.count_nonzero(np.abs(lc.time - m) < half)) for m in seen]
    mean_depth = [float(np.mean(dip[np.abs(lc.time - m) < half])) for m in seen]
    truth = {
        **truth,
        "transit_times": seen,
        "mean_depths": mean_depth,
        # The ideal SNR of one transit, against white noise alone.
        "single_snr": float(truth["depth"] / sigma * np.sqrt(max(n_in))) if n_in else 0.0,
    }
    injected = LightCurve(
        lc.target_id, lc.time, lc.flux - dip, lc.flux_err, lc.label, dict(lc.meta)
    )
    return injected, truth


def noise_snr(flat: FlattenedLightCurve, truth: dict[str, Any], min_coverage: float) -> float:
    """SNR of the strongest injected transit against this star's own noise.

    Its mean injected depth over the error the box search assigns a box of
    the true duration at the true time: the detrended scatter, times the
    red-noise factor at that duration, times ``sqrt(1/n_in + 1/n_out)``.
    0 when no transit has enough data around it.
    """
    duration = float(truth["duration_t14"])
    beta = binned_noise_factor(flat, duration)
    stats = _BoxStats(flat)
    best = 0.0
    for mid, depth in zip(truth["transit_times"], truth["mean_depths"], strict=True):
        measured, err, _ = stats.depth(np.array([mid]), duration, beta, min_coverage)
        if np.isfinite(measured[0]) and err[0] > 0:
            best = max(best, depth / float(err[0]))
    return best


def _one(
    index: int,
    source: SyntheticTESSSource,
    inject: bool,
    planet: PlanetConfig,
    preprocess: PreprocessConfig,
    config: SingleEventConfig,
    seed: int,
) -> dict[str, Any]:
    lc = source.generate(index)
    truth: dict[str, Any] = {}
    if inject:
        lc, truth = inject_long_period(lc, np.random.default_rng([seed, index, 7]), planet)
    try:
        flat = flatten(lc, preprocess)
        found = search_single_events(flat, config)
    except ValueError:
        return {"index": index, "injected": inject, "error": True}
    times = [e.time for e in found.events]
    row: dict[str, Any] = {
        "index": index,
        "injected": inject,
        "error": False,
        "n_events": len(times),
        "n_ramps": len(found.ramps),
        "best_snr": max((e.snr for e in found.events), default=0.0),
        "n_transits_seen": len(truth.get("transit_times", [])),
    }
    if not inject:
        return row
    tol = 0.5 * truth["duration_t14"]
    found_transits = [
        any(abs(t - m) < tol for t in times) for m in truth["transit_times"]
    ]
    duo = None
    if len(truth["transit_times"]) == 2:
        first, second = truth["transit_times"]
        pair = [
            d for d in found.duos
            if abs(d.first.time - first) < tol and abs(d.second.time - second) < tol
        ]
        duo = {
            "both_found": all(found_transits),
            "paired": bool(pair),
            "true_period_allowed": any(
                abs(p / truth["period"] - 1.0) < 0.02 for d in pair for p in d.allowed_periods
            ),
        }
    return {
        **row,
        "recovered": any(found_transits),
        "duo": duo,
        "noise_snr": noise_snr(flat, truth, config.min_coverage) if truth["transit_times"] else 0.0,
        **{k: truth[k] for k in ("period", "depth", "duration_t14", "single_snr")},
    }


def run(n_curves: int, seed: int, n_jobs: int, config: SingleEventConfig) -> dict[str, Any]:
    """Run the benchmark and return its summary."""
    base = default_config()
    source = SyntheticTESSSource(
        n_curves, 0.0, 0.0, seed=seed, survey=base.survey, noise=base.noise, star=base.star
    )
    planet = replace(base.planet, period_range_days=(14.0, 400.0))
    rows = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(_one)(i, source, i % 2 == 0, planet, base.preprocess, config, seed)
        for i in range(n_curves)
    )
    clean = [r for r in rows if not r["injected"] and not r["error"]]
    injected = [r for r in rows if r["injected"] and not r["error"]]
    visible = [r for r in injected if r["n_transits_seen"] > 0]
    longest = max(config.durations_days)
    bins = []
    for lo, hi in pairwise(SNR_BINS):
        sel = [r for r in visible if lo <= r["noise_snr"] < hi]
        short = [r for r in sel if r["duration_t14"] <= longest]
        bins.append({
            "snr_low": lo,
            "snr_high": hi if np.isfinite(hi) else None,
            "n": len(sel),
            "recovered": sum(r["recovered"] for r in sel),
            "n_within_longest_box": len(short),
            "recovered_within_longest_box": sum(r["recovered"] for r in short),
        })
    duos = [r["duo"] for r in visible if r["duo"] is not None]
    return {
        "n_curves": n_curves,
        "seed": seed,
        "min_snr": config.min_snr,
        "ramp_delta_chi2": config.ramp_delta_chi2,
        "longest_box_days": longest,
        "false_alarm_stars": sum(r["n_events"] > 0 for r in clean),
        "clean_stars": len(clean),
        "clean_stars_with_a_ramp_set_aside": sum(r["n_ramps"] > 0 for r in clean),
        "injected": len(injected),
        "with_a_transit_in_the_data": len(visible),
        "by_transits_seen": {
            str(n): sum(r["n_transits_seen"] == n for r in injected) for n in (0, 1, 2, 3)
        },
        "recovery_by_snr": bins,
        "median_white_over_noise_snr": float(np.median(
            [r["single_snr"] / r["noise_snr"] for r in visible if r["noise_snr"] > 0]
        )) if visible else float("nan"),
        "duos": len(duos),
        "duo_both_found": sum(d["both_found"] for d in duos),
        "duo_paired": sum(d["paired"] for d in duos),
        "duo_true_period_allowed": sum(d["true_period_allowed"] for d in duos),
        "errors": sum(r["error"] for r in rows),
        "clean_best_snrs": sorted((r["best_snr"] for r in clean), reverse=True)[:10],
    }


def _fraction(k: int, n: int) -> str:
    return f"{k / n:.0%}" if n else "n/a"


def format_report(summary: dict[str, Any]) -> str:
    clean = max(summary["clean_stars"], 1)
    seen = ", ".join(f"{k}: {v}" for k, v in summary["by_transits_seen"].items())
    best = ", ".join(f"{s:.1f}" for s in summary["clean_best_snrs"])
    longest = summary["longest_box_days"] * 24
    lines = [
        "Single-transit injection-recovery",
        "=" * 72,
        (
            f"{summary['n_curves']} synthetic variable stars, seed {summary['seed']}; "
            f"events need SNR >= {summary['min_snr']:g}"
        ),
        "",
        (
            f"False alarms: {summary['false_alarm_stars']} of {summary['clean_stars']} "
            "stars with nothing injected show an event "
            f"({summary['false_alarm_stars'] / clean:.1%})"
        ),
        f"  highest SNRs on clean stars: {best}",
        (
            f"  stars where a ramp-shaped dip was set aside: "
            f"{summary['clean_stars_with_a_ramp_set_aside']}"
        ),
        "",
        f"Injected: {summary['injected']} planets with P = 14-400 d; transits in the data: {seen}",
        "",
        "  SNR against the star's own noise (white-noise SNR is "
        f"{summary['median_white_over_noise_snr']:.1f}x higher, median)",
        f"                         all planets        transits up to {longest:.1f} h",
        "  SNR               n   recovered          n   recovered",
    ]
    for row in summary["recovery_by_snr"]:
        hi = "inf" if row["snr_high"] is None else f"{row['snr_high']:g}"
        lines.append(
            f"  {row['snr_low']:>5g} - {hi:<5}  {row['n']:>4}   {row['recovered']:>4}"
            f" ({_fraction(row['recovered'], row['n']):>4})"
            f"     {row['n_within_longest_box']:>4}   {row['recovered_within_longest_box']:>4}"
            f" ({_fraction(row['recovered_within_longest_box'], row['n_within_longest_box']):>4})"
        )
    lines += [
        "",
        (
            f"Duos (two transits in the data): {summary['duos']}; both found "
            f"{summary['duo_both_found']}, paired {summary['duo_paired']}, true period "
            f"among the allowed aliases {summary['duo_true_period_allowed']}"
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.single_benchmark",
        description="Inject long-period planets and measure single-event recovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-curves", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results") / "single_transit"
    )
    args = parser.parse_args(argv)
    summary = run(args.n_curves, args.seed, args.n_jobs, SingleEventConfig())
    report = format_report(summary)
    print(report)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "report.txt").write_text(report + "\n")
    (args.results_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
