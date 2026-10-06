"""Do the fitted intervals mean what they say?  Injection-recovery coverage.

    python -m transitml.fit_coverage            # writes results/fit_coverage.json,
                                                # figures/10_fit_coverage.png and
                                                # figures/09_fit_example.png

A 68% credible interval is only useful if the truth lands inside it about
68% of the time.  This injects planets with known parameters and counts.
Every injection is run twice, with the same planet on the same time grid:

``white``     the transit in white noise of the star's level, fitted
              directly: a test of the fitter alone.
``pipeline``  the transit multiplied into a synthetic variable star from the
              training generator (rotation, red noise, ramps, flares), then
              detrended, searched and fitted exactly as ``vet --fit`` does:
              a test of the whole chain, detrending included.

A planet counts only if the search finds its period (within 1%); the others
are reported but not scored, since a fit of the wrong signal has no truth to
cover.  For each parameter the output gives the fraction of truths inside
the 68% and 95% intervals and the posterior quantile of every truth, which
is uniform when the intervals are calibrated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time as clock
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from .config import default_config
from .data.base import LightCurve
from .features import run_bls
from .fit import (
    FitConfig,
    FitError,
    TransitModel,
    _derived,
    a_over_rs,
    durations,
    fit_transit,
    mid_transit_depth,
    u_to_q,
)
from .preprocess import FlattenedLightCurve, flatten

#: Parameters scored, in display order.
SCORED: tuple[str, ...] = ("period", "t0", "k", "b", "rho_star", "t14_hours", "depth_ppm")

#: Limb darkening of every injected star (a Sun-like star in the TESS band).
INJECTED_U: tuple[float, float] = (0.40, 0.25)


def draw_planet(rng: np.random.Generator, time: np.ndarray, rho_star: float) -> dict[str, float]:
    """One planet: log-uniform period and radius ratio, uniform impact parameter."""
    period = float(10 ** rng.uniform(0.0, 1.0))
    k = float(10 ** rng.uniform(math.log10(0.04), math.log10(0.12)))
    b = float(rng.uniform(0.0, 0.9))
    a = float(a_over_rs(period, rho_star))
    t14 = float(durations(period, a, k, b)[0])
    return {
        "period": period,
        "t0": float(time[0] + rng.uniform(0.0, period)),
        "k": k,
        "b": b,
        "rho_star": float(rho_star),
        "a_over_rs": a,
        "t14": t14,
    }


def transit_flux(time: np.ndarray, planet: dict[str, float], exposure_days: float) -> np.ndarray:
    q1, q2 = u_to_q(*INJECTED_U)
    theta = np.array([
        planet["t0"], planet["period"], planet["k"], planet["b"],
        math.log10(planet["t14"]), q1, q2,
    ])
    return TransitModel(time, exposure_days, 3.0 / 1440.0).flux(theta)


def truth_values(planet: dict[str, float], t0_reference: float) -> dict[str, float]:
    """The scored quantities' true values, with t0 moved to the fit's reference transit."""
    q1, q2 = u_to_q(*INJECTED_U)
    n = round((t0_reference - planet["t0"]) / planet["period"])
    theta = np.array([[
        planet["t0"], planet["period"], planet["k"], planet["b"],
        math.log10(planet["t14"]), q1, q2,
    ]])
    return {
        "period": planet["period"],
        "t0": planet["t0"] + n * planet["period"],
        "k": planet["k"],
        "b": planet["b"],
        "rho_star": planet["rho_star"],
        "t14_hours": 24.0 * planet["t14"],
        "depth_ppm": float(1e6 * mid_transit_depth(theta)[0]),
    }


def _score(flat: FlattenedLightCurve, planet: dict[str, float], fit_config: FitConfig, bls_config):
    """Search, fit and score one curve; ``None`` if the search missed the planet."""
    found = run_bls(flat, bls_config)
    if abs(found["period"] / planet["period"] - 1.0) > 0.01:
        return {"recovered": False, "bls_period": float(found["period"])}
    try:
        fit = fit_transit(
            flat, found["period"], found["transit_time"], found["duration"], found["depth"],
            fit_config,
        )
    except FitError as exc:
        return {"recovered": False, "bls_period": float(found["period"]), "error": str(exc)}
    draws = _derived(fit.samples)
    draws["depth_ppm"] = 1e6 * mid_transit_depth(fit.samples)
    truth = truth_values(planet, fit.value("t0"))
    quantile = {name: float(np.mean(draws[name] < truth[name])) for name in SCORED}
    inside = {
        name: {
            "68": fit.parameters[name]["lower"] <= truth[name] <= fit.parameters[name]["upper"],
            "95": fit.parameters[name]["lower95"] <= truth[name] <= fit.parameters[name]["upper95"],
        }
        for name in SCORED
    }
    return {
        "recovered": True,
        "truth": truth,
        "median": {name: fit.value(name) for name in SCORED},
        "width68": {
            name: fit.parameters[name]["upper"] - fit.parameters[name]["lower"] for name in SCORED
        },
        "quantile": quantile,
        "inside": inside,
        "converged": fit.converged,
        "autocorr_time": fit.sampler["autocorr_time_max"],
        "n_steps": fit.sampler["n_steps"],
        "beta": fit.noise["beta"],
        "sigma_ppm": fit.noise["sigma_ppm"],
    }


def _host_and_planet(index: int, seed: int):
    """Injection ``index``: a synthetic variable star, its planet and the transit profile."""
    from .data.synthetic import SyntheticTESSSource

    config = default_config()
    source = SyntheticTESSSource(
        index + 1, 0.0, 0.0, seed=seed, survey=config.survey, noise=config.noise,
        star=config.star, planet=config.planet, eb=config.eb,
    )
    star = source.generate(index)  # a variable star with no companion
    rng = np.random.default_rng([seed, index, 1])
    planet = draw_planet(rng, star.time, star.meta["rho_star_cgs"])
    dip = transit_flux(star.time, planet, config.survey.cadence_days)
    return config, star, planet, dip, rng


def run_injection(index: int, seed: int, fit_config: FitConfig) -> dict[str, Any]:
    """Both experiments for injection ``index``."""
    config, star, planet, dip, rng = _host_and_planet(index, seed)
    sigma = float(star.meta["sigma_white"])
    fit_config = replace(fit_config, seed=index)

    white_flux = dip + rng.normal(0.0, sigma, star.time.size)
    white = FlattenedLightCurve(
        star.target_id, star.time, white_flux, np.full(star.time.size, sigma),
        np.ones(star.time.size), sigma, 0,
    )
    injected = LightCurve(star.target_id, star.time, star.flux * dip, star.flux_err, meta=star.meta)
    out = {"index": index, "planet": planet, "sigma_ppm": 1e6 * sigma}
    out["white"] = _score(white, planet, fit_config, config.bls)
    try:
        out["pipeline"] = _score(flatten(injected, config.preprocess), planet, fit_config, config.bls)
    except ValueError as exc:
        out["pipeline"] = {"recovered": False, "error": str(exc)}
    return out


def example_fit(index: int, seed: int, fit_config: FitConfig):
    """The whole-chain fit of injection ``index``, with the host's density and radius."""
    config, star, planet, dip, _ = _host_and_planet(index, seed)
    injected = LightCurve(star.target_id, star.time, star.flux * dip, star.flux_err, meta=star.meta)
    flat = flatten(injected, config.preprocess)
    found = run_bls(flat, config.bls)
    rho = star.meta["rho_star_cgs"]
    return fit_transit(
        flat, found["period"], found["transit_time"], found["duration"], found["depth"],
        replace(fit_config, seed=index),
        stellar_density=(rho, 0.1 * rho), stellar_radius=star.meta["r_star_rsun"],
    ), planet


def summarise(rows: list[dict[str, Any]], experiment: str) -> dict[str, Any]:
    scored = [r[experiment] for r in rows if r[experiment].get("recovered")]
    out: dict[str, Any] = {
        "n_injected": len(rows),
        "n_recovered": len(scored),
        "n_converged": sum(bool(s["converged"]) for s in scored),
        "median_beta": float(np.median([s["beta"] for s in scored])) if scored else None,
        "parameters": {},
    }
    for name in SCORED:
        if not scored:
            break
        in68 = np.array([s["inside"][name]["68"] for s in scored])
        in95 = np.array([s["inside"][name]["95"] for s in scored])
        q = np.array([s["quantile"][name] for s in scored])
        out["parameters"][name] = {
            "in68": float(in68.mean()),
            "in95": float(in95.mean()),
            "median_quantile": float(np.median(q)),
        }
    return out


def plot_coverage(summary: dict[str, Any], path: Path) -> Path:
    from .plots import GRID, INK_SOFT, NEUTRAL, SERIES, _save, _style, plt

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
    names = list(SCORED)
    x = np.arange(len(names))
    pretty = {"k": "Rp/R*", "rho_star": "density", "t14_hours": "T14", "depth_ppm": "depth"}
    labels = [pretty.get(n, n) for n in names]
    n_scored = min(max(summary[e]["n_recovered"], 1) for e in ("white", "pipeline"))
    for ax, level in zip(axes, ("68", "95")):
        target = int(level) / 100.0
        spread = math.sqrt(target * (1 - target) / n_scored)
        ax.axhspan(target - spread, target + spread, color=NEUTRAL, alpha=0.18, lw=0,
                   label=f"nominal {level}%, with the 1-sigma spread of {n_scored} planets")
        ax.axhline(target, color=INK_SOFT, lw=1.0, ls="--")
        for j, (experiment, colour, label) in enumerate((
            ("white", SERIES[0], "fitter alone (white noise)"),
            ("pipeline", SERIES[1], "whole chain (variable star, detrended)"),
        )):
            s = summary[experiment]
            values = [s["parameters"].get(name, {}).get(f"in{level}", np.nan) for name in names]
            ax.plot(x + (j - 0.5) * 0.24, values, "o", color=colour, ms=7,
                    label=f"{label}, {s['n_recovered']} planets")
        ax.set_xticks(x, labels)
        ax.set_title(f"Truths inside the {level}% interval", loc="left")
        ax.set_ylim(0.0, 1.04)
        ax.grid(axis="x", visible=False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
    axes[0].set_ylabel("fraction of injected planets")
    axes[0].legend(loc="lower left")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.fit_coverage",
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=60, help="Planets injected.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--baseline", choices=("offset", "line", "quadratic"), default=FitConfig.baseline,
        help="The per-transit baseline the fits marginalise over.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/fit_coverage.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/10_fit_coverage.png"))
    parser.add_argument(
        "--example-figure", type=Path, default=Path("figures/09_fit_example.png"),
        help="The fit of the first injection the search found, as vet --fit draws it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = clock.time()
    fit_config = FitConfig(baseline=args.baseline)
    rows = Parallel(n_jobs=args.n_jobs)(
        delayed(run_injection)(i, args.seed, fit_config) for i in range(args.n)
    )
    summary = {
        "n": args.n,
        "seed": args.seed,
        "injected_limb_darkening_u": list(INJECTED_U),
        "fit_config": {k: v for k, v in vars(fit_config).items()},
        "white": summarise(rows, "white"),
        "pipeline": summarise(rows, "pipeline"),
        "runtime_seconds": round(clock.time() - start, 1),
        "injections": rows,
    }
    first = next((r["index"] for r in rows if r["pipeline"].get("recovered")), None)
    if first is not None:
        from .plots import plot_fit

        fit, planet = example_fit(first, args.seed, fit_config)
        plot_fit(fit, args.example_figure)
        summary["example"] = {"index": first, "planet": planet, "fit": fit.to_dict()}
    summary["runtime_seconds"] = round(clock.time() - start, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    plot_coverage(summary, args.figure)
    for experiment in ("white", "pipeline"):
        s = summary[experiment]
        print(f"{experiment}: {s['n_recovered']} of {s['n_injected']} found by the search, "
              f"{s['n_converged']} converged, median beta {s['median_beta']}")
        for name, v in s["parameters"].items():
            print(f"  {name:10s} in68 {v['in68']:.2f}  in95 {v['in95']:.2f}  "
                  f"median quantile {v['median_quantile']:.2f}")
    print(f"wrote {args.out} and {args.figure} in {summary['runtime_seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    from transitml.fit_coverage import main as _main

    sys.exit(_main())
