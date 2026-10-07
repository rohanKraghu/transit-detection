"""BLS against TLS on the same planets: which search finds more periods, and at what cost?

    python -m transitml.search_benchmark            # 600 planets, ~2 minutes on 4 cores

Every light curve holds one planet from the synthetic population (no
binaries, no planet-free stars), is detrended once, and is searched by both
BLS and Transit Least Squares between the same period limits.  A period
counts as found when it is the injected one or a low-order alias, exactly as
in the pipeline's report.  Pairs make the comparison: the same planet, the
same noise, two searches.

Writes ``results/search_comparison/report.txt`` and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from .config import BLSConfig, PreprocessConfig, default_config
from .data.synthetic import SyntheticTESSSource
from .evaluate import period_recovered
from .features import run_bls
from .preprocess import flatten

SNR_BINS: tuple[float, ...] = (0.0, 7.0, 10.0, 15.0, 25.0, 50.0, np.inf)


def _one(
    index: int, source: SyntheticTESSSource, preprocess: PreprocessConfig, bls: BLSConfig
) -> dict[str, Any]:
    lc = source.generate(index)
    flat = flatten(lc, preprocess)
    row: dict[str, Any] = {
        "index": index,
        "period": float(lc.meta["period"]),
        "depth": float(lc.meta["depth"]),
        "true_snr": float(lc.meta["true_snr"]),
    }
    for search in ("bls", "tls"):
        start = time.perf_counter()
        result = run_bls(flat, replace(bls, search=search))
        row[f"{search}_seconds"] = time.perf_counter() - start
        row[f"{search}_period"] = float(result["period"])
        row[f"{search}_fell_back"] = result["searched_with"] != search
    return row


def run(n_curves: int, seed: int, n_jobs: int) -> dict[str, Any]:
    """Search every planet with both methods and summarise."""
    base = default_config()
    source = SyntheticTESSSource(
        n_curves, 1.0, 0.0, seed=seed, survey=base.survey, noise=base.noise,
        star=base.star, planet=base.planet, eb=base.eb,
    )
    rows = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(_one)(i, source, base.preprocess, base.bls) for i in range(n_curves)
    )
    period = np.array([r["period"] for r in rows])
    snr = np.array([r["true_snr"] for r in rows])
    found = {
        s: period_recovered(np.array([r[f"{s}_period"] for r in rows]), period)
        for s in ("bls", "tls")
    }
    bins = []
    for lo, hi in pairwise(SNR_BINS):
        sel = (snr >= lo) & (snr < hi)
        bins.append({
            "snr_low": lo,
            "snr_high": hi if np.isfinite(hi) else None,
            "n": int(sel.sum()),
            "bls": int(found["bls"][sel].sum()),
            "tls": int(found["tls"][sel].sum()),
            "only_bls": int((found["bls"] & ~found["tls"])[sel].sum()),
            "only_tls": int((found["tls"] & ~found["bls"])[sel].sum()),
        })
    return {
        "n_curves": n_curves,
        "seed": seed,
        "bls_found": int(found["bls"].sum()),
        "tls_found": int(found["tls"].sum()),
        "only_bls": int((found["bls"] & ~found["tls"]).sum()),
        "only_tls": int((found["tls"] & ~found["bls"]).sum()),
        "by_snr": bins,
        "bls_ms_per_curve": 1e3 * float(np.mean([r["bls_seconds"] for r in rows])),
        "tls_ms_per_curve": 1e3 * float(np.mean([r["tls_seconds"] for r in rows])),
        "tls_fell_back_to_bls": int(sum(r["tls_fell_back"] for r in rows)),
    }


def format_report(summary: dict[str, Any]) -> str:
    n = summary["n_curves"]
    lines = [
        "Periodic search: BLS against TLS on the same planets",
        "=" * 72,
        f"{n} synthetic planets, seed {summary['seed']}; one detrending, two searches",
        "",
        f"Periods found: BLS {summary['bls_found']} ({summary['bls_found'] / n:.1%}), "
        f"TLS {summary['tls_found']} ({summary['tls_found'] / n:.1%})",
        f"  found by only one: BLS {summary['only_bls']}, TLS {summary['only_tls']}",
        f"  TLS found nothing it could fit and fell back to BLS on "
        f"{summary['tls_fell_back_to_bls']}",
        "",
        "  injected SNR       n     BLS     TLS   only BLS   only TLS",
    ]
    for row in summary["by_snr"]:
        hi = "inf" if row["snr_high"] is None else f"{row['snr_high']:g}"
        lines.append(
            f"  {row['snr_low']:>5g} - {hi:<5}  {row['n']:>4}   {row['bls']:>5}   {row['tls']:>5}"
            f"   {row['only_bls']:>8}   {row['only_tls']:>8}"
        )
    lines += [
        "",
        f"Time per light curve, one core: BLS {summary['bls_ms_per_curve']:.0f} ms, "
        f"TLS {summary['tls_ms_per_curve']:.0f} ms",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.search_benchmark",
        description="Search the same synthetic planets with BLS and with TLS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-curves", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results") / "search_comparison"
    )
    args = parser.parse_args(argv)
    summary = run(args.n_curves, args.seed, args.n_jobs)
    report = format_report(summary)
    print(report)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "report.txt").write_text(report + "\n")
    (args.results_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
