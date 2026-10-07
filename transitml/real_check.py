"""Checks against real TESS data: known planets' fits, and a real sector's ranking.

Two questions the synthetic and injected studies cannot answer:

``python -m transitml.real_check planets``
    Fit known TESS planets with ``vet --fit`` and compare the radius ratio,
    impact parameter, duration, period and transit-implied stellar density
    with the published values (``data/real_planets/published.csv``, from the
    NASA Exoplanet Archive's composite table).  Each comparison is a
    difference in units of the combined uncertainty, the fit's interval on the
    side facing the published value and the published error added in
    quadrature.

``python -m transitml.real_check sector``
    Rank a real sector run by ``python -m transitml.batch`` against the TOI
    catalogue: how the hosts of confirmed planets, of open candidates and of
    known false positives place among stars that are not TOIs, and how often
    the search found the TOI's period.

The composite table takes each parameter from whichever paper the archive
picked, so a planet's numbers can come from different solutions; a
difference of two or three units is a reason to look, not proof of a bug.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .data.toi import read_toi_table, star_label
from .evaluate import fast_average_precision, period_recovered

#: Fitted parameter, published column, published uncertainty column.
QUANTITIES: tuple[tuple[str, str, str | None], ...] = (
    ("period", "pl_orbper", None),
    ("k", "pl_ratror", "pl_ratrorerr1"),
    ("b", "pl_imppar", "pl_impparerr1"),
    ("t14_hours", "pl_trandur", "pl_trandurerr1"),
    ("rho_star", "st_dens", "st_denserr1"),
)


def _number(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def read_published(path: str | Path) -> list[dict[str, Any]]:
    """The planets to fit and their published values (``#`` lines are comments)."""
    with open(path, newline="") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        return [dict(row) for row in rows]


def difference(fitted: Mapping[str, float], published: float, error: float) -> float:
    """Published minus fitted median, in units of the combined uncertainty.

    ``fitted`` is a posterior summary with ``median``, ``lower`` and
    ``upper`` (the 68% interval); the half of it facing the published value
    is used.  NaN when there is no published value.
    """
    median = fitted["median"]
    if not (math.isfinite(published) and math.isfinite(median)):
        return math.nan
    side = fitted["upper"] - median if published > median else median - fitted["lower"]
    error = error if math.isfinite(error) else 0.0
    scale = math.hypot(side, error)
    return (published - median) / scale if scale > 0 else math.nan


def compare_planet(published: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """One planet's row: its vetting score, then fitted, published and difference."""
    row: dict[str, Any] = {
        "planet": published["pl_name"],
        "tic": published["tic_id"],
        "sector": int(published["sector"]),
        "score": report.get("score"),
        "flagged": report.get("above_threshold"),
    }
    candidates = report.get("candidates") or []
    row["search_period"] = candidates[0]["period"] if candidates else math.nan
    fit = report.get("fit") or {}
    parameters = fit.get("parameters")
    if not parameters:
        row["fit_status"] = fit.get("error", "no fit")
        return row
    row["fit_status"] = "ok"
    row["converged"] = bool(fit.get("sampler", {}).get("converged"))
    for name, column, error_column in QUANTITIES:
        summary = parameters[name]
        value = _number(published.get(column))
        error = _number(published.get(error_column)) if error_column else 0.0
        row[f"{name}_fit"] = summary["median"]
        row[f"{name}_lower"] = summary["lower"]
        row[f"{name}_upper"] = summary["upper"]
        row[f"{name}_published"] = value
        row[f"{name}_diff"] = difference(summary, value, error)
    check = fit.get("density_check")
    if check:
        row["tic_density"] = check["stellar_density"]
        row["density_consistent"] = check["consistent"]
    row["warnings"] = "; ".join(fit.get("warnings", []))
    return row


def run_planets(
    published_path: str | Path,
    out_dir: str | Path,
    *,
    model: str | Path = "results/model.joblib",
    fit_max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Vet and fit every planet in ``published_path``; write the comparison."""
    from . import vet

    out_dir = Path(out_dir)
    rows = []
    for planet in read_published(published_path):
        stem = planet["pl_name"].replace(" ", "_")
        planet_dir = out_dir / "reports" / stem
        argv = [
            planet["tic_id"], "--sector", planet["sector"], "--author", planet["author"],
            "--exposure-time", planet["exptime"], "--fit", "--model", str(model),
            "--out-dir", str(planet_dir),
        ]
        if fit_max_steps is not None:
            argv += ["--fit-max-steps", str(fit_max_steps)]
        print(f"{planet['pl_name']} ({planet['tic_id']}, sector {planet['sector']})")
        vet.main(argv)
        report_path = planet_dir / f"vet_{planet['tic_id'].replace(' ', '_')}.json"
        rows.append(compare_planet(planet, json.loads(report_path.read_text())))
    write_planets(rows, out_dir)
    return rows


def write_planets(rows: Sequence[Mapping[str, Any]], out_dir: str | Path) -> tuple[Path, Path]:
    """``comparison.csv`` (every number) and ``comparison.txt`` (the table to read)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})
    txt_path = out_dir / "comparison.txt"
    txt_path.write_text(planets_table(rows))
    return csv_path, txt_path


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6g}"
    return value


def planets_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Fitted against published, with the difference in combined uncertainties."""
    lines = [
        "Fitted (median) / published (difference in combined 1-sigma units)",
        f"{'planet':<12} {'score':>5}  {'period d':>24}  {'Rp/R*':>22}  {'b':>17}  "
        f"{'T14 h':>19}  {'rho_star g/cm3':>22}",
    ]
    for row in rows:
        if row.get("fit_status") != "ok":
            lines.append(f"{row['planet']:<12} {row.get('score', math.nan):5.3f}  {row['fit_status']}")
            continue
        cells = []
        for name, width, digits in (
            ("period", 24, 5), ("k", 22, 4), ("b", 17, 2), ("t14_hours", 19, 3), ("rho_star", 22, 3)
        ):
            fit, pub, diff = row[f"{name}_fit"], row[f"{name}_published"], row[f"{name}_diff"]
            pub_text = f"{pub:.{digits}f}" if math.isfinite(pub) else "-"
            diff_text = f"{diff:+.1f}" if math.isfinite(diff) else "-"
            cells.append(f"{f'{fit:.{digits}f} / {pub_text} ({diff_text})':>{width}}")
        lines.append(f"{row['planet']:<12} {row['score']:5.3f}  " + "  ".join(cells))
    diffs = np.array([
        row[f"{name}_diff"] for row in rows if row.get("fit_status") == "ok"
        for name, _, _ in QUANTITIES if name != "period"
    ], dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size:
        lines.append(
            f"\nRp/R*, b, T14 and density: {np.mean(np.abs(diffs) < 1):.0%} of {diffs.size} "
            f"comparisons within 1 unit, {np.mean(np.abs(diffs) < 2):.0%} within 2, "
            f"{np.mean(np.abs(diffs) < 3):.0%} within 3."
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# A real sector against the TOI catalogue
# --------------------------------------------------------------------------
#: How a star is grouped: the hosts of confirmed planets (any CP or KP TOI),
#: of known false positives only (every TOI FP or FA), of open TOIs, and stars
#: that are not TOIs at all.
GROUPS: tuple[str, ...] = ("planet", "false positive", "open TOI", "not a TOI")


def sector_ranking(
    candidates_path: str | Path, toi_path: str | Path, sector: int
) -> dict[str, Any]:
    """Where the TOI hosts of ``sector`` rank in a batch's ``candidates.csv``."""
    with open(candidates_path, newline="") as handle:
        stars = list(csv.DictReader(handle))
    by_tic: dict[int, list] = {}
    for toi in read_toi_table(toi_path):
        if sector in toi.sectors:
            by_tic.setdefault(toi.tic, []).append(toi)

    groups, recovered = [], []
    for star in stars:
        tic = int(star["target_id"].split()[-1])
        tois = by_tic.get(tic)
        if tois is None:
            groups.append("not a TOI")
            recovered.append(False)
            continue
        label = star_label(tois)
        groups.append({1: "planet", 0: "false positive"}.get(label, "open TOI"))
        found = _number(star["period_days"])
        recovered.append(any(
            bool(period_recovered(np.array([found]), np.array([t.period]))[0]) for t in tois
        ))
    scores = np.array([_number(s["score"]) for s in stars])
    flagged = np.array([s["flagged"] in ("True", "true", "1") for s in stars])
    groups_arr = np.array(groups)
    recovered_arr = np.array(recovered)
    order = np.argsort(-scores, kind="stable")

    out: dict[str, Any] = {
        "candidates": str(candidates_path),
        "toi_table": str(toi_path),
        "sector": sector,
        "n_stars": len(stars),
        "n_toi_hosts_in_sector": len(by_tic),
        "groups": {},
        "average_precision": {},
        "top": {},
    }
    for group in GROUPS:
        mask = groups_arr == group
        if not mask.any():
            continue
        ranks = np.empty(len(stars), dtype=int)
        ranks[order] = np.arange(1, len(stars) + 1)
        entry = {
            "n": int(mask.sum()),
            "flagged": float(flagged[mask].mean()),
            "median_rank": float(np.median(ranks[mask])),
        }
        if group != "not a TOI":
            found = recovered_arr[mask]
            entry["period_found"] = float(found.mean())
            entry["flagged_when_period_found"] = (
                float(flagged[mask][found].mean()) if found.any() else None
            )
        out["groups"][group] = entry
    background = groups_arr == "not a TOI"
    for group in ("planet", "open TOI", "false positive"):
        mask = groups_arr == group
        if mask.any() and background.any():
            keep = mask | background
            y = mask[keep].astype(int)
            out["average_precision"][f"{group} vs not a TOI"] = {
                "ap": fast_average_precision(y, scores[keep]),
                "chance": float(y.mean()),
            }
    tois_only = np.isin(groups_arr, ("planet", "false positive"))
    if tois_only.any():
        y = (groups_arr[tois_only] == "planet").astype(int)
        out["average_precision"]["planet vs false positive"] = {
            "ap": fast_average_precision(y, scores[tois_only]),
            "chance": float(y.mean()),
        }
    for n in (10, 50, 100, int(flagged.sum())):
        out["top"][str(n)] = dict(Counter(groups_arr[order[:n]].tolist()))
    return out


def sector_text(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Sector {summary['sector']}: {summary['n_stars']} stars ranked, against "
        f"{summary['n_toi_hosts_in_sector']} TOI hosts in the sector",
        f"{'group':<15} {'stars':>5} {'flagged':>8} {'median rank':>12} {'period found':>13} "
        f"{'flagged if found':>17}",
    ]
    for group, g in summary["groups"].items():
        found = f"{g['period_found']:.0%}" if "period_found" in g else "-"
        when = g.get("flagged_when_period_found")
        when_text = f"{when:.0%}" if when is not None else "-"
        lines.append(
            f"{group:<15} {g['n']:>5} {g['flagged']:>8.0%} {g['median_rank']:>12.0f} "
            f"{found:>13} {when_text:>17}"
        )
    lines.append("")
    for name, ap in summary["average_precision"].items():
        lines.append(f"average precision, {name}: {ap['ap']:.3f} (chance {ap['chance']:.3f})")
    lines.append("")
    for n, counts in summary["top"].items():
        parts = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        lines.append(f"top {n}: {parts}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m transitml.real_check",
        description="Check the pipeline against real TESS data (needs MAST for 'planets').",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    planets = sub.add_parser("planets", help="Fit known planets and compare with published values.")
    planets.add_argument("--published", type=Path, default=Path("data/real_planets/published.csv"))
    planets.add_argument("--out-dir", type=Path, default=Path("results/real_planets"))
    planets.add_argument("--model", type=Path, default=Path("results/model.joblib"))
    planets.add_argument("--fit-max-steps", type=int, default=None)
    sector = sub.add_parser("sector", help="Rank a batch's candidates against the TOI catalogue.")
    sector.add_argument("candidates", type=Path, help="candidates.csv written by transitml.batch.")
    sector.add_argument("--tois", type=Path, required=True, help="ExoFOP TOI table (CSV).")
    sector.add_argument("--sector", type=int, required=True)
    sector.add_argument("--out", type=Path, default=None, help="Default: beside candidates.csv.")
    args = parser.parse_args(argv)

    if args.command == "planets":
        rows = run_planets(
            args.published, args.out_dir, model=args.model, fit_max_steps=args.fit_max_steps
        )
        print(planets_table(rows))
        print(f"wrote {args.out_dir / 'comparison.csv'} and {args.out_dir / 'comparison.txt'}")
        return 0

    summary = sector_ranking(args.candidates, args.tois, args.sector)
    out = args.out or args.candidates.parent / "toi_ranking.json"
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    text = sector_text(summary)
    out.with_suffix(".txt").write_text(text)
    print(text, end="")
    print(f"wrote {out} and {out.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
