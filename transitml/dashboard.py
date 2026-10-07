"""The batch dashboard: one self-contained HTML page per batch run.

Everything is inline (styles, script and data), nothing is fetched, so the
file opens from disk, attaches to an email and still works in five years.
The page shows the run's headline numbers, a histogram of P(planet), and a
sortable, filterable table of every star with a thumbnail of its folded
light curve; clicking a row opens the star's SHAP reasons, its other
signals and, for the top candidates, a link to the full ``vet`` report.

Strings from the data (target ids come from user files) are only ever
inserted with ``textContent``, never as markup.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

#: Rows ranked below this (and not flagged) are sent in a compact form: the
#: table columns and the top reason, without the fold thumbnail, the other
#: signals or the full truth record.  That keeps a sector of tens of
#: thousands of stars to a few hundred bytes per star; candidates.csv has
#: everything for every star.
THUMBNAIL_RANKS = 300

_TABLE_DIGITS = {
    "p_planet": 4, "score": 5, "period_days": 6, "depth_ppm": 4,
    "duration_hours": 3, "depth_snr": 3, "sde": 3,
}


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, float) and math.isfinite(value):
        return float(f"{value:.{digits}g}")
    return value


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    rank = row.get("rank")
    detailed = bool(row.get("flagged")) or (rank is not None and rank <= THUMBNAIL_RANKS)
    out: dict[str, Any] = {
        "id": row["id"],
        "rank": rank,
        "status": row["status"],
        "flagged": bool(row.get("flagged")),
    }
    for key, digits in _TABLE_DIGITS.items():
        if row.get(key) is not None:
            out[key] = _round(row[key], digits)
    if row["status"] == "ok":
        out["n_signals"] = len(row.get("signals", []))
    reasons = row.get("reasons", [])[: 3 if detailed else 1]
    out["reasons"] = [
        {"feature": r["feature"], "value": _round(r["value"], 3), "shap": _round(r["shap"], 3)}
        for r in reasons
    ]
    truth = row.get("truth")
    if truth:
        out["truth"] = dict(truth) if detailed else {
            k: truth[k] for k in ("label", "kind") if k in truth
        }
    if row.get("error"):
        out["error"] = row["error"]
    if detailed:
        for key in ("sector", "n_cadences", "baseline_days", "report"):
            if row.get(key) is not None:
                out[key] = row[key]
        if row.get("fit"):
            fit = row["fit"]
            out["fit"] = {
                k: fit[k]
                for k in ("status", "error", "parameters", "converged", "warnings",
                          "density_ratio", "density_consistent")
                if k in fit
            }
        if row.get("centroid"):
            out["centroid"] = {
                k: row["centroid"][k]
                for k in ("status", "message", "significant", "offset_distance_pixels",
                          "offset_arcsec", "offset_sigma", "difference_snr")
                if k in row["centroid"]
            }
        out["signals"] = [
            {k: _round(v, 5) for k, v in sig.items() if k != "depth_snr"}
            for sig in row.get("signals", [])
        ]
        if row.get("fold"):
            fold = row["fold"]
            out["fold"] = {
                "half_window_hours": fold["half_window_hours"],
                "duration_hours": fold["duration_hours"],
                "ppt": [_round(v, 3) for v in fold["ppt"]],
            }
    return out


def dashboard_payload(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    """The data embedded in the page: every star, trimmed to what the page shows."""
    return {"summary": summary, "rows": [_slim(row) for row in rows]}


def render_dashboard(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """The page as a string."""
    payload = json.dumps(dashboard_payload(rows, summary), allow_nan=False, separators=(",", ":"))
    # A "</script>" inside a string would end the data block early.
    payload = payload.replace("</", "<\\/")
    title = html.escape(f"Transit candidates: {summary.get('source', 'batch')}")
    return (
        _TEMPLATE.replace("__TITLE__", title)
        .replace("__THUMBNAIL_RANKS__", str(THUMBNAIL_RANKS))
        .replace("__DATA__", payload)
    )


def write_dashboard(rows: list[dict[str, Any]], summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(rows, summary))
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #fcfcfb; --surface: #ffffff; --ink: #0b0b0b; --ink-soft: #52514e;
  --grid: #e6e5e1; --accent: #2a78d6; --warm: #eb6834; --cool: #1baf7a; --neutral: #8d8b84;
  --hover: rgba(42, 120, 214, 0.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141413; --surface: #1c1c1a; --ink: #f2f1ed; --ink-soft: #a8a69f;
    --grid: #33322f; --accent: #5b9be6; --warm: #f08a5d; --cool: #3cc995; --neutral: #7d7b75;
    --hover: rgba(91, 155, 230, 0.10);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 28px 16px 56px; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 15px; margin: 0 0 10px; }
.sub { color: var(--ink-soft); margin: 0 0 22px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; }
.card { background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 12px 14px; }
.card .v { font-size: 23px; font-weight: 650; font-variant-numeric: tabular-nums; }
.card .k { color: var(--ink-soft); font-size: 12.5px; }
.card.truth { border-color: var(--cool); }
.panels { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 12px; margin-top: 12px; }
.panel { background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 14px; }
.panel p { margin: 0 0 6px; color: var(--ink-soft); }
.panel code { font-size: 12.5px; }
@media (max-width: 760px) { .panels { grid-template-columns: 1fr; } }
.controls { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin: 22px 0 10px; }
.controls input[type=search] { flex: 1 1 220px; max-width: 340px; padding: 7px 10px; border-radius: 8px;
  border: 1px solid var(--grid); background: var(--surface); color: var(--ink); font: inherit; }
.controls label { color: var(--ink-soft); display: flex; gap: 6px; align-items: center; }
.count { color: var(--ink-soft); margin-left: auto; }
.table-wrap { overflow-x: auto; border: 1px solid var(--grid); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th { position: sticky; top: 0; background: var(--surface); text-align: right; font-weight: 600;
  color: var(--ink-soft); padding: 9px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap;
  user-select: none; }
th.sortable { cursor: pointer; }
th.sortable:hover { color: var(--ink); }
th .arrow { display: inline-block; width: 0.9em; }
td { padding: 5px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
.l { text-align: left; }
tr.row { cursor: pointer; }
tr.row:hover { background: var(--hover); }
tr.flagged td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
tr.error td { color: var(--ink-soft); }
.p { display: inline-flex; align-items: center; gap: 6px; justify-content: flex-end; }
.p .bar { width: 46px; height: 6px; border-radius: 3px; background: var(--grid); overflow: hidden; }
.p .bar i { display: block; height: 100%; background: var(--accent); }
.tag { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 12px; border: 1px solid var(--grid); }
.tag.planet { border-color: var(--cool); color: var(--cool); }
.tag.other { color: var(--ink-soft); }
tr.detail td { background: var(--bg); white-space: normal; text-align: left; padding: 12px 16px 16px; }
/* The table can be wider than the page; keep the detail inside the visible part of it. */
.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px;
  position: sticky; left: 16px; width: calc(var(--wrap-width, 100%) - 32px); }
/* Narrow enough for the four-column detail grid a fitted star gets. */
.reason { display: grid; grid-template-columns: minmax(0, 1fr) minmax(48px, 96px) 44px; gap: 8px; align-items: center; margin: 3px 0; }
.reason > span:first-child { overflow-wrap: anywhere; }
.reason .track { position: relative; height: 10px; background: var(--grid); border-radius: 2px; }
.reason .track i { position: absolute; top: 0; height: 100%; border-radius: 2px; }
.reason .track .mid { position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: var(--neutral); }
.muted { color: var(--ink-soft); }
.warn { color: var(--warm); }
a { color: var(--accent); }
.more { margin: 12px auto 0; display: block; padding: 8px 16px; border-radius: 8px; border: 1px solid var(--grid);
  background: var(--surface); color: var(--ink); font: inherit; cursor: pointer; }
footer { color: var(--ink-soft); font-size: 12.5px; margin-top: 26px; max-width: 900px; }
footer p { margin: 0 0 6px; }
svg text { fill: var(--ink-soft); font-size: 10px; }
</style>
</head>
<body>
<main>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>
  <section class="cards" id="cards"></section>
  <section class="panels">
    <div class="panel"><h2>P(planet) of every vetted star</h2><div id="hist"></div></div>
    <div class="panel" id="about"><h2>How to read this</h2></div>
  </section>
  <div class="controls">
    <input type="search" id="q" placeholder="Filter by target id" aria-label="Filter by target id">
    <label><input type="checkbox" id="flaggedOnly"> flagged only</label>
    <label><input type="checkbox" id="errorsToo"> show failed stars</label>
    <span class="count" id="count"></span>
  </div>
  <div class="table-wrap"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
  <button class="more" id="more" type="button">Show more</button>
  <footer id="footer"></footer>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const S = DATA.summary;
const ROWS = DATA.rows;
const HAS_TRUTH = ROWS.some(r => r.truth);
const THUMBNAIL_RANKS = __THUMBNAIL_RANKS__;
const SVG = "http://www.w3.org/2000/svg";
const state = { sort: "rank", dir: 1, q: "", flaggedOnly: false, errorsToo: false, shown: 100, open: new Set() };

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v; else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
function svg(tag, attrs) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  return node;
}
const fmt = {
  num: (v, d) => v == null ? "" : Number(v).toFixed(d),
  sig: v => v == null ? "" : Number(v).toPrecision(3).replace(/\.?0+$/, ""),
  pct: v => v == null ? "" : (v >= 0.995 ? ">99%" : v < 0.001 ? "<0.1%" : (100 * v).toFixed(v < 0.1 ? 1 : 0) + "%"),
  int: v => v == null ? "" : Math.round(v).toLocaleString(),
};

// ---- header and cards ----------------------------------------------------
document.getElementById("title").textContent = "Transit candidates";
document.getElementById("subtitle").textContent =
  S.source + " · model " + (S.model.fingerprint || "") + " · " + S.runtime_seconds + " s (" +
  S.computed + " vetted now, " + S.from_cache + " from the cache)";
const cards = document.getElementById("cards");
function card(value, label, cls) {
  cards.append(el("div", { class: "card" + (cls ? " " + cls : "") }, el("div", { class: "v" }, value), el("div", { class: "k" }, label)));
}
card(fmt.int(S.n_vetted), "stars vetted" + (S.n_errors ? " (" + S.n_errors + " failed)" : ""));
card(fmt.int(S.n_flagged), "flagged at the operating threshold");
card(fmt.num(S.expected_planets_flagged, 1), "planets expected among the flagged");
card(fmt.pct(S.threshold_probability), "threshold as P(planet)");
if (S.truth) {
  const t = S.truth;
  card(t.flagged_planets + " / " + S.n_flagged, "flagged that are planets (ground truth)", "truth");
  card(t.recall == null ? "n/a" : fmt.pct(t.recall), "of " + t.n_planets + " planets flagged (ground truth)", "truth");
}

// ---- histogram of P(planet), log axis -------------------------------------
(function histogram() {
  const W = 640, H = 170, L = 38, R = 10, T = 8, B = 30, lo = -4, hi = 0, nb = 32;
  const ok = ROWS.filter(r => r.p_planet != null);
  const binOf = p => Math.min(nb - 1, Math.max(0, Math.floor((Math.log10(Math.max(p, 1e-4)) - lo) / (hi - lo) * nb)));
  const counts = [new Array(nb).fill(0), new Array(nb).fill(0)];
  for (const r of ok) counts[r.flagged ? 1 : 0][binOf(r.p_planet)]++;
  const total = counts[0].map((c, i) => c + counts[1][i]);
  const ymax = Math.max(1, ...total);
  const ly = c => c <= 0 ? 0 : Math.log10(1 + c) / Math.log10(1 + ymax);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
    "aria-label": "Histogram of P(planet) on a log axis, flagged stars in blue" });
  const x = v => L + (v - lo) / (hi - lo) * (W - L - R);
  const y = f => T + (1 - f) * (H - T - B);
  for (const tick of [-4, -3, -2, -1, 0]) {
    s.append(svg("line", { x1: x(tick), x2: x(tick), y1: T, y2: H - B, stroke: "var(--grid)" }));
    const label = svg("text", { x: x(tick), y: H - B + 14, "text-anchor": "middle" });
    label.textContent = tick === 0 ? "1" : "1e" + tick;
    s.append(label);
  }
  const bw = (W - L - R) / nb;
  for (let i = 0; i < nb; i++) {
    const all = ly(total[i]);
    if (total[i] === 0) continue;
    const share = counts[1][i] / total[i];
    s.append(svg("rect", { x: L + i * bw + 1, width: bw - 2, y: y(all), height: y(0) - y(all), fill: "var(--neutral)", opacity: 0.45 }));
    if (counts[1][i]) {
      const fy = all * share;
      s.append(svg("rect", { x: L + i * bw + 1, width: bw - 2, y: y(fy), height: y(0) - y(fy), fill: "var(--accent)" }));
    }
  }
  const tp = Math.log10(Math.max(S.threshold_probability, 1e-4));
  s.append(svg("line", { x1: x(tp), x2: x(tp), y1: T, y2: H - B, stroke: "var(--ink-soft)", "stroke-dasharray": "4 3" }));
  const tl = svg("text", { x: x(tp) + 4, y: T + 10 });
  tl.textContent = "threshold";
  s.append(tl);
  const xl = svg("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle" });
  xl.textContent = "P(planet), log scale; bar height log(1 + stars); blue share = flagged";
  s.append(xl);
  document.getElementById("hist").append(s);
})();

// ---- notes ----------------------------------------------------------------
(function notes() {
  const about = document.getElementById("about");
  const rate = fmt.pct(S.planet_rate);
  const lines = [
    ["Score", " is the classifier's output. A star is flagged when its score reaches " + fmt.num(S.threshold, 3) +
      ", the operating threshold chosen in training for a precision of at least 0.5."],
    ["P(planet)", " is the calibrated probability that the star hosts a detectable transiting planet, for a population where " +
      rate + " of stars do. It ranks exactly like the score."],
    ["Reasons", " are SHAP values in log-odds: how far each feature moved this star from the average star. Positive pushes towards planet; +1 multiplies the odds by e."],
    ["Fold", " is the detrended light curve folded on the strongest signal; the shaded band is the transit duration."],
  ];
  for (const [b, t] of lines) about.append(el("p", {}, el("strong", {}, b), t));
  const foot = document.getElementById("footer");
  foot.append(el("p", {}, "Model: " + S.model.path + " (trained on " + (S.model.n_train || "?") + " stars, " +
    (S.model.n_train_positive || "?") + " planets; seed " + (S.model.seed ?? "?") + "). Search: up to " +
    S.settings.max_signals + " signals per star, SDE at least " + S.settings.min_sde + "."));
  if (S.truth) foot.append(el("p", {}, "Ground truth is known because these stars are synthetic or injected; real survey stars have none, and these cards disappear."));
  foot.append(el("p", {}, "Files beside this page: candidates.csv (every star, ranked), summary.json, reports/ (full vet reports for the top candidates)."));
})();

// ---- table ------------------------------------------------------------------
const COLUMNS = [
  { key: "rank", label: "#", f: r => r.rank ?? "" },
  { key: "id", label: "Target", cls: "l", f: r => r.id },
  { key: "p_planet", label: "P(planet)", f: pcell },
  { key: "score", label: "Score", f: r => fmt.num(r.score, 3) },
  { key: "period_days", label: "Period (d)", f: r => fmt.num(r.period_days, 4) },
  { key: "depth_ppm", label: "Depth (ppm)", f: r => fmt.int(r.depth_ppm) },
  { key: "duration_hours", label: "T14 (h)", f: r => fmt.num(r.duration_hours, 2) },
  { key: "depth_snr", label: "SNR", f: r => fmt.num(r.depth_snr, 1) },
  { key: "sde", label: "SDE", f: r => fmt.num(r.sde, 1) },
  { key: "n_signals", label: "Signals", f: r => r.n_signals ?? "" },
  { key: "fold", label: "Fold", sortable: false, f: foldCell },
  { key: "reason", label: "Top reason", cls: "l", sortable: false, f: reasonCell },
];
if (HAS_TRUTH) COLUMNS.push({ key: "truth", label: "Truth", cls: "l", f: truthCell, sortKey: r => r.truth ? r.truth.label : -1 });

function pcell(r) {
  if (r.p_planet == null) return r.status === "ok" ? "" : "failed";
  const bar = el("span", { class: "bar" }, el("i", {}));
  bar.firstChild.style.width = Math.max(2, 100 * r.p_planet).toFixed(1) + "%";
  return el("span", { class: "p" }, bar, fmt.pct(r.p_planet));
}
function foldCell(r) {
  const f = r.fold;
  if (!f || !f.ppt || !f.ppt.length || f.half_window_hours == null) return "";
  const W = 120, H = 30, vals = f.ppt.filter(v => v != null);
  if (!vals.length) return "";
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 0), span = (hi - lo) || 1;
  const s = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" });
  const half = f.half_window_hours, d = f.duration_hours || 0;
  const xw = Math.min(W, W * d / (2 * half));
  s.append(svg("rect", { x: (W - xw) / 2, y: 0, width: xw, height: H, fill: "var(--accent)", opacity: 0.12 }));
  const yz = 2 + (hi - 0) / span * (H - 4);
  s.append(svg("line", { x1: 0, x2: W, y1: yz, y2: yz, stroke: "var(--grid)" }));
  let d_ = "";
  f.ppt.forEach((v, i) => {
    if (v == null) return;
    const px = (i + 0.5) / f.ppt.length * W, py = 2 + (hi - v) / span * (H - 4);
    d_ += (d_ ? "L" : "M") + px.toFixed(1) + " " + py.toFixed(1);
  });
  s.append(svg("path", { d: d_, fill: "none", stroke: r.flagged ? "var(--accent)" : "var(--neutral)", "stroke-width": 1.4 }));
  return s;
}
function reasonCell(r) {
  const top = (r.reasons || [])[0];
  if (!top || top.shap == null) return r.error ? el("span", { class: "muted" }, r.error.slice(0, 60)) : "";
  return el("span", {}, top.feature + " ", el("span", { class: "muted" }, (top.shap >= 0 ? "+" : "") + top.shap.toFixed(2)));
}
function truthCell(r) {
  if (!r.truth) return "";
  const kind = r.truth.kind || (r.truth.label ? "planet" : "not a planet");
  return el("span", { class: "tag " + (r.truth.label ? "planet" : "other") }, kind.replace(/_/g, " "));
}

const FIT_ROWS = [
  ["k", "Rp/R*", 4], ["b", "impact parameter b", 2], ["t14_hours", "T14 (h)", 2],
  ["depth_ppm", "depth (ppm)", 0], ["rho_star", "stellar density (g/cm3)", 2],
  ["rp_earth", "Rp (Earth radii)", 2],
];
function fitBlock(fit) {
  const box = el("div", {}, el("h2", {}, "Transit fit (batman + emcee)"));
  if (fit.status !== "ok") {
    box.append(el("p", { class: "muted" }, "fit failed: " + (fit.error || "")));
    return box;
  }
  for (const [key, label, digits] of FIT_ROWS) {
    const v = (fit.parameters || {})[key];
    if (!v || v[0] == null) continue;
    const [m, lo, hi] = v;
    box.append(el("div", {}, label + " = " + m.toFixed(digits) + " (+" + (hi - m).toFixed(digits) +
      " / -" + (m - lo).toFixed(digits) + ")"));
  }
  if (fit.density_ratio) box.append(el("div", { class: fit.density_consistent ? "" : "warn" },
    "fitted / stellar density " + fit.density_ratio[0].toFixed(2) +
    (fit.density_consistent ? ", consistent with the star" : ", inconsistent with the star")));
  if (!fit.converged) box.append(el("div", { class: "warn" }, "chain not converged: intervals are rough"));
  for (const w of (fit.warnings || []).filter(w => !w.includes("autocorrelation")))
    box.append(el("div", { class: "muted" }, w));
  return box;
}

function centroidBlock(c) {
  const box = el("div", {}, el("h2", {}, "Centroid test (target pixels)"));
  if (c.status !== "ok") {
    box.append(el("p", { class: "muted" }, "not placed: " + (c.message || c.status)));
    return box;
  }
  box.append(el("div", { class: c.significant ? "warn" : "" }, c.significant
    ? "the dip sits off the target: likely another star"
    : "the dip sits on the target"));
  box.append(el("div", {}, "offset " + fmt.num(c.offset_distance_pixels, 2) + " px (" +
    fmt.num(c.offset_arcsec, 1) + "\"), " + fmt.num(c.offset_sigma, 1) + " sigma"));
  box.append(el("div", { class: "muted" }, "difference image SNR " + fmt.num(c.difference_snr, 1)));
  return box;
}

function detailRow(r, ncol) {
  const grid = el("div", { class: "detail-grid" });
  const reasons = el("div", {}, el("h2", {}, "Why (SHAP, log-odds)"));
  const rs = (r.reasons || []).filter(x => x.shap != null);
  const m = Math.max(0.5, ...rs.map(x => Math.abs(x.shap)));
  for (const x of rs) {
    const track = el("span", { class: "track" }, el("span", { class: "mid" }));
    const bar = el("i", {});
    const w = 50 * Math.abs(x.shap) / m;
    bar.style.left = (x.shap >= 0 ? 50 : 50 - w) + "%";
    bar.style.width = w + "%";
    bar.style.background = x.shap >= 0 ? "var(--accent)" : "var(--warm)";
    track.append(bar);
    reasons.append(el("div", { class: "reason" },
      el("span", {}, x.feature + " = " + fmt.sig(x.value)), track,
      el("span", {}, (x.shap >= 0 ? "+" : "") + x.shap.toFixed(2))));
  }
  if (!rs.length) reasons.append(el("p", { class: "muted" }, r.error || "no reasons"));
  grid.append(reasons);
  const sig = el("div", {}, el("h2", {}, "Signals from the iterative search"));
  for (const [i, s] of (r.signals || []).entries()) {
    sig.append(el("div", {}, (i + 1) + ". P = " + fmt.num(s.period_days, 4) + " d, depth " + fmt.int(s.depth_ppm) +
      " ppm, T14 " + fmt.num(s.duration_hours, 2) + " h, SDE " + fmt.num(s.sde, 1)));
  }
  if (r.signals === undefined) sig.append(el("p", { class: "muted" },
    "kept in candidates.csv: the page carries full detail for flagged stars and the top " + THUMBNAIL_RANKS));
  else if (!r.signals.length) sig.append(el("p", { class: "muted" }, "no signal above the SDE floor"));
  grid.append(sig);
  const more = el("div", {}, el("h2", {}, "More"));
  more.append(el("div", {}, r.n_cadences ? r.n_cadences + " cadences over " + fmt.num(r.baseline_days, 1) + " d" : ""));
  if (r.sector != null) more.append(el("div", {}, "sector " + r.sector));
  if (r.truth) more.append(el("div", {}, "truth: " + (r.truth.kind || r.truth.label) +
    (r.truth.period ? ", injected P = " + fmt.num(r.truth.period, 4) + " d" : "") +
    (r.truth.true_snr != null ? ", SNR " + fmt.num(r.truth.true_snr, 1) : "")));
  if (r.report) more.append(el("div", {}, el("a", { href: r.report }, "full vet report")));
  grid.append(more);
  if (r.fit) grid.append(fitBlock(r.fit));
  if (r.centroid) grid.append(centroidBlock(r.centroid));
  return el("tr", { class: "detail" }, el("td", { colspan: ncol }, grid));
}

function sortValue(r, key) {
  const c = COLUMNS.find(c => c.key === key);
  if (c && c.sortKey) return c.sortKey(r);
  const v = r[key];
  return v == null ? null : v;
}
function visible() {
  const q = state.q.trim().toLowerCase();
  let out = ROWS.filter(r => (state.errorsToo || r.status === "ok") &&
    (!state.flaggedOnly || r.flagged) && (!q || r.id.toLowerCase().includes(q)));
  const k = state.sort, dir = state.dir;
  out.sort((a, b) => {
    const x = sortValue(a, k), y = sortValue(b, k);
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * dir;
  });
  return out;
}
function renderHead() {
  const head = document.getElementById("head");
  head.replaceChildren();
  for (const c of COLUMNS) {
    const sortable = c.sortable !== false;
    const arrow = el("span", { class: "arrow" }, state.sort === c.key ? (state.dir > 0 ? "▲" : "▼") : "");
    const th = el("th", { class: (c.cls || "") + (sortable ? " sortable" : "") }, c.label, arrow);
    if (sortable) th.addEventListener("click", () => {
      if (state.sort === c.key) state.dir = -state.dir;
      else { state.sort = c.key; state.dir = (c.key === "rank" || c.key === "id") ? 1 : -1; }
      renderHead(); renderBody();
    });
    head.append(th);
  }
}
function renderBody() {
  const body = document.getElementById("body");
  body.replaceChildren();
  const rows = visible();
  for (const r of rows.slice(0, state.shown)) {
    const tr = el("tr", { class: "row" + (r.flagged ? " flagged" : "") + (r.status !== "ok" ? " error" : "") });
    for (const c of COLUMNS) tr.append(el("td", { class: c.cls || "" }, c.f(r)));
    tr.addEventListener("click", () => {
      if (state.open.has(r.id)) state.open.delete(r.id); else state.open.add(r.id);
      renderBody();
    });
    body.append(tr);
    if (state.open.has(r.id)) body.append(detailRow(r, COLUMNS.length));
  }
  document.getElementById("count").textContent =
    Math.min(state.shown, rows.length) + " of " + rows.length + " shown";
  document.getElementById("more").style.display = rows.length > state.shown ? "block" : "none";
}
document.getElementById("q").addEventListener("input", e => { state.q = e.target.value; state.shown = 100; renderBody(); });
document.getElementById("flaggedOnly").addEventListener("change", e => { state.flaggedOnly = e.target.checked; state.shown = 100; renderBody(); });
document.getElementById("errorsToo").addEventListener("change", e => { state.errorsToo = e.target.checked; renderBody(); });
document.getElementById("more").addEventListener("click", () => { state.shown += 200; renderBody(); });
function fitDetail() {
  const wrap = document.querySelector(".table-wrap");
  wrap.style.setProperty("--wrap-width", wrap.clientWidth + "px");
}
window.addEventListener("resize", fitDetail);
fitDetail();
renderHead();
renderBody();
</script>
</body>
</html>
"""
