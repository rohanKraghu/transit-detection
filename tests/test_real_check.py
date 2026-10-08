"""The real-data checks' bookkeeping, offline: comparisons and the TOI ranking."""

from __future__ import annotations

import csv
import json
import math

import pytest

from transitml import real_check
from transitml.real_check import compare_planet, difference, planets_table, sector_ranking


def _summary(median, lower, upper):
    return {"median": median, "lower": lower, "upper": upper}


def test_the_difference_uses_the_side_of_the_interval_facing_the_published_value():
    fitted = _summary(1.0, 0.9, 1.3)  # -0.1 / +0.3
    assert difference(fitted, 1.6, 0.0) == pytest.approx(2.0)
    assert difference(fitted, 0.8, 0.0) == pytest.approx(-2.0)
    assert difference(fitted, 1.6, 0.4) == pytest.approx(0.6 / 0.5)
    assert math.isnan(difference(fitted, math.nan, 0.1))
    assert difference(fitted, 1.6, math.nan) == pytest.approx(2.0)  # no error given


PUBLISHED = {
    "pl_name": "Test b", "tic_id": "TIC 1", "sector": "3", "pl_orbper": "2.0",
    "pl_ratror": "0.1", "pl_ratrorerr1": "0.001", "pl_imppar": "", "pl_impparerr1": "",
    "pl_trandur": "2.0", "pl_trandurerr1": "0.05", "st_dens": "1.4", "st_denserr1": "0.1",
}


def _report(**fit_changes):
    parameters = {
        "period": _summary(2.0, 1.999, 2.001),
        "k": _summary(0.102, 0.101, 0.103),
        "b": _summary(0.3, 0.1, 0.5),
        "t14_hours": _summary(2.0, 1.95, 2.05),
        "rho_star": _summary(1.4, 1.3, 1.5),
    }
    fit = {"parameters": parameters, "sampler": {"converged": True}, "warnings": [],
           "density_check": {"stellar_density": 1.3, "consistent": True}, **fit_changes}
    return {"score": 0.9, "above_threshold": True, "candidates": [{"period": 2.0}], "fit": fit}


def test_a_planet_row_carries_fit_published_and_difference():
    row = compare_planet(PUBLISHED, _report())
    assert row["fit_status"] == "ok" and row["converged"] and row["flagged"]
    assert row["k_fit"] == 0.102 and row["k_published"] == 0.1
    assert row["k_diff"] == pytest.approx(-0.002 / math.hypot(0.001, 0.001))
    assert math.isnan(row["b_published"]) and math.isnan(row["b_diff"])
    assert row["tic_density"] == 1.3 and row["density_consistent"]
    table = planets_table([row])
    assert "Test b" in table and "0.1020 / 0.1000 (-1.4)" in table
    failed = compare_planet(PUBLISHED, {"score": 0.1, "candidates": [], "fit": {"error": "too few"}})
    assert failed["fit_status"] == "too few"
    assert "too few" in planets_table([failed])


def test_the_published_table_skips_comment_lines(tmp_path):
    path = tmp_path / "published.csv"
    path.write_text("# a note\n# another\npl_name,tic_id,sector\nTest b,TIC 1,3\n")
    assert real_check.read_published(path) == [{"pl_name": "Test b", "tic_id": "TIC 1", "sector": "3"}]


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_a_sector_is_ranked_against_the_tois_observed_in_it(tmp_path):
    tois = tmp_path / "toi.csv"
    _write_csv(tois, ["TIC ID", "TOI", "TFOPWG Disposition", "Period (days)", "Sectors"], [
        [1, "1.01", "CP", 3.0, "14,15"],
        [2, "2.01", "FP", 5.0, "14"],
        [3, "3.01", "PC", 7.0, "14"],
        [4, "4.01", "CP", 2.0, "15"],  # not observed in sector 14: a background star here
    ])
    candidates = tmp_path / "candidates.csv"
    _write_csv(candidates, ["target_id", "score", "flagged", "period_days"], [
        ["TIC 1", 0.9, "True", 6.0],   # twice the TOI period: an alias, found
        ["TIC 2", 0.8, "True", 5.0],
        ["TIC 5", 0.5, "False", 1.0],
        ["TIC 3", 0.4, "False", 2.2],  # wrong period
        ["TIC 4", 0.1, "False", 2.0],
    ])
    summary = sector_ranking(candidates, tois, 14)
    groups = summary["groups"]
    assert summary["n_toi_hosts_in_sector"] == 3
    assert groups["planet"] == {"n": 1, "flagged": 1.0, "median_rank": 1.0, "period_found": 1.0,
                                "flagged_when_period_found": 1.0}
    assert groups["open TOI"]["period_found"] == 0.0
    assert groups["open TOI"]["flagged_when_period_found"] is None
    assert groups["not a TOI"]["n"] == 2
    assert summary["average_precision"]["planet vs not a TOI"]["ap"] == pytest.approx(1.0)
    assert summary["top"]["2"] == {"planet": 1, "false positive": 1}
    json.dumps(summary, allow_nan=False)  # strict JSON, as the command writes it
    text = real_check.sector_text(summary)
    assert "planet" in text and "top 10" in text


def test_the_sector_command_writes_beside_the_candidates(tmp_path, capsys):
    tois = tmp_path / "toi.csv"
    _write_csv(tois, ["TIC ID", "TOI", "TFOPWG Disposition", "Period (days)", "Sectors"],
               [[1, "1.01", "KP", 3.0, "14"]])
    candidates = tmp_path / "candidates.csv"
    _write_csv(candidates, ["target_id", "score", "flagged", "period_days"],
               [["TIC 1", 0.9, "True", 3.0], ["TIC 2", 0.2, "False", 4.0]])
    assert real_check.main(["sector", str(candidates), "--tois", str(tois), "--sector", "14"]) == 0
    assert (tmp_path / "toi_ranking.json").exists() and (tmp_path / "toi_ranking.txt").exists()
    assert "Sector 14: 2 stars ranked" in capsys.readouterr().out


def test_the_centroid_veto_moves_off_target_stars_down(tmp_path):
    tois = tmp_path / "toi.csv"
    _write_csv(tois, ["TIC ID", "TOI", "TFOPWG Disposition", "Period (days)", "Sectors"], [
        [1, "1.01", "CP", 3.0, "14"],
        [2, "2.01", "FP", 5.0, "14"],
        [3, "3.01", "FP", 7.0, "14"],
    ])
    candidates = tmp_path / "candidates.csv"
    _write_csv(candidates, ["target_id", "score", "flagged", "period_days", "centroid_status",
                            "centroid_offset"], [
        ["TIC 2", 0.9, "True", 5.0, "ok", "True"],  # a blend, ranked first by score
        ["TIC 1", 0.8, "True", 3.0, "ok", "False"],
        ["TIC 3", 0.7, "True", 7.0, "no pixel file", "False"],
        ["TIC 4", 0.1, "False", 1.0, "", ""],
    ])
    summary = sector_ranking(candidates, tois, 14)
    centroid = summary["centroid"]
    assert centroid["groups"]["false positive"] == {"tested": 2, "placed": 1, "off_target": 1}
    assert centroid["groups"]["planet"] == {"tested": 1, "placed": 1, "off_target": 0}
    ap = centroid["average_precision"]["planet vs false positive"]
    assert ap["without"] == pytest.approx(0.5) and ap["with_veto"] == pytest.approx(1.0)
    assert "with it" in real_check.sector_text(summary)
