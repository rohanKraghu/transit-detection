"""TOI catalogue parsing and the per-star labels the real-label benchmark uses."""

from __future__ import annotations

import math

import pytest

from transitml.data.toi import (
    TOI,
    false_positive_reason,
    parse_sector_spec,
    parse_sectors,
    read_toi_comments,
    read_toi_table,
    select_benchmark_targets,
    star_label,
)

EXOFOP = """\
TIC ID,TOI,TESS Disposition,TFOPWG Disposition,TESS Mag,Epoch (BJD),Period (days),Duration (hours),Depth (ppm),Planet SNR,Sectors
100,101.01,KP,KP,9.5,2458683.1,3.5,2.1,5000,80.2,"14,15"
200,102.01,PC,FP,11.0,2458684.2,1.2,1.5,9000,40.0,"14"
200,102.02,PC,FA,11.0,2458684.3,7.0,3.0,300,8.0,"14"
300,103.01,PC,PC,12.0,2458685.0,5.0,2.0,800,10.0,"14"
400,104.01,PC,CP,10.0,2458686.0,9.0,3.5,1200,15.0,"41,15"
400,104.02,PC,PC,10.0,2458686.5,2.0,1.5,400,9.0,"41,15"
500,105.01,PC,FP,12.5,,,,,,"1,2"
600,106.01,PC,FP,12.5,2458687.0,4.0,2.0,700,12.0,"14"
600,106.02,PC,PC,12.5,2458687.2,6.0,2.0,500,9.0,"14"
"""


@pytest.fixture
def exofop(tmp_path):
    path = tmp_path / "exofop_toi.csv"
    path.write_text(EXOFOP)
    return read_toi_table(path)


def test_reads_the_exofop_export(exofop):
    assert len(exofop) == 9
    first = exofop[0]
    assert first == TOI(
        tic=100,
        toi="101.01",
        disposition="KP",
        period=3.5,
        epoch_bjd=2458683.1,
        duration_hours=2.1,
        depth_ppm=5000.0,
        snr=80.2,
        tess_mag=9.5,
        sectors=(14, 15),
    )
    # The TFOPWG column is the label, not the pipeline's own TESS Disposition.
    assert exofop[1].disposition == "FP"
    assert math.isnan(exofop[6].period)


def test_reads_the_exoplanet_archive_and_short_layouts(tmp_path):
    archive = tmp_path / "toi_archive.csv"
    archive.write_text("# NASA Exoplanet Archive\ntoi,tid,tfopwg_disp,pl_orbper\n101.01,100,cp,3.5\n")
    (row,) = read_toi_table(archive)
    assert (row.tic, row.disposition, row.period, row.sectors) == (100, "CP", 3.5, ())

    short = tmp_path / "toi.csv"  # the layout of data/real_injection/toi.csv
    short.write_text('TIC ID,TOI,Disposition\n299799658,"1062.01","CP"\n')
    assert read_toi_table(short)[0].label == 1

    bad = tmp_path / "bad.csv"
    bad.write_text("TIC ID,Period\n1,2\n")
    with pytest.raises(ValueError, match="disposition"):
        read_toi_table(bad)


def test_sector_parsing():
    assert parse_sectors("14, 15,41") == (14, 15, 41)
    assert parse_sectors("") == ()
    assert parse_sector_spec("14") == [14]
    assert parse_sector_spec("16,14-15,14") == [16, 14, 15]
    with pytest.raises(ValueError):
        parse_sector_spec("26-14")


def test_star_labels():
    def toi(disposition):
        return TOI(tic=1, toi="1.01", disposition=disposition)

    assert star_label([toi("CP")]) == 1
    assert star_label([toi("KP"), toi("FP")]) == 1  # a planet host, whatever else
    assert star_label([toi("CP"), toi("PC")]) == 1
    assert star_label([toi("FP"), toi("FA")]) == 0
    assert star_label([toi("FP"), toi("PC")]) is None  # the PC might be real
    assert star_label([toi("APC")]) is None
    assert star_label([toi("")]) is None
    assert star_label([]) is None


def test_selection_one_sector_per_star_in_preference_order(exofop):
    targets, counts = select_benchmark_targets(exofop, [15, 14])
    by_tic = {t.tic: t for t in targets}
    assert sorted(by_tic) == [100, 200, 400]
    assert by_tic[100].sector == 15  # observed in 14 and 15; 15 is preferred
    assert by_tic[200].sector == 14
    assert by_tic[400].sector == 15
    assert [t.label for t in targets] == [1, 0, 1]
    assert counts == {
        "stars_in_table": 6,
        "unlabelled": 2,  # TIC 300 (PC) and TIC 600 (FP beside an open PC)
        "not_in_sectors": 1,  # TIC 500
        "in_training_set": 0,
        "selected": 3,
        "positives": 2,
        "negatives": 1,
    }


def test_reference_toi_agrees_with_the_label(exofop):
    targets, _ = select_benchmark_targets(exofop, [14, 15])
    by_tic = {t.tic: t for t in targets}
    # TIC 200 has two false-positive TOIs; the higher-SNR one is the reference.
    assert by_tic[200].reference.toi == "102.01"
    # TIC 400's reference is its CP, not the higher-numbered open PC.
    assert by_tic[400].reference.toi == "104.01"
    assert len(by_tic[400].tois) == 2


def test_training_stars_are_excluded(exofop):
    targets, counts = select_benchmark_targets(exofop, [14], exclude_tics={100})
    assert [t.tic for t in targets] == [200]
    assert counts["in_training_set"] == 1


def test_a_zero_period_is_unknown(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("TIC ID,TFOPWG Disposition,Period (days)\n1,CP,0\n2,CP,-1\n")
    assert all(math.isnan(t.period) for t in read_toi_table(path))


@pytest.mark.parametrize(
    ("comment", "reason"),
    [
        ("retired as TFOP FP/NEB", "off target"),
        ("Could be on neighbor; TFOP FP/NEB", "off target"),
        ("centroid offset to SW in QLP s71+s73", "off target"),
        ("v-shaped; Centroids show source is TIC 95129100", "off target"),
        ("EB at 2x alerted period off-target on TIC 294176981; TFOP FP", "off target"),
        ("retired as TFOP NPC (nearby planet candidate)", "off target"),
        ("TFOP FP/ NPC; true source is TIC 147660207", "off target"),
        ("SG1 determined correct source as TIC 431899136", "off target"),
        ("v-shaped; this signal is actually from neighboring TIC 420112589.01", "off target"),
        ("2000 ppm secondary; centered on TIC 840790978", "off target"),
        ("TFOP FP/EB/SB2", "binary on target"),
        ("TFOP FP(SEB1)", "binary on target"),
        ("V-shaped; crowded field", "binary on target"),
        ("likely eccentric EB; 1000ppm secondary;", "binary on target"),
        ("found in faint-star QLP search", "not stated"),
        ("", "not stated"),
    ],
)
def test_false_positive_reason(comment, reason):
    assert false_positive_reason(comment) == reason


def test_read_toi_comments(tmp_path):
    path = tmp_path / "toi_comments.csv"
    path.write_text(
        "# a header line\nTOI,Comments\n102.01,\"TFOP FP/NEB, offset to TIC 5\"\n102.02,\n"
    )
    assert read_toi_comments(path) == {"102.01": "TFOP FP/NEB, offset to TIC 5", "102.02": ""}
