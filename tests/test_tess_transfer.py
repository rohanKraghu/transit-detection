"""The Kepler-to-TESS transfer test, offline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from transitml import tess_transfer as tt
from transitml.data.base import LightCurve
from transitml.data.toi import TOI, BenchmarkTarget

from .test_kepler_dr25 import make_star


def toi(name, disposition, period=3.3, epoch=2458700.5, hours=3.0, tic=1):
    return TOI(tic=tic, toi=name, disposition=disposition, period=period,
               epoch_bjd=epoch, duration_hours=hours, depth_ppm=1000.0, snr=20.0)


def test_ephemeris_converts_bjd_to_btjd_and_rejects_gaps():
    eph = tt.toi_ephemeris(toi("1.01", "CP"))
    assert eph.epoch == pytest.approx(1700.5) and eph.duration == pytest.approx(0.125)
    assert tt.toi_ephemeris(toi("1.02", "PC", period=float("nan"))) is None


def test_every_toi_on_a_star_is_folded():
    time, flux = make_star(period=3.3, epoch=1700.5 - 1690.0, duration=0.125, depth=1e-3, days=27)
    lc = LightCurve("TIC 7", time + 1690.0, flux, np.full(time.size, 1e-4))
    target = BenchmarkTarget(
        tic=7, label=1, sector=20, reference=toi("1.01", "CP", tic=7),
        tois=(toi("1.01", "CP", tic=7), toi("1.02", "PC", period=float("nan"), tic=7),
              toi("1.03", "FP", period=5.1, tic=7)),
    )
    ts = tt.build_toi_set([lc], [target])
    assert list(ts.tce_ids) == ["1.01", "1.03"]
    assert list(ts.labels) == [1, 0]
    assert ts.views["local"].shape == (2, 201)
    assert ts.views["local"][0].min() == pytest.approx(-1.0)
    assert ts.scalars["depth_scale"][0] == pytest.approx(1e-3, rel=0.15)


def test_star_score_is_the_label_blind_maximum():
    best = tt.star_max(np.array([1, 1, 2]), np.array([0.2, 0.9, 0.4]))
    assert best == {1: 0.9, 2: 0.4}


def test_compare_scores_common_stars_and_the_recovered_subset():
    rng = np.random.default_rng(0)
    stars, cnn, gbm = [], {}, {}
    for tic in range(60):
        label = tic % 2
        stars.append({"target_id": f"TIC {tic}", "disposition": "CP" if label else "FP",
                      "model_score": float(rng.random()), "period_recovered": tic % 3 != 0})
        cnn[tic] = label + 0.5 * rng.random()
        gbm[tic] = rng.random()
    del gbm[59]  # a star one model could not score drops out everywhere
    result = tt.compare({"kepler_cnn": cnn, "kepler_gbm": gbm}, {"tess_synthetic": stars},
                        cnn_threshold=0.75, n_tois_scored=70)
    assert result.n_stars == 59
    assert result.average_precision["kepler_cnn"] > 0.95
    assert result.recovered_subset["n_stars"] == sum(1 for t in range(59) if t % 3 != 0)
    lo, hi = result.cnn_minus_best_tess
    assert 0 < lo <= hi
    report = tt.format_transfer_report(result)
    assert "Kepler DR25 CNN" in report and "found the TOI period" in report
    json.dumps(result.to_dict(), default=str)
