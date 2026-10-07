"""The Kepler DR25 training set, run offline: labels, FITS, views, model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transitml import kepler_dr25
from transitml.data import kepler
from transitml.data.base import LightCurve
from transitml.data.kepler import KeplerTCE
from transitml.views import (
    Ephemeris,
    ViewConfig,
    detrend_masked,
    in_transit_mask,
    knot_spacing_for,
    make_views,
)

CADENCE = 0.0204  # Kepler long cadence, days


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def _tce(kepid, plnt, period=10.0):
    return {
        "kepid": str(kepid),
        "tce_plnt_num": str(plnt),
        "tce_period": str(period),
        "tce_time0bk": "135.0",
        "tce_duration": "4.0",
        "tce_depth": "500",
        "tce_max_mult_ev": "12.5",
    }


def _koi(kepid, plnt, disposition, nt=0, name="K00001.01"):
    return {
        "kepid": str(kepid),
        "koi_tce_plnt_num": str(plnt),
        "kepoi_name": name,
        "koi_pdisposition": disposition,
        "koi_disposition": "CONFIRMED" if disposition == "CANDIDATE" else disposition,
        "koi_fpflag_nt": str(nt),
        "koi_fpflag_ss": "0",
        "koi_fpflag_co": "0",
        "koi_fpflag_ec": "0",
        "koi_score": "0.9",
    }


def test_catalogue_labels_every_class():
    tces = [_tce(1, 1), _tce(1, 2), _tce(2, 1), _tce(3, 1)]
    kois = [
        _koi(1, 1, "CANDIDATE"),
        _koi(1, 2, "FALSE POSITIVE", nt=0),
        _koi(2, 1, "FALSE POSITIVE", nt=1),
    ]
    catalogue = kepler.build_catalogue(tces, kois)
    classes = {(t.kepid, t.tce_plnt_num): t.tce_class for t in catalogue}
    assert classes == {(1, 1): "PC", (1, 2): "AFP", (2, 1): "NTP", (3, 1): "NTP"}
    labels = {(t.kepid, t.tce_plnt_num): t.label for t in catalogue}
    assert labels[(1, 1)] == 1 and sum(labels.values()) == 1
    no_koi = next(t for t in catalogue if t.kepid == 3)
    assert no_koi.kepoi_name == "" and no_koi.robovetter_disposition == ""


def test_two_kois_on_one_tce_is_an_error():
    with pytest.raises(ValueError, match="two KOIs"):
        kepler.build_catalogue([_tce(1, 1)], [_koi(1, 1, "CANDIDATE"), _koi(1, 1, "CANDIDATE")])


def test_catalogue_round_trips_through_csv(tmp_path):
    catalogue = kepler.build_catalogue(
        [_tce(1, 1), _tce(2, 1)], [_koi(1, 1, "CANDIDATE")]
    )
    path = kepler.write_catalogue(catalogue, tmp_path / "labels.csv")
    # repr, because the missing columns are NaN and NaN != NaN.
    assert repr(kepler.read_catalogue(path)) == repr(catalogue)
    # No network when the file exists.
    assert repr(kepler.load_or_fetch_catalogue(path)) == repr(catalogue)


def test_stratified_sample_is_balanced_and_reproducible():
    catalogue = kepler.build_catalogue(
        [_tce(k, 1) for k in range(60)],
        [_koi(k, 1, "CANDIDATE") for k in range(10)]
        + [_koi(k, 1, "FALSE POSITIVE") for k in range(10, 15)],
    )
    sample = kepler.stratified_sample(catalogue, per_class=8, seed=3)
    assert kepler.class_counts(sample) == {"PC": 8, "AFP": 5, "NTP": 8}
    assert sample == kepler.stratified_sample(catalogue, per_class=8, seed=3)
    assert sample != kepler.stratified_sample(catalogue, per_class=8, seed=4)


# --------------------------------------------------------------------------
# Light curves
# --------------------------------------------------------------------------
def _llc_bytes(quarter=5, n=200):
    from astropy.io import fits

    time = 400.0 + CADENCE * np.arange(n)
    flux = np.full(n, 1000.0)
    flux[3] = np.nan
    quality = np.zeros(n, dtype=np.int32)
    quality[7] = 32  # desaturation: dropped
    quality[9] = 1 << 20  # a bit outside the mask: kept
    columns = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="PDCSAP_FLUX", format="E", array=flux),
        fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=np.full(n, 0.1)),
        fits.Column(name="SAP_QUALITY", format="J", array=quality),
    ]
    primary = fits.PrimaryHDU()
    primary.header["QUARTER"] = quarter
    hdul = fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)])
    import io

    buffer = io.BytesIO()
    hdul.writeto(buffer)
    return buffer.getvalue()


def test_read_llc_fits_drops_bad_cadences():
    data = kepler.read_llc_fits(_llc_bytes(quarter=5, n=200))
    assert data["time"].size == 198
    assert np.all(data["quarter"] == 5)
    assert np.all(np.isfinite(data["flux"]))


def test_long_cadence_uris_skips_short_cadence(monkeypatch):
    def fake_mast(request):
        if request["service"] == "Mast.Caom.Filtered":
            return [{"obsid": 1, "t_exptime": 1800}, {"obsid": 2, "t_exptime": 60}]
        assert request["params"]["obsid"] == "1"
        return [
            {"productFilename": "kplr000000001-2009166043257_llc.fits", "dataURI": "mast:a_llc"},
            {"productFilename": "kplr000000001-2009166043257_lpd-targ.fits.gz", "dataURI": "mast:tpf"},
            {"productFilename": "kplr000000001_lc_Q1.tar", "dataURI": "mast:tar"},
        ]

    monkeypatch.setattr(kepler, "_mast", fake_mast)
    assert kepler.long_cadence_uris(1) == ["mast:a_llc"]


def test_download_star_normalises_each_quarter(monkeypatch):
    monkeypatch.setattr(kepler, "long_cadence_uris", lambda kepid: ["q1", "q2"])
    blobs = {"q1": _llc_bytes(quarter=1), "q2": _llc_bytes(quarter=2)}

    def fake_http(url, params=None, **_):
        return blobs[params["uri"]]

    monkeypatch.setattr(kepler, "_http", fake_http)
    star = kepler.download_star(7)
    # Both fake quarters share timestamps: the duplicate cadences are kept once.
    assert star["time"].size == 198
    assert np.all(np.diff(star["time"]) > 0)
    assert np.allclose(star["flux"], 1.0)


def test_star_cache_downloads_once(monkeypatch, tmp_path):
    calls = []

    def fake_download(kepid):
        calls.append(kepid)
        time = np.arange(100) * CADENCE
        return {
            "time": time,
            "flux": np.ones(100),
            "flux_err": np.full(100, 1e-4),
            "quarter": np.full(100, 3, dtype=np.int16),
        }

    monkeypatch.setattr(kepler, "download_star", fake_download)
    first = kepler.load_or_download_star(42, tmp_path)
    second = kepler.load_or_download_star(42, tmp_path)
    assert calls == [42]
    assert first.target_id == "KIC 42" and second.n_cadences == 100

    monkeypatch.setattr(
        kepler,
        "download_star",
        lambda kepid: {k: np.empty(0) for k in ("time", "flux", "flux_err", "quarter")},
    )
    assert kepler.load_or_download_star(43, tmp_path) is None
    assert kepler.curve_path(tmp_path, 43).exists()  # the miss is remembered


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------
def make_star(
    *,
    period=7.3,
    epoch=3.1,
    duration=0.2,
    depth=1e-3,
    odd_depth=None,
    secondary=0.0,
    noise=1e-4,
    trend=0.0,
    days=180.0,
    seed=0,
):
    """Kepler-like curve with box transits, optional odd/even and secondary, and a trend."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, days, CADENCE)
    time = time[(time % 30.0) > 1.0]  # monthly downlink gaps
    flux = 1.0 + noise * rng.standard_normal(time.size)
    flux += trend * np.sin(2 * np.pi * time / 11.0)
    number = np.round((time - epoch) / period).astype(int)
    phase = time - epoch - number * period
    in_transit = np.abs(phase) < duration / 2
    depths = np.where(number % 2 == 1, odd_depth if odd_depth is not None else depth, depth)
    flux[in_transit] -= depths[in_transit]
    shifted = time - epoch - 0.5 * period
    flux[np.abs(shifted - np.round(shifted / period) * period) < duration / 2] -= secondary
    return time, flux


def views_for(time, flux, eph, config=None):
    config = config or ViewConfig()
    mask = in_transit_mask(time, [eph], config.mask_half_width)
    flat = detrend_masked(time, flux, mask, knot_spacing_for([eph], config))
    return make_views(time, flat, eph, config)


def test_planet_views_bottom_out_at_minus_one_and_odd_matches_even():
    eph = Ephemeris(7.3, 3.1, 0.2)
    views = views_for(*make_star(trend=5e-3), eph)
    assert views.local.shape == (201,) and views.global_view.shape == (2001,)
    assert views.local.min() == pytest.approx(-1.0)
    assert views.depth_scale == pytest.approx(1e-3, rel=0.1)  # depth survives the trend
    assert abs(views.odd.min() - views.even.min()) < 0.15
    assert views.secondary.min() > -0.2
    assert abs(np.median(views.local[:40])) < 0.1  # flat out of transit
    assert views.local_empty_fraction == 0.0


def test_binary_at_twice_the_period_shows_in_odd_even():
    eph = Ephemeris(7.3, 3.1, 0.2)
    views = views_for(*make_star(depth=2e-3, odd_depth=1e-3), eph)
    # Odd transits are half as deep as even ones, and the views keep that ratio.
    assert views.odd.min() / views.even.min() == pytest.approx(0.5, abs=0.1)


def test_secondary_eclipse_shows_in_secondary_view():
    eph = Ephemeris(7.3, 3.1, 0.2)
    views = views_for(*make_star(depth=2e-3, secondary=1e-3), eph)
    assert views.secondary.min() == pytest.approx(-0.5, abs=0.12)


def test_mask_keeps_a_long_transit_out_of_the_trend():
    eph = Ephemeris(40.0, 12.0, 0.7)
    time, flux = make_star(period=40.0, epoch=12.0, duration=0.7, depth=3e-3, trend=4e-3)
    views = views_for(time, flux, eph)
    assert views.depth_scale == pytest.approx(3e-3, rel=0.1)
    assert knot_spacing_for([eph], ViewConfig()) == pytest.approx(4.2)


def test_unusable_ephemeris_is_rejected():
    time, flux = make_star()
    with pytest.raises(ValueError):
        make_views(time, flux, Ephemeris(float("nan"), 0.0, 0.1))


# --------------------------------------------------------------------------
# The training set and the first model
# --------------------------------------------------------------------------
def synthetic_catalogue(n_per_class=30):
    """Planets (PC), binaries at twice the period (AFP), and sinusoids (NTP)."""
    tces, curves = [], {}
    for i in range(3 * n_per_class):
        kepid = 1000 + i
        cls = ("PC", "AFP", "NTP")[i % 3]
        period = 3.0 + (i % 7)
        eph = dict(period=period, epoch=1.5, duration=0.15)
        if cls == "PC":
            time, flux = make_star(**eph, depth=1e-3, seed=i, days=120)
        elif cls == "AFP":
            time, flux = make_star(**eph, depth=2e-3, odd_depth=0.6e-3, secondary=5e-4, seed=i, days=120)
        else:
            time, flux = make_star(**eph, depth=0.0, seed=i, days=120)
            flux += 1e-3 * np.cos(2 * np.pi * (time - 1.5) / period)
        curves[kepid] = LightCurve(f"KIC {kepid}", time, flux, np.full(time.size, 1e-4))
        tces.append(
            KeplerTCE(
                kepid=kepid,
                tce_plnt_num=1,
                period_days=period,
                epoch_bkjd=1.5,
                duration_hours=0.15 * 24,
                depth_ppm=1000.0,
                tce_class=cls,
                mes=10.0 + (i % 5),
            )
        )
    return tces, curves


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tces, curves = synthetic_catalogue()
    patch = pytest.MonkeyPatch()
    patch.setattr(kepler_dr25, "load_or_download_star", lambda kepid, cache: curves.get(kepid))
    try:
        missing = KeplerTCE(9999, 1, 5.0, 1.0, 3.0, 100.0, "NTP")
        ts = kepler_dr25.build_training_set(
            tces + [missing], tmp_path_factory.mktemp("cache"), catalogue=tces, n_jobs=1
        )
    finally:
        patch.undo()
    return ts


def test_training_set_has_every_view_and_records_failures(built, tmp_path):
    assert len(built) == 90
    assert built.failures == {"000009999-01": "no long-cadence light curve at MAST"}
    assert built.views["global"].shape == (90, 2001)
    assert built.views["local"].shape == (90, 201)
    assert built.views["local"].dtype == np.float32
    again = kepler_dr25.TrainingSet.load(built.save(tmp_path / "ts.npz"))
    assert np.array_equal(again.views["odd"], built.views["odd"])
    assert list(again.classes) == list(built.classes)
    assert again.failures == built.failures


def test_group_split_never_shares_a_star():
    kepids = np.repeat(np.arange(50), 3)
    train, test = kepler_dr25.group_split(kepids, 0.3, seed=1)
    assert not set(kepids[train]) & set(kepids[test])
    assert train.size + test.size == kepids.size


def test_threshold_and_catalogue_precision():
    y = np.array([1, 1, 1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.5])
    assert kepler_dr25.threshold_for_recall(y, s, 0.75) == 0.7
    counts = {"PC": 100, "AFP": 100, "NTP": 800}
    rates = {"PC": 0.9, "AFP": 0.1, "NTP": 0.01}
    assert kepler_dr25.catalogue_precision(rates, counts) == pytest.approx(90 / (90 + 10 + 8))


def test_model_learns_shape_on_synthetic_classes(built, tmp_path):
    counts = {"PC": 4034, "AFP": 3025, "NTP": 26973}
    result = kepler_dr25.evaluate_training_set(built, counts, seed=0)
    assert result.average_precision > result.chance_average_precision + 0.2
    report = kepler_dr25.format_report(result)
    assert "average precision, views model" in report and "Robovetter" in report
    json.dumps(result.to_dict())  # serialisable
    png = kepler_dr25.plot_pr(result, tmp_path / "pr.png")
    views_png = kepler_dr25.plot_examples(built, tmp_path / "views.png")
    assert png.stat().st_size > 0 and views_png.stat().st_size > 0


def test_cli_offline_on_a_cached_catalogue(monkeypatch, tmp_path):
    tces, curves = synthetic_catalogue(n_per_class=12)
    path = kepler.write_catalogue(tces, tmp_path / "labels.csv")
    monkeypatch.setattr(kepler_dr25, "load_or_download_star", lambda kepid, cache: curves.get(kepid))
    code = kepler_dr25.main(
        [
            "--catalogue", str(path),
            "--per-class", "12",
            "--n-jobs", "1",
            "--results-dir", str(tmp_path / "results"),
            "--figures-dir", str(tmp_path / "figures"),
        ]
    )
    assert code == 0
    assert (tmp_path / "results" / "report.txt").exists()
    assert json.loads((tmp_path / "results" / "metrics.json").read_text())["n_tces"] == 36
    assert (tmp_path / "figures" / "02_precision_recall.png").exists()
