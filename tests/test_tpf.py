"""Target pixel data: validation, npz round trip, lightkurve conversion (offline)."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from transitml.data import tpf as tpf_module
from transitml.data.tpf import (
    TargetPixelData,
    download_tpfs,
    from_lightkurve,
    load_tpf,
    save_tpf,
)


def small_tpf(n_t: int = 50, with_err: bool = True, **kwargs) -> TargetPixelData:
    rng = np.random.default_rng(3)
    time = np.arange(n_t) * 0.02
    flux = 100.0 + rng.normal(0.0, 1.0, size=(n_t, 5, 6))
    aperture = np.zeros((5, 6), bool)
    aperture[1:4, 2:5] = True
    return TargetPixelData(
        target_id="TIC 1",
        time=time,
        flux=flux,
        aperture=aperture,
        flux_err=np.ones_like(flux) if with_err else None,
        column0=100,
        row0=200,
        target_position=(3.2, 2.1),
        meta={"sector": 14},
        **kwargs,
    )


def test_shapes_are_validated():
    tpf = small_tpf()
    assert tpf.shape == (5, 6) and tpf.n_cadences == 50
    with pytest.raises(ValueError, match="aperture"):
        TargetPixelData("x", tpf.time, tpf.flux, np.ones((6, 5), bool))
    with pytest.raises(ValueError, match="flux frames"):
        TargetPixelData("x", tpf.time[:-1], tpf.flux, tpf.aperture)
    with pytest.raises(ValueError, match=r"\[time, row, column\]"):
        TargetPixelData("x", tpf.time, tpf.flux[:, 0, :], tpf.aperture)
    with pytest.raises(ValueError, match="strictly increasing"):
        TargetPixelData("x", tpf.time[::-1], tpf.flux, tpf.aperture)


def test_pixel_grid_is_column_row():
    cols, rows = small_tpf().pixel_grid()
    assert cols.shape == rows.shape == (5, 6)
    assert cols[0, 5] == 5 and rows[4, 0] == 4


def test_npz_round_trip_without_pickle(tmp_path):
    tpf = small_tpf()
    path = save_tpf(tpf, tmp_path / "t.npz")
    back = load_tpf(path)
    assert back.target_id == "TIC 1"
    np.testing.assert_array_equal(back.time, tpf.time)
    np.testing.assert_array_equal(back.flux, tpf.flux)
    np.testing.assert_array_equal(back.flux_err, tpf.flux_err)
    np.testing.assert_array_equal(back.aperture, tpf.aperture)
    assert (back.column0, back.row0) == (100, 200)
    assert back.target_position == (3.2, 2.1)
    assert back.meta == {"sector": 14}
    with zipfile.ZipFile(path) as archive:  # every member is a plain array
        for name in archive.namelist():
            with archive.open(name) as member:
                np.lib.format.read_array(member, allow_pickle=False)


def test_npz_round_trip_without_errors_or_position(tmp_path):
    tpf = small_tpf(with_err=False)
    tpf.target_position = None
    back = load_tpf(save_tpf(tpf, tmp_path / "t.npz"))
    assert back.flux_err is None and back.target_position is None


def test_load_refuses_a_light_curve_cache(tmp_path):
    from transitml.data.base import LightCurve
    from transitml.data.injection import save_curves

    t = np.arange(10.0)
    save_curves([LightCurve("a", t, np.ones(10), np.ones(10))], tmp_path / "lc.npz")
    with pytest.raises(ValueError, match="not a target pixel file"):
        load_tpf(tmp_path / "lc.npz")


def test_aperture_light_curve_drops_cadences_with_nan_pixels():
    tpf = small_tpf()
    tpf.flux[7, 2, 3] = np.nan  # inside the aperture
    tpf.flux[8, 0, 0] = np.nan  # outside: harmless
    lc = tpf.to_light_curve()
    assert lc.n_cadences == 49
    assert np.median(lc.flux) == pytest.approx(1.0)
    assert np.all(lc.flux_err > 0)


def test_aperture_light_curve_without_errors_uses_point_to_point_scatter():
    lc = small_tpf(with_err=False).to_light_curve()
    assert np.all(np.isfinite(lc.flux_err)) and np.ptp(lc.flux_err) == 0.0


# --------------------------------------------------------------------------
# lightkurve, replaced by stand-ins: only the conversion is tested here
# --------------------------------------------------------------------------
class _Quantity:
    def __init__(self, value):
        self.value = value


class _FakeWCS:
    def all_world2pix(self, coords, origin):
        ra, dec = coords[0]
        return np.array([[ra - 10.0, dec - 20.0]])


class FakeLightkurveTPF:
    """The attributes of ``lightkurve.TargetPixelFile`` that the conversion reads."""

    def __init__(self, n_t: int = 30, mask: bool = True):
        rng = np.random.default_rng(1)
        time = np.arange(n_t) * 0.01 + 1600.0
        time[5] = np.nan  # a cadence without a timestamp
        self.time = _Quantity(time)
        self.flux = _Quantity(rng.normal(50.0, 1.0, size=(n_t, 4, 4)))
        self.flux_err = _Quantity(np.ones((n_t, 4, 4)))
        self._mask = mask
        self.column, self.row = 512, 1024
        self.ra, self.dec = 11.5, 21.25
        self.wcs = _FakeWCS()
        self.targetid = 99
        self.meta = {"SECTOR": 14, "CAMERA": 1, "CCD": 2, "TESSMAG": np.float64(9.5)}

    @property
    def pipeline_mask(self):
        if not self._mask:
            raise AttributeError("no aperture extension")
        mask = np.zeros((4, 4), bool)
        mask[1:3, 1:3] = True
        return mask

    def create_threshold_mask(self):
        mask = np.zeros((4, 4), bool)
        mask[0, 0] = True
        return mask


def test_conversion_from_a_lightkurve_tpf():
    tpf = from_lightkurve(FakeLightkurveTPF(), "TIC 99")
    assert tpf.target_id == "TIC 99"
    assert tpf.n_cadences == 29  # the NaN timestamp is dropped
    assert tpf.flux.shape == (29, 4, 4) and tpf.flux_err.shape == (29, 4, 4)
    assert tpf.aperture.sum() == 4
    assert (tpf.column0, tpf.row0) == (512, 1024)
    assert tpf.target_position == (1.5, 1.25)
    assert tpf.meta["sector"] == 14 and tpf.meta["tess_mag"] == 9.5


def test_conversion_falls_back_to_a_threshold_aperture():
    tpf = from_lightkurve(FakeLightkurveTPF(mask=False))
    assert tpf.aperture.sum() == 1 and tpf.target_id == "99"


def test_download_goes_through_lightkurve_search(monkeypatch):
    calls = {}

    class FakeSearch:
        def __len__(self):
            return 1

        def download_all(self, quality_bitmask):
            calls["quality_bitmask"] = quality_bitmask
            return [FakeLightkurveTPF()]

    class FakeLK:
        @staticmethod
        def search_targetpixelfile(target, **kwargs):
            calls["target"], calls["kwargs"] = target, kwargs
            return FakeSearch()

    monkeypatch.setattr(tpf_module, "_import_lightkurve", lambda: FakeLK)
    tpfs = download_tpfs("TIC 99", author="TESS-SPOC", exposure_time=1800, sector=14)
    assert len(tpfs) == 1 and tpfs[0].target_id == "TIC 99"
    assert calls["target"] == "TIC 99"
    assert calls["kwargs"] == {
        "mission": "TESS",
        "author": "TESS-SPOC",
        "exptime": 1800,
        "sector": 14,
    }
    assert calls["quality_bitmask"] == "default"


def test_download_failure_is_a_warning_and_an_empty_list(monkeypatch):
    class BrokenLK:
        @staticmethod
        def search_targetpixelfile(target, **kwargs):
            raise ConnectionError("no route to MAST")

    monkeypatch.setattr(tpf_module, "_import_lightkurve", lambda: BrokenLK)
    with pytest.warns(UserWarning, match="no route to MAST"):
        assert download_tpfs("TIC 99") == []
