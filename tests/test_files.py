"""Local light-curve files: CSV and the npz cache, read onto the common scale."""

from __future__ import annotations

import numpy as np
import pytest

from transitml.data.base import LightCurve
from transitml.data.files import read_csv_light_curve, read_light_curves
from transitml.data.injection import save_curves


def write_csv(path, time, flux, flux_err=None, header=("time", "flux", "flux_err")):
    columns = [time, flux] + ([flux_err] if flux_err is not None else [])
    header = header[: len(columns)]
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="")


def test_csv_is_normalised_sorted_and_cleaned(tmp_path):
    time = np.array([3.0, 1.0, 2.0, 2.0, 4.0, 5.0])
    flux = np.array([1000.0, 1010.0, 990.0, 990.0, np.nan, 1000.0])
    write_csv(tmp_path / "star.csv", time, flux, np.full(6, 5.0))
    lc = read_csv_light_curve(tmp_path / "star.csv")

    assert lc.target_id == "star"
    np.testing.assert_array_equal(lc.time, [1.0, 2.0, 3.0, 5.0])
    assert np.median(lc.flux) == pytest.approx(1.0)
    np.testing.assert_allclose(lc.flux_err, 5.0 / 1000.0)
    assert lc.label is None


def test_csv_without_errors_gets_the_point_to_point_scatter(tmp_path):
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 10.0, 0.02)
    flux = 1.0 + rng.normal(0.0, 1e-3, time.size)
    write_csv(tmp_path / "noerr.csv", time, flux, header=("Time", "FLUX"))
    lc = read_csv_light_curve(tmp_path / "noerr.csv")
    assert np.all(lc.flux_err == lc.flux_err[0])
    assert lc.flux_err[0] == pytest.approx(1e-3, rel=0.15)


def test_csv_without_a_flux_column_is_refused(tmp_path):
    (tmp_path / "bad.csv").write_text("t,f\n1,2\n")
    with pytest.raises(ValueError, match="'time' and 'flux'"):
        read_csv_light_curve(tmp_path / "bad.csv")


def test_npz_cache_round_trips_and_filters_by_target(tmp_path):
    curves = [
        LightCurve(f"TIC {i}", np.arange(5.0), np.ones(5), np.full(5, 1e-3), meta={"sector": i})
        for i in (1, 2)
    ]
    save_curves(curves, tmp_path / "cache.npz")
    assert [lc.target_id for lc in read_light_curves(tmp_path / "cache.npz")] == ["TIC 1", "TIC 2"]
    only = read_light_curves(tmp_path / "cache.npz", target_id="TIC 2")
    assert len(only) == 1 and only[0].meta["sector"] == 2
    with pytest.raises(ValueError, match="no curve"):
        read_light_curves(tmp_path / "cache.npz", target_id="TIC 3")


def test_unknown_extension_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        read_light_curves(tmp_path / "x.fits")
