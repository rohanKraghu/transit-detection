"""Transit Least Squares as a drop-in alternative to the BLS search.

TLS (Hippke & Heller 2019) fits a limb-darkened transit template instead of a
box, over a period grid and a set of durations derived from stellar density.
For small planets its ingress and egress model buys signal that a box throws
away, which is the reason to try it here.

:func:`run_tls` returns the same fields as :func:`transitml.features.run_bls`,
so every feature downstream is computed identically: the period, duration and
mid-transit time come from TLS, the periodogram used for ``bls_sde`` and
``power_contrast`` is the TLS spectrum, and depth, depth SNR and
log-likelihood are the box statistics at TLS's ephemeris (the same formulas
astropy's BLS uses), so the features keep their meaning.

Selected with ``BLSConfig(search="tls")`` or ``run_pipeline.py --search tls``.
Needs ``pip install transitleastsquares`` (2.0 or later); it is slower than
BLS, about half a second per light curve here against a tenth.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from astropy.timeseries import BoxLeastSquares
from numpy.typing import NDArray

from .config import BLSConfig
from .preprocess import FlattenedLightCurve


def box_statistics(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    duration: float,
    transit_time: float,
) -> dict[str, float]:
    """Depth, its error, SNR and log-likelihood of one box, as astropy's BLS defines them.

    The flux is median-subtracted and weighted by inverse variance; the depth
    is the weighted out-of-transit mean minus the in-transit mean, its error
    ``sqrt(1/sum_in(w) + 1/sum_out(w))`` and the log-likelihood
    ``0.5 * sum_in(w) * depth**2``.  NaN when the box holds no cadence.
    """
    y = flux - np.median(flux)
    ivar = 1.0 / flux_err**2
    phase = (time - transit_time + 0.5 * period) % period - 0.5 * period
    inside = np.abs(phase) < 0.5 * duration
    w_in, w_out = ivar[inside].sum(), ivar[~inside].sum()
    if w_in <= 0 or w_out <= 0:
        nan = float("nan")
        return {"depth": nan, "depth_err": nan, "depth_snr": nan, "log_likelihood": nan}
    depth = float((y[~inside] * ivar[~inside]).sum() / w_out - (y[inside] * ivar[inside]).sum() / w_in)
    err = float(np.sqrt(1.0 / w_in + 1.0 / w_out))
    return {
        "depth": depth,
        "depth_err": err,
        "depth_snr": depth / err,
        "log_likelihood": float(0.5 * w_in * depth**2),
    }


def run_tls(lc: FlattenedLightCurve, config: BLSConfig, p_max: float) -> dict[str, Any]:
    """Search ``lc`` with TLS between ``config.min_period_days`` and ``p_max``.

    Single-threaded on purpose: the pipeline already runs one light curve
    per core.  Raises ``RuntimeError`` if TLS finds no usable signal, so the
    caller can fall back rather than featurise NaNs.
    """
    try:
        from transitleastsquares import transitleastsquares
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "the TLS search needs transitleastsquares (pip install transitleastsquares)"
        ) from exc

    err = np.where(lc.flux_err > 0, lc.flux_err, lc.scatter)
    model = transitleastsquares(lc.time, lc.flux, err, verbose=False)
    try:
        with warnings.catch_warnings():
            # "period_grid defaults to R_star=1 and M_star=1" (expected for a
            # one-sector baseline) and divide-by-zero in TLS's per-transit
            # statistics when a transit falls in a gap: not actionable per
            # light curve, and thousands of lines over a full run.
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            result = model.power(
                period_min=config.min_period_days,
                period_max=p_max,
                use_threads=1,
                show_progress_bar=False,
                verbose=False,
            )
    except (ValueError, ZeroDivisionError, IndexError) as exc:
        raise RuntimeError(f"TLS failed: {exc}") from exc
    period, duration, epoch = float(result.period), float(result.duration), float(result.T0)
    periods = np.asarray(result.periods, dtype=float)
    power = np.asarray(result.power, dtype=float)
    if not (np.isfinite(period) and np.isfinite(duration) and np.isfinite(epoch)) or duration <= 0:
        raise RuntimeError("TLS returned no finite solution")
    stats = box_statistics(lc.time, lc.flux, err, period, duration, epoch)
    return {
        "bls": BoxLeastSquares(lc.time, lc.flux, err),
        "periods": periods,
        "power": power,
        "best_index": int(np.nanargmax(power)),
        "period": period,
        "duration": duration,
        "transit_time": epoch,
        **stats,
    }
