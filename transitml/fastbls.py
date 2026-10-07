"""Box Least Squares as whole-array operations, on the CPU or a GPU.

astropy's BLS loops over trial periods in C, one period at a time.  This
module computes the same periodogram for a block of periods at once: every
cadence is phase-folded at every period in the block, binned with one
``bincount``, and every box of every duration is evaluated from cumulative
sums.  Nothing in it is specific to NumPy, so passing ``xp=cupy`` runs the
identical code on a GPU.

It reproduces astropy's ``method="fast"`` algorithm step for step (same bin
width, same wrap-around padding, same objective and the same first-maximum
tie-breaking), so the periodogram and the best box agree with astropy to
rounding error and the features, the trained model and the headline numbers
do not depend on which engine ran.  ``tests/test_fastbls.py`` checks that on
synthetic planets, binaries and noise.

The CuPy path has not been run on a GPU: this repository's environment has
none.  It uses only functions CuPy implements with NumPy's semantics, and
``python -m transitml.fastbls --engine gpu`` times it against astropy and
checks the two agree, so the first run on a GPU says whether it works.

On a CPU the array version is much slower than astropy's compiled loop (it
materialises every box of every period instead of streaming them), so
astropy stays the default; the array form only pays where thousands of GPU
threads evaluate the boxes at once.
"""

from __future__ import annotations

import argparse
import time
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Cadence-period products evaluated per block.  Bounds memory at a few tens
#: of megabytes per array on the CPU; a GPU can take far larger blocks.
DEFAULT_BLOCK_ELEMENTS = 4_000_000


def array_module(name: str) -> ModuleType:
    """``numpy`` for ``"cpu"``, ``cupy`` for ``"gpu"`` (which must be installed)."""
    if name == "cpu":
        return np
    if name == "gpu":
        try:
            import cupy
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "the GPU BLS engine needs CuPy and a CUDA device "
                "(pip install cupy-cuda12x); use the cpu or astropy engine instead"
            ) from exc
        return cupy
    raise ValueError(f"unknown array engine {name!r}; expected 'cpu' or 'gpu'")


def bls_power(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None,
    periods: NDArray[np.float64],
    durations: NDArray[np.float64],
    *,
    oversample: int = 10,
    xp: ModuleType = np,
    block_elements: int = DEFAULT_BLOCK_ELEMENTS,
) -> dict[str, NDArray[np.float64]]:
    """The BLS periodogram with the depth-SNR objective, as astropy computes it.

    Returns NumPy arrays, one value per period: ``power`` (the depth SNR of
    the best box), ``depth``, ``depth_err``, ``duration``, ``transit_time``,
    ``depth_snr`` and ``log_likelihood``, the same fields and meaning as
    :meth:`astropy.timeseries.BoxLeastSquares.power` with ``objective="snr"``.
    A period with no valid box has ``power = -inf`` and NaN elsewhere.
    """
    periods = np.asarray(periods, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if periods.size == 0 or durations.size == 0:
        raise ValueError("need at least one period and one duration")
    if durations.max() > periods.min() or durations.min() <= 0:
        raise ValueError("every duration must be positive and no longer than the shortest period")
    if oversample < 1:
        raise ValueError("oversample must be at least 1")

    t = np.asarray(time, dtype=np.float64)
    t_ref = float(t.min())
    t = xp.asarray(t - t_ref)
    y = np.asarray(flux, dtype=np.float64)
    y = xp.asarray(y - np.median(y))
    if flux_err is None:
        ivar = xp.ones_like(y)
    else:
        ivar = 1.0 / xp.asarray(np.asarray(flux_err, dtype=np.float64)) ** 2
    y_ivar = y * ivar
    sum_y = float(y_ivar.sum())
    sum_ivar = float(ivar.sum())

    bin_duration = float(durations.min()) / oversample
    dur_bins = np.round(durations / bin_duration).astype(np.int64)

    out = {
        key: np.full(periods.size, np.nan)
        for key in ("depth", "depth_err", "duration", "transit_time", "depth_snr",
                    "log_likelihood")
    }
    out["power"] = np.full(periods.size, -np.inf)

    per_block = max(1, block_elements // max(t.size, 1))
    for start in range(0, periods.size, per_block):
        stop = min(start + per_block, periods.size)
        block = _block(
            xp, t, y_ivar, ivar, sum_y, sum_ivar, periods[start:stop],
            dur_bins, bin_duration, oversample,
        )
        for key, values in block.items():
            out[key][start:stop] = values
    out["transit_time"] = out["transit_time"] + t_ref
    return out


def _block(
    xp: ModuleType,
    t: Any,
    y_ivar: Any,
    ivar: Any,
    sum_y: float,
    sum_ivar: float,
    periods: NDArray[np.float64],
    dur_bins: NDArray[np.int64],
    bin_duration: float,
    oversample: int,
) -> dict[str, NDArray[np.float64]]:
    """One block of periods; see :func:`bls_power`.  Mirrors astropy's ``bls.c``."""
    m = periods.size
    p = xp.asarray(periods)
    n_bins = np.ceil(periods / bin_duration).astype(np.int64) + oversample
    width = int(n_bins.max()) + 1  # row length: indices 0 .. n_bins

    # Fold and bin: index 1 + floor(phase time / bin), as bls.c does.
    phase_time = t[None, :] - p[:, None] * xp.floor(t[None, :] / p[:, None])
    index = (phase_time / bin_duration).astype(xp.int64) + 1
    flat = (xp.arange(m, dtype=xp.int64)[:, None] * width + index).ravel()
    total = m * width
    mean_y = xp.bincount(flat, weights=xp.broadcast_to(y_ivar, (m, t.size)).ravel(),
                         minlength=total).reshape(m, width)
    mean_ivar = xp.bincount(flat, weights=xp.broadcast_to(ivar, (m, t.size)).ravel(),
                            minlength=total).reshape(m, width)

    # Wrap around: bins n_bins - oversample .. n_bins - 1 are overwritten by
    # bins 1 .. oversample (bls.c overwrites rather than adds, and so do we).
    rows = xp.arange(m)[:, None]
    src = xp.arange(1, oversample + 1)[None, :]
    dst = xp.asarray(n_bins - oversample)[:, None] + xp.arange(oversample)[None, :]
    mean_y[rows, dst] = mean_y[rows, src]
    mean_ivar[rows, dst] = mean_ivar[rows, src]
    # Entries past a row's own n_bins stay zero, as in bls.c's work array.
    beyond = xp.arange(width)[None, :] > xp.asarray(n_bins)[:, None]
    mean_y[beyond] = 0.0
    mean_ivar[beyond] = 0.0
    cum_y = xp.cumsum(mean_y, axis=1)
    cum_ivar = xp.cumsum(mean_ivar, axis=1)

    best = xp.full(m, -xp.inf)
    best_n = xp.zeros(m, dtype=xp.int64)
    best_k = xp.zeros(m, dtype=xp.int64)
    best_y_in = xp.zeros(m)
    best_ivar_in = xp.zeros(m)
    n_bins_x = xp.asarray(n_bins)
    eps = np.finfo(np.float64).eps
    for k, dur in enumerate(dur_bins):
        dur = int(dur)
        if dur >= width:
            continue
        y_in = cum_y[:, dur:] - cum_y[:, :-dur] if dur > 0 else xp.zeros_like(cum_y)
        ivar_in = cum_ivar[:, dur:] - cum_ivar[:, :-dur] if dur > 0 else xp.zeros_like(cum_ivar)
        y_out = sum_y - y_in
        ivar_out = sum_ivar - ivar_in
        valid = (ivar_in >= eps) & (ivar_out >= eps)
        valid &= xp.arange(y_in.shape[1])[None, :] <= (n_bins_x - dur)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_in = y_in / ivar_in
            mean_out = y_out / ivar_out
            snr = (mean_out - mean_in) / xp.sqrt(1.0 / ivar_in + 1.0 / ivar_out)
        snr = xp.where(valid & (mean_out >= mean_in), snr, -xp.inf)
        n = xp.argmax(snr, axis=1)  # first maximum, as bls.c's strict ">" keeps
        value = xp.take_along_axis(snr, n[:, None], axis=1)[:, 0]
        better = value > best
        best = xp.where(better, value, best)
        best_n = xp.where(better, n, best_n)
        best_k = xp.where(better, k, best_k)
        best_y_in = xp.where(better, xp.take_along_axis(y_in, n[:, None], axis=1)[:, 0], best_y_in)
        best_ivar_in = xp.where(
            better, xp.take_along_axis(ivar_in, n[:, None], axis=1)[:, 0], best_ivar_in
        )

    found = best > -xp.inf
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_in = best_y_in / best_ivar_in
        mean_out = (sum_y - best_y_in) / (sum_ivar - best_ivar_in)
        depth = mean_out - mean_in
        depth_err = xp.sqrt(1.0 / best_ivar_in + 1.0 / (sum_ivar - best_ivar_in))
    duration = xp.asarray(dur_bins)[best_k] * bin_duration
    transit_time = xp.fmod(best_n * bin_duration + 0.5 * duration, p)
    result = {
        "power": best,
        "depth": xp.where(found, depth, xp.nan),
        "depth_err": xp.where(found, depth_err, xp.nan),
        "duration": xp.where(found, duration, xp.nan),
        "transit_time": xp.where(found, transit_time, xp.nan),
        "depth_snr": xp.where(found, depth / depth_err, xp.nan),
        "log_likelihood": xp.where(found, 0.5 * best_ivar_in * depth**2, xp.nan),
    }
    return {key: _to_numpy(value) for key, value in result.items()}


def _to_numpy(values: Any) -> NDArray[np.float64]:
    get = getattr(values, "get", None)  # CuPy arrays copy back with .get()
    return np.asarray(get() if callable(get) else values, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    """Time an engine against astropy on synthetic light curves and check they agree."""
    from astropy.timeseries import BoxLeastSquares

    from .config import default_config
    from .data.synthetic import SyntheticTESSSource
    from .features import period_grid
    from .preprocess import flatten

    parser = argparse.ArgumentParser(
        prog="python -m transitml.fastbls",
        description="Time the array BLS against astropy and check the periodograms agree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--engine", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--n-curves", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    xp = array_module(args.engine)
    config = default_config()
    source = SyntheticTESSSource(
        args.n_curves, 0.3, 0.2, seed=args.seed, survey=config.survey, noise=config.noise,
        star=config.star, planet=config.planet, eb=config.eb,
    )
    t_astropy = t_engine = 0.0
    worst = 0.0
    same_peak = 0
    for i in range(args.n_curves):
        flat = flatten(source.generate(i), config.preprocess)
        err = np.where(flat.flux_err > 0, flat.flux_err, flat.scatter)
        periods = period_grid(flat.baseline_days, config.bls)
        durations = np.asarray(config.bls.durations_days)
        durations = durations[durations < 0.5 * periods.min()]
        start = time.perf_counter()
        ref = BoxLeastSquares(flat.time, flat.flux, err).power(periods, durations, objective="snr")
        t_astropy += time.perf_counter() - start
        start = time.perf_counter()
        ours = bls_power(flat.time, flat.flux, err, periods, durations, xp=xp)
        t_engine += time.perf_counter() - start
        reference = np.asarray(ref.power)
        worst = max(worst, float(np.max(np.abs(ours["power"] - reference) / np.abs(reference))))
        same_peak += int(np.argmax(ours["power"]) == np.argmax(reference))
    n = args.n_curves
    print(f"{n} light curves, {config.bls.n_periods} periods each")
    print(f"  astropy:      {1e3 * t_astropy / n:8.1f} ms per curve")
    print(f"  {args.engine} engine:   {1e3 * t_engine / n:8.1f} ms per curve")
    print(f"  largest relative power difference: {worst:.1e}")
    print(f"  same best period on {same_peak} of {n}")
    return 0 if same_peak == n and worst < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
