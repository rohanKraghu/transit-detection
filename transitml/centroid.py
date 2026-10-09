"""Centroid tests: is the dip on the target, or on a neighbour?

A background eclipsing binary a pixel or two from the target, diluted by the
target's light, makes a shallow dip in the aperture that looks like a planet
transit.  The light curve alone cannot tell the two apart.  The pixels can: the
light that disappears during the event disappears from where its source is.

Two tests are run on a :class:`~transitml.data.tpf.TargetPixelData` given a
transit ephemeris (period, epoch, duration), both following the Kepler and TESS
data-validation reports (Bryson et al. 2013; Twicken et al. 2018):

**Difference image.**  For each transit, the mean image of the cadences just
before and after it (out of transit) minus the mean in-transit image.  What
remains is the light that went missing, so its flux-weighted centroid is where
the eclipsed source sits.  The per-transit difference images are averaged and
the difference-image centroid is compared with

* the target's catalogue position, when the file carries one (as a TESS TPF
  does, through its WCS).  This is the reference the flag uses, because it is
  not biased by neighbours: the out-of-transit centroid of a crowded stamp is
  the light-weighted mean of every star in it, so even an on-target transit
  sits off it.
* the out-of-transit flux-weighted centroid, always reported, and the
  reference when no catalogue position is known.

**Centroid motion.**  The flux-weighted centroid of every cadence, in transit
against the local out-of-transit baseline.  It moves whenever there is any
other light in the window, even for an on-target transit, so it is reported
beside the shift an on-target transit would produce given the measured depth
and the out-of-transit centroid (``expected = (c_oot - target) * d / (1 - d)``
for depth ``d``).  It carries the same information as the difference image
and is not used for the flag.

Statistics
----------
Centroids are measured over the same window for every image: the pixels within
``window_radius`` pixels of the reference position.  A whole-stamp
flux-weighted centroid lets noise in far pixels, with their long lever arms,
dominate.

Uncertainties come from a bootstrap.  With at least ``min_transits_bootstrap``
usable transits, whole transits are resampled, which carries transit-to-transit
systematics (pointing, scattered light) into the error.  With fewer, cadences
are resampled within each transit and the result is labelled so; that error
ignores correlated noise and is optimistic.

The significance of an offset is its Mahalanobis distance under the bootstrap
covariance of the two pixel axes.  That covariance is itself estimated from a
handful of transits (or a few dozen cadences), so the distance is referred to
Hotelling's T-squared distribution, ``F(2, n - 2)``, not to a chi-square, and
then expressed as the Gaussian-equivalent number of sigma (a 3-sigma flag has
the 0.27% false-alarm rate of a 1-D 3-sigma event).  ``n`` is the number of
transits, or for the cadence bootstrap the Welch-Satterthwaite effective
sample size.  Without that correction a nominal 3-sigma offset was reached by
7% of on-target transits with 9 transits and 43% with 3, in the synthetic
scenes of :mod:`transitml.data.synthetic_tpf`; with it, 3 to 6% exceed
2 sigma (nominal 4.6%) and at most 1 in 150 reaches 3 sigma, for 1 to 9
transits.

The flux-weighted centroid of a source near the edge of the window is pulled
towards the window's centre, so the offset to a neighbour is underestimated
(a neighbour 1.8 pixels away is measured at 1.5 to 1.7): the direction and the
significance are what to read, the length is a lower bound.

An offset is flagged when the dip itself is detected in the difference image
(its summed flux at least ``min_difference_snr`` times its bootstrap error),
the offset is at least ``significance_sigma`` significant, and it is at least
``min_offset_pixels`` long.  The floor absorbs what the bootstrap does not
see: the flux-weighted centroid of a real, undersampled, asymmetric TESS PRF
is not exactly the catalogue position, and catalogue and WCS positions carry
their own errors.  It is half a pixel (10.5 arcsec) by default, set on real
TESS-SPOC pixel files of confirmed planets: their transits are on the target
by definition, yet about a fifth of those the difference image measures sit
a significant 0.1 to 0.5 pixel from it.  The confirmed planets came from
sectors 1 to 13, and the floor was frozen before it was applied to the
sector 14 to 26 benchmark (see the README).  On synthetic stamps, with a
circular PSF and exact positions, a floor of 0.1 pixel would do.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import ndtri_exp

from .data.tpf import TESS_PIXEL_SCALE_ARCSEC, TargetPixelData


@dataclass(frozen=True)
class CentroidConfig:
    """Every tunable number of the centroid tests."""

    #: In-transit cadences: ``|phase| < 0.5 * in_transit_fraction * duration``.
    in_transit_fraction: float = 1.0
    #: Gap between the transit edge and the out-of-transit windows, in durations.
    gap_durations: float = 0.25
    #: Width of each out-of-transit window (before and after), in durations.
    out_of_transit_durations: float = 1.5
    #: Centroids use the pixels within this many pixels of the reference.
    window_radius: float = 3.0
    #: A pixel finite in fewer than this fraction of the used cadences is dropped.
    min_pixel_coverage: float = 0.9
    n_bootstrap: int = 1000
    min_transits_bootstrap: int = 5
    significance_sigma: float = 3.0
    #: Half a pixel: on real TESS pixels, significant offsets shorter than this
    #: are more common for confirmed planets than for false positives.
    min_offset_pixels: float = 0.5
    min_difference_snr: float = 3.0
    seed: int = 0


def _gaussian_sigma_from_log_p(log_p: float) -> float:
    """The ``s`` with ``P(|z| > s) = exp(log_p)`` for a standard normal ``z``.

    Works in log space so a tiny p-value does not underflow to infinity.
    """
    if not np.isfinite(log_p):
        return float("nan") if np.isnan(log_p) else float("inf")
    if log_p >= 0.0:
        return 0.0
    return float(-ndtri_exp(log_p - math.log(2.0)))


def gaussian_sigma_2d(chi2: float) -> float:
    """Gaussian-equivalent significance of a chi-square with 2 degrees of freedom.

    ``P(chi2_2 > x) = exp(-x / 2)``; the result is the ``s`` with
    ``P(|z| > s) = exp(-x / 2)``.
    """
    if not np.isfinite(chi2):
        return float("nan")
    return _gaussian_sigma_from_log_p(-0.5 * max(chi2, 0.0))


def _significance(
    offset: NDArray[np.float64],
    samples: NDArray[np.float64],
    n_units: float | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """``(sigma, per-axis 1-sigma error)`` of a 2-vector under its bootstrap distribution.

    Bootstrap samples further than 5 robust sigma from the median on either
    axis are dropped first: when the summed difference flux in a resample
    comes close to zero its centroid diverges, and a handful of such samples
    would otherwise set the covariance.

    ``n_units`` is the number of independent units resampled when there are
    few of them (transits).  The bootstrap covariance is then itself estimated
    from ``n_units`` values, and treating it as exact overstates significance
    badly: with 9 transits a nominal 3-sigma offset is reached by 7% of
    on-target transits.  The covariance is scaled by ``n / (n - 1)`` (the
    bootstrap's small-sample bias) and the distance is referred to Hotelling's
    T-squared distribution, ``F(2, n - 2)``, instead of a chi-square.
    """
    nan2 = np.full(2, np.nan)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if samples.shape[0] < 10 or not np.all(np.isfinite(offset)):
        return float("nan"), nan2
    median = np.median(samples, axis=0)
    mad = 1.4826 * np.median(np.abs(samples - median), axis=0)
    keep = np.all(np.abs(samples - median) <= 5.0 * np.maximum(mad, 1e-12), axis=1)
    samples = samples[keep]
    if samples.shape[0] < 10:
        return float("nan"), nan2
    cov = np.cov(samples, rowvar=False)
    if n_units is not None:
        cov = cov * n_units / (n_units - 1.0)
    errors = np.sqrt(np.diag(cov))
    try:
        chi2 = float(offset @ np.linalg.solve(cov, offset))
    except np.linalg.LinAlgError:
        return float("nan"), errors
    if n_units is None:
        return gaussian_sigma_2d(chi2), errors
    f_stat = chi2 * (n_units - 2.0) / (2.0 * (n_units - 1.0))
    log_p = float(stats.f.logsf(f_stat, 2, n_units - 2))
    return _gaussian_sigma_from_log_p(log_p), errors


def _satterthwaite_units(counts: list[tuple[int, int]]) -> float:
    """Effective number of independent units behind a cadence-bootstrap covariance.

    Each transit contributes an out-of-transit and an in-transit mean of
    ``n_o`` and ``n_i`` cadences.  Assuming equal per-cadence variance, the
    Welch-Satterthwaite degrees of freedom of the summed variance
    ``sum(1/n_o + 1/n_i)`` are
    ``(sum(1/n_o + 1/n_i))**2 / sum(1/(n**2 (n - 1)))``; one more than that is
    the sample size that gives a covariance the same reliability.  Groups of
    one cadence carry no variance information and are left out of the
    denominator.
    """
    total = sum(1.0 / n_o + 1.0 / n_i for n_o, n_i in counts)
    spread = sum(1.0 / (n * n * (n - 1.0)) for pair in counts for n in pair if n > 1)
    if spread <= 0.0:
        return 3.0  # the smallest sample F(2, n - 2) accepts
    return max(total * total / spread + 1.0, 3.0)


@dataclass
class CentroidResult:
    """Outcome of :func:`centroid_test`.  Positions are ``(column, row)`` stamp pixels."""

    target_id: str
    status: str
    message: str
    significant: bool = False
    reference: str = ""
    uncertainty_method: str = ""
    n_transits: int = 0
    n_in_transit_cadences: int = 0
    n_out_of_transit_cadences: int = 0
    n_window_pixels: int = 0
    target_position: tuple[float, float] | None = None
    reference_position: tuple[float, float] | None = None
    out_of_transit_centroid: tuple[float, float] | None = None
    difference_centroid: tuple[float, float] | None = None
    offset_pixels: tuple[float, float] | None = None
    offset_error_pixels: tuple[float, float] | None = None
    offset_distance_pixels: float = float("nan")
    offset_arcsec: float = float("nan")
    offset_sigma: float = float("nan")
    offset_from_oot_pixels: tuple[float, float] | None = None
    offset_from_oot_sigma: float = float("nan")
    difference_depth: float = float("nan")
    difference_snr: float = float("nan")
    centroid_shift_pixels: tuple[float, float] | None = None
    centroid_shift_sigma: float = float("nan")
    expected_shift_on_target_pixels: tuple[float, float] | None = None
    shift_vs_on_target_sigma: float = float("nan")
    config: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    #: Images for the report (not in the JSON): out-of-transit mean and difference.
    out_of_transit_image: NDArray[np.float64] | None = field(default=None, repr=False)
    difference_image: NDArray[np.float64] | None = field(default=None, repr=False)
    aperture: NDArray[np.bool_] | None = field(default=None, repr=False)
    window: NDArray[np.bool_] | None = field(default=None, repr=False)

    @property
    def verdict(self) -> str:
        if self.status != "ok":
            return f"inconclusive ({self.message})"
        if self.significant:
            return (
                f"OFFSET: the dip is {self.offset_arcsec:.1f} arcsec from the "
                f"{self.reference.replace('_', ' ')} ({self.offset_sigma:.1f} sigma); "
                "likely a blended neighbour"
            )
        return "no significant offset: the dip is consistent with the target"

    def to_dict(self) -> dict[str, Any]:
        skip = {"out_of_transit_image", "difference_image", "aperture", "window"}
        out = {k: v for k, v in self.__dict__.items() if k not in skip}
        out["verdict"] = self.verdict
        out["pixel_scale_arcsec"] = TESS_PIXEL_SCALE_ARCSEC
        return out


def _windows(time, period, epoch, duration, config):
    """``(event number, in-transit mask, out-of-transit mask)`` per cadence."""
    number = np.round((time - epoch) / period)
    phase = np.abs(time - epoch - number * period)
    in_transit = phase < 0.5 * config.in_transit_fraction * duration
    inner = (0.5 + config.gap_durations) * duration
    outer = inner + config.out_of_transit_durations * duration
    out_of_transit = (phase >= inner) & (phase <= outer)
    return number.astype(np.int64), in_transit, out_of_transit


def _centroid(image, cols, rows, mask):
    total = float(np.sum(image[mask]))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(2, np.nan), total
    return np.array(
        [np.sum(image[mask] * cols[mask]), np.sum(image[mask] * rows[mask])]
    ) / total, total


def _as_tuple(v) -> tuple[float, float] | None:
    return None if v is None else (float(v[0]), float(v[1]))


def centroid_test(
    tpf: TargetPixelData,
    period: float,
    epoch: float,
    duration: float,
    config: CentroidConfig | None = None,
) -> CentroidResult:
    """Run the difference-image and centroid-motion tests for one ephemeris.

    Never raises on bad data: an empty selection, all-NaN pixels or an
    undetected dip return a result with ``status`` other than ``"ok"`` and
    ``significant`` false.
    """
    config = config or CentroidConfig()
    base = {
        "target_id": tpf.target_id,
        "target_position": tpf.target_position,
        "config": dict(config.__dict__),
        "meta": {
            "sector": tpf.meta.get("sector"),
            "column0": tpf.column0,
            "row0": tpf.row0,
            "stamp_shape": list(tpf.shape),
            "ephemeris": {"period": period, "epoch": epoch, "duration": duration},
        },
        "aperture": tpf.aperture,
    }

    def fail(status: str, message: str, **extra) -> CentroidResult:
        return CentroidResult(status=status, message=message, **base, **extra)

    if (
        not (np.isfinite(period) and np.isfinite(epoch) and np.isfinite(duration))
        or period <= 0
        or duration <= 0
    ):
        return fail(
            "bad_ephemeris", "period, epoch and duration must be finite and positive"
        )

    flux = tpf.flux
    usable_cadence = np.isfinite(tpf.time) & np.any(np.isfinite(flux), axis=(1, 2))
    number, in_t, out_t = _windows(tpf.time, period, epoch, duration, config)
    in_t &= usable_cadence
    out_t &= usable_cadence
    if not in_t.any():
        return fail("no_in_transit_cadences", "no cadence falls inside a transit")
    if not out_t.any():
        return fail(
            "no_out_of_transit_cadences", "no cadence in the out-of-transit windows"
        )

    # Pixels finite in nearly every used cadence; then cadences finite in all of them.
    used = in_t | out_t
    coverage = np.mean(np.isfinite(flux[used]), axis=0)
    pixels = coverage >= config.min_pixel_coverage
    if not pixels.any():
        return fail("no_valid_pixels", "no pixel is finite in enough cadences")
    complete = np.all(np.isfinite(flux[:, pixels]), axis=1)
    in_t &= complete
    out_t &= complete

    events = [
        k
        for k in np.unique(number[in_t])
        if in_t[number == k].any() and out_t[number == k].any()
    ]
    if not events:
        return fail(
            "no_in_transit_cadences",
            "no transit has both in-transit and out-of-transit cadences with finite pixels",
        )
    keep_in = in_t & np.isin(number, events)
    keep_out = out_t & np.isin(number, events)

    # Images for the report: per-transit means, then averaged over transits.
    oot_images = np.stack([flux[keep_out & (number == k)].mean(axis=0) for k in events])
    in_images = np.stack([flux[keep_in & (number == k)].mean(axis=0) for k in events])
    oot_image = np.where(pixels, oot_images.mean(axis=0), np.nan)
    diff_image = np.where(pixels, (oot_images - in_images).mean(axis=0), np.nan)

    cols, rows = tpf.pixel_grid()
    if tpf.target_position is not None:
        reference_name, reference = (
            "target_position",
            np.asarray(tpf.target_position, float),
        )
    else:
        seed_mask = pixels & tpf.aperture if (pixels & tpf.aperture).any() else pixels
        reference, _ = _centroid(oot_image, cols, rows, seed_mask)
        reference_name = "out_of_transit_centroid"
        if not np.all(np.isfinite(reference)):
            return fail("no_valid_pixels", "out-of-transit image has no positive flux")
    window = pixels & (
        (cols - reference[0]) ** 2 + (rows - reference[1]) ** 2
        <= config.window_radius**2
    )
    if not window.any():
        return fail("no_valid_pixels", "no valid pixel within the centroid window")
    base.update(
        window=window, out_of_transit_image=oot_image, difference_image=diff_image
    )

    # Per-cadence window sums: flux, flux * column, flux * row.  Every centroid
    # below is a ratio of (means of) these, so the bootstrap only resamples them.
    w = window
    sums = np.stack(
        [
            flux[:, w].sum(axis=1),
            (flux[:, w] * cols[w]).sum(axis=1),
            (flux[:, w] * rows[w]).sum(axis=1),
        ],
        axis=1,
    )
    sums = np.where(np.isfinite(sums), sums, 0.0)
    positive = sums[:, :1] > 0
    cadence_centroid = np.where(
        positive, sums[:, 1:] / np.where(positive, sums[:, :1], 1.0), np.nan
    )

    groups = [
        (
            np.flatnonzero(keep_out & (number == k)),
            np.flatnonzero(keep_in & (number == k)),
        )
        for k in events
    ]

    def statistics(terms: NDArray) -> NDArray:
        """Every tested quantity from per-transit terms, vectorised over leading axes.

        ``terms[..., k, :]`` is transit ``k``'s ``[out-of-transit mean sums (3),
        in-transit mean sums (3), out-of-transit mean centroid (2), in-transit
        mean centroid (2)]``.  Returns ``[..., diff_c(2), oot_c(2), diff_flux,
        oot_flux, shift(2)]``.
        """
        mean = terms.mean(axis=-2)
        oot_s, in_s = mean[..., 0:3], mean[..., 3:6]
        diff = oot_s - in_s
        shift = mean[..., 8:10] - mean[..., 6:8]
        with np.errstate(invalid="ignore", divide="ignore"):
            diff_c = np.where(diff[..., :1] > 0, diff[..., 1:] / diff[..., :1], np.nan)
            oot_c = np.where(
                oot_s[..., :1] > 0, oot_s[..., 1:] / oot_s[..., :1], np.nan
            )
        return np.concatenate(
            [diff_c, oot_c, diff[..., :1], oot_s[..., :1], shift], axis=-1
        )

    def transit_terms(o_idx: NDArray, i_idx: NDArray) -> NDArray:
        """Terms of one transit; index arrays may carry leading bootstrap axes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN centroids
            return np.concatenate(
                [
                    sums[o_idx].mean(axis=-2),
                    sums[i_idx].mean(axis=-2),
                    np.nanmean(cadence_centroid[o_idx], axis=-2),
                    np.nanmean(cadence_centroid[i_idx], axis=-2),
                ],
                axis=-1,
            )

    terms = np.stack([transit_terms(o, i) for o, i in groups])  # [K, 10]
    point = statistics(terms)

    rng = np.random.default_rng(config.seed)
    n_events = len(events)
    n_boot = config.n_bootstrap
    if n_events >= config.min_transits_bootstrap:
        method = f"bootstrap over {n_events} transits, Hotelling T-squared calibration"
        units = n_events
        draws = rng.integers(0, n_events, size=(n_boot, n_events))
        boot = statistics(terms[draws])
    else:
        method = (
            f"bootstrap over cadences within {n_events} transit(s), Hotelling "
            "T-squared calibration with Welch-Satterthwaite degrees of freedom; "
            "ignores correlated noise, so the error is optimistic"
        )
        units = _satterthwaite_units([(o.size, i.size) for o, i in groups])
        resampled = [
            transit_terms(
                o[rng.integers(0, o.size, size=(n_boot, o.size))],
                i[rng.integers(0, i.size, size=(n_boot, i.size))],
            )
            for o, i in groups
        ]
        boot = statistics(np.stack(resampled, axis=1))  # [B, K, 10] -> [B, 8]

    diff_c, oot_c = point[0:2], point[2:4]
    diff_flux, oot_flux, shift = point[4], point[5], point[6:8]
    flux_err = np.nanstd(boot[:, 4]) if np.isfinite(boot[:, 4]).sum() > 1 else np.nan
    if units is not None:
        flux_err *= math.sqrt(units / (units - 1.0))
    difference_snr = float(diff_flux / flux_err) if flux_err > 0 else float("nan")
    depth = float(diff_flux / oot_flux) if oot_flux > 0 else float("nan")

    offset = (
        diff_c - reference if reference_name == "target_position" else diff_c - oot_c
    )
    if reference_name == "target_position":
        offset_boot = boot[:, 0:2] - reference
    else:
        offset_boot = boot[:, 0:2] - boot[:, 2:4]
    offset_sigma, offset_err = _significance(offset, offset_boot, units)
    from_oot = diff_c - oot_c
    from_oot_sigma, _ = _significance(from_oot, boot[:, 0:2] - boot[:, 2:4], units)
    shift_sigma, _ = _significance(shift, boot[:, 6:8], units)

    expected, shift_vs_sigma = None, float("nan")
    if tpf.target_position is not None:
        target = np.asarray(tpf.target_position, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_depth = boot[:, 4] / boot[:, 5]
            expected = (oot_c - target) * depth / (1.0 - depth)
            boot_expected = (boot[:, 2:4] - target) * (boot_depth / (1.0 - boot_depth))[
                :, None
            ]
        shift_vs_sigma, _ = _significance(
            shift - expected, boot[:, 6:8] - boot_expected, units
        )

    distance = float(np.hypot(*offset))
    status, message = "ok", "difference image measured"
    if not np.isfinite(difference_snr) or difference_snr < config.min_difference_snr:
        status = "weak_difference_image"
        message = (
            f"the dip is not detected in the difference image (SNR {difference_snr:.1f} "
            f"< {config.min_difference_snr:g}), so its centroid is not meaningful"
        )
    significant = bool(
        status == "ok"
        and np.isfinite(offset_sigma)
        and offset_sigma >= config.significance_sigma
        and distance >= config.min_offset_pixels
    )
    return CentroidResult(
        status=status,
        message=message,
        significant=significant,
        reference=reference_name,
        uncertainty_method=method,
        n_transits=n_events,
        n_in_transit_cadences=int(keep_in.sum()),
        n_out_of_transit_cadences=int(keep_out.sum()),
        n_window_pixels=int(window.sum()),
        reference_position=_as_tuple(reference),
        out_of_transit_centroid=_as_tuple(oot_c),
        difference_centroid=_as_tuple(diff_c),
        offset_pixels=_as_tuple(offset),
        offset_error_pixels=_as_tuple(offset_err),
        offset_distance_pixels=distance,
        offset_arcsec=distance * TESS_PIXEL_SCALE_ARCSEC,
        offset_sigma=offset_sigma,
        offset_from_oot_pixels=_as_tuple(from_oot),
        offset_from_oot_sigma=from_oot_sigma,
        difference_depth=depth,
        difference_snr=difference_snr,
        centroid_shift_pixels=_as_tuple(shift),
        centroid_shift_sigma=shift_sigma,
        expected_shift_on_target_pixels=_as_tuple(expected),
        shift_vs_on_target_sigma=shift_vs_sigma,
        **base,
    )


def combine_sector_tests(
    tests: Sequence[Mapping[str, Any]], config: CentroidConfig | None = None
) -> dict[str, Any]:
    """One verdict from the centroid tests of one star in several sectors.

    Each sector's test runs on its own pixel file at the same ephemeris.  The
    stamps are not aligned with one another (the spacecraft turns between
    sectors, so a pixel axis points somewhere else on the sky), so the offset
    vectors are not averaged; what carries over between sectors is how long
    the offset is and how significant:

    * the significance is Stouffer's combination of the sectors' p-values:
      each turned into a one-sided normal deviate, summed and divided by the
      square root of their number.  An offset seen in every sector adds up,
      noise in each stays noise, and one wild sector among many quiet ones
      does not decide it (Fisher's ``-2 sum ln p`` lets the smallest p-value
      do just that);
    * the length is the mean of the sectors' lengths, each weighted by the
      inverse of its bootstrap variance (the mean of its two axes');
    * the difference-image SNR is the sectors' SNRs added in quadrature.

    ``tests`` are dicts of :class:`CentroidResult` fields (at least
    ``status``, ``offset_sigma``, ``offset_distance_pixels``,
    ``offset_error_pixels``, ``difference_snr`` and ``n_transits``).  A
    sector enters the offset when its difference image detects the dip
    (status ``"ok"``), its significance and error are measured, and its
    centroid lies within the ``window_radius`` it was measured over: a
    flux-weighted centroid of that window can only land outside it when the
    difference image there is not a dip but noise of both signs, and such a
    sector would otherwise flag the star from a centroid 10 or 20 pixels away.
    Every sector with an SNR enters the SNR.  The star's dip is placed
    (status ``"ok"``) when any sector placed it, and flagged by the rule for
    one sector: at least ``significance_sigma`` and at least
    ``min_offset_pixels``.  One sector placed inside its window gives back
    its own numbers.
    """
    config = config or CentroidConfig()
    measured = [
        t
        for t in tests
        if t["status"] == "ok"
        and not np.isnan(t["offset_sigma"])
        and t["offset_error_pixels"] is not None
        and np.all(np.isfinite(t["offset_error_pixels"]))
    ]
    placed = [t for t in measured if t["offset_distance_pixels"] <= config.window_radius]
    snrs = np.array([t["difference_snr"] for t in tests], dtype=float)
    snrs = snrs[np.isfinite(snrs)]
    snr = float(np.sqrt(np.sum(snrs**2))) if snrs.size else float("nan")
    sigma = distance = float("nan")
    if placed:
        # Each sector's P(|z| > s), as a one-sided normal deviate; a p-value of
        # exactly one (an offset of exactly zero) is kept finite.
        log_p = np.array(
            [math.log(2.0) + float(stats.norm.logcdf(-t["offset_sigma"])) for t in placed]
        )
        with np.errstate(invalid="ignore"):
            deviates = -ndtri_exp(np.minimum(log_p, math.log1p(-1e-12)))
        deviates[np.isneginf(log_p)] = np.inf
        total = float(np.sum(deviates) / math.sqrt(len(placed)))
        sigma = _gaussian_sigma_from_log_p(float(stats.norm.logsf(total)))
        weights = np.array([2.0 / np.sum(np.square(t["offset_error_pixels"])) for t in placed])
        lengths = np.array([t["offset_distance_pixels"] for t in placed], dtype=float)
        distance = float(np.sum(weights * lengths) / np.sum(weights))
        status = "ok"
    elif measured:
        status = "centroid_outside_window"
    else:
        statuses = [str(t["status"]) for t in tests]
        status = (
            "weak_difference_image"
            if "weak_difference_image" in statuses
            else statuses[0] if statuses else "no_pixel_file"
        )
    significant = bool(
        placed and sigma >= config.significance_sigma and distance >= config.min_offset_pixels
    )
    return {
        "status": status,
        "significant": significant,
        "offset_distance_pixels": distance,
        "offset_arcsec": distance * TESS_PIXEL_SCALE_ARCSEC,
        "offset_sigma": sigma,
        "difference_snr": snr,
        "n_transits": int(sum(t["n_transits"] for t in tests)),
        "n_sectors": len(tests),
        "n_sectors_placed": len(placed),
    }


def combined_pixel_files(tests: Sequence[Mapping[str, Any] | None]) -> int | None:
    """How many pixel files :func:`combine_sector_tests` combined into ``tests``.

    ``None`` when none of ``tests`` was combined across sectors.
    """
    counts = [t["n_sectors"] for t in tests if t is not None and "n_sectors" in t]
    return int(sum(counts)) if counts else None

