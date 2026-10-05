"""Astrophysically-motivated synthetic TESS-like light curves.

The generator builds each light curve additively, in the same order the real
signals appear in a real photometric time series::

    flux = 1
         + stellar variability (rotation / pulsation, sinusoidal harmonics)
         + red noise            (1/f^alpha: jitter, thermal drift, background)
         + instrumental ramps   (momentum dumps)
         + flares               (upward outliers -- these break naive sigma clipping)
         - transit / eclipse    (trapezoidal, from real geometry)
         + white noise          (shot + read, from a magnitude-scatter relation)

Three populations are produced:

``planet``
    A genuine transiting planet.  Label 1.  Depth follows (Rp/Rs)^2, duration
    follows from Kepler's third law and the impact parameter, so depth and
    duration are *correlated the way real transits are* -- a classifier cannot
    cheat by learning an unphysical depth/duration combination.

``eclipsing_binary``
    An eclipsing binary.  Label 0.  These are the astrophysical false positives
    that dominate real transit-search candidate lists.  They are deeper on
    average, often V-shaped (grazing), and may show a secondary eclipse or an
    odd/even depth difference when the search locks onto half the true period.

``noise``
    A variable star with no eclipsing companion.  Label 0.  The bulk of the set.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

import numpy as np
from numpy.typing import NDArray

from ..config import (
    EclipsingBinaryConfig,
    NoiseConfig,
    PlanetConfig,
    StarConfig,
    SurveyConfig,
)
from ..physics import RHO_SUN_CGS, scaled_semi_major_axis, transit_durations
from .base import LightCurve, LightCurveSource

CurveKind = Literal["planet", "eclipsing_binary", "noise"]


# --------------------------------------------------------------------------
# Component builders.  Each is a pure function of (time, params, rng).
# --------------------------------------------------------------------------
def sample_time_grid(survey: SurveyConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Build an irregular TESS-like time axis.

    A regular cadence grid, minus a mid-sector downlink gap, minus randomly
    scattered dropped cadences.  Real light curves are never gap-free and the
    detrending has to cope with that.
    """
    n = int(round(survey.baseline_days / survey.cadence_days))
    time = np.arange(n, dtype=np.float64) * survey.cadence_days

    # Mid-sector data downlink, jittered so every curve has a different gap.
    gap_centre = survey.baseline_days * rng.uniform(0.45, 0.55)
    half = survey.downlink_gap_days / 2.0
    keep = np.abs(time - gap_centre) > half

    # Scattered losses: scattered light near perigee, momentum dumps, cosmic rays.
    keep &= rng.random(n) > survey.random_dropout_fraction
    return time[keep]


def white_noise_sigma(tess_mag: float, noise: NoiseConfig) -> float:
    """Per-cadence fractional scatter from an empirical magnitude relation.

    Roughly 60 ppm at T = 8 rising to ~1900 ppm at T = 14 for 30-minute bins,
    which brackets the TESS photometric-precision curve well enough for our
    purposes.
    """
    bright = noise.tess_mag_range[0]
    return noise.white_ppm_at_bright_end * 1e-6 * 10 ** (
        noise.white_mag_slope * (tess_mag - bright)
    )


def power_law_noise(
    n: int, dt: float, alpha: float, rms: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Generate zero-mean noise with PSD ~ f**-alpha and the requested RMS.

    Built in the Fourier domain: assign amplitude f**(-alpha/2) and uniform
    random phase to every positive frequency, then inverse-FFT.  ``alpha = 0``
    gives white noise, ``alpha = 1`` flicker noise, ``alpha = 2`` a random walk.
    """
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.zeros_like(freqs)
    amp[1:] = freqs[1:] ** (-alpha / 2.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
    spectrum = amp * np.exp(1j * phases)
    spectrum[0] = 0.0  # no DC term: the curve is median-normalised later
    series = np.fft.irfft(spectrum, n=n)
    sd = float(np.std(series))
    return series * (rms / sd) if sd > 0 else series


def stellar_variability(
    time: NDArray[np.float64], star: StarConfig, rng: np.random.Generator
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Rotational modulation / pulsation as a sum of harmonics with random phases.

    This is the signal the detrender has to remove.  Its amplitude is typically
    10-100x the transit depth, which is exactly why detrending is the crux of
    the problem rather than an afterthought.
    """
    amp = float(
        10 ** rng.uniform(*np.log10(star.variability_amplitude_range))
    )
    period = float(10 ** rng.uniform(*np.log10(star.variability_period_range_days)))
    signal = np.zeros_like(time)
    for k in range(1, star.n_variability_harmonics + 1):
        # Harmonic amplitudes fall off; spot patterns are not pure sinusoids.
        harmonic_amp = amp * rng.uniform(0.15, 1.0) / k
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += harmonic_amp * np.sin(2.0 * np.pi * k * time / period + phase)
    return signal, {"variability_amplitude": amp, "variability_period": period}


def instrumental_ramps(
    time: NDArray[np.float64], sigma_white: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Exponential settling ramps after spacecraft momentum dumps."""
    signal = np.zeros_like(time)
    span = time[-1] - time[0]
    for _ in range(rng.integers(1, 5)):
        t0 = time[0] + rng.uniform(0.0, span)
        tau = rng.uniform(0.05, 0.5)
        amp = rng.normal(0.0, 8.0 * sigma_white)
        after = time >= t0
        signal[after] += amp * np.exp(-(time[after] - t0) / tau)
    return signal


def stellar_flares(
    time: NDArray[np.float64], sigma_white: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sharp-rise / exponential-decay flares: strictly positive outliers.

    Present so that the preprocessing has to clip *upward only*.  A symmetric
    sigma clip would happily delete transit points along with the flares.
    """
    signal = np.zeros_like(time)
    span = time[-1] - time[0]
    for _ in range(rng.integers(1, 6)):
        t0 = time[0] + rng.uniform(0.0, span)
        tau = rng.uniform(0.01, 0.06)
        amp = rng.uniform(5.0, 60.0) * sigma_white
        after = time >= t0
        signal[after] += amp * np.exp(-(time[after] - t0) / tau)
    return signal


def trapezoid_transit(
    time: NDArray[np.float64],
    period: float,
    epoch: float,
    depth: float,
    t14: float,
    t23: float,
    *,
    secondary_depth: float = 0.0,
    odd_even_fraction: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate a trapezoidal eclipse model (returns a non-negative dip profile).

    Parameters
    ----------
    depth:
        Fractional depth at mid-transit of the *primary* event.
    t14, t23:
        Total and flat-bottom durations.  ``t23 = 0`` yields a V shape.
    secondary_depth:
        Depth of the secondary eclipse at phase 0.5 (eclipsing binaries only).
    odd_even_fraction:
        Odd eclipses are deepened and even eclipses shallowed by this fraction.
        Non-zero only for binaries whose true period is twice the detected one.
    """
    dip = np.zeros_like(time)
    if t14 <= 0 or depth <= 0:
        return dip

    half_total, half_flat = t14 / 2.0, t23 / 2.0
    ingress = max(half_total - half_flat, 1e-9)

    epoch_number = np.round((time - epoch) / period)
    phase_time = time - epoch - epoch_number * period
    x = np.abs(phase_time)

    profile = np.clip((half_total - x) / ingress, 0.0, 1.0)
    # Odd/even modulation keys off the parity of the epoch number.
    parity = np.where(epoch_number % 2 == 0, 1.0 + odd_even_fraction, 1.0 - odd_even_fraction)
    dip += depth * profile * parity

    if secondary_depth > 0:
        sec_time = time - (epoch + period / 2.0)
        sec_time -= np.round(sec_time / period) * period
        sec_profile = np.clip((half_total - np.abs(sec_time)) / ingress, 0.0, 1.0)
        dip += secondary_depth * sec_profile

    return dip


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------
class SyntheticTESSSource(LightCurveSource):
    """Generate a labelled population of TESS-like light curves.

    The class composition is set by ``positive_rate`` and ``eclipsing_binary_rate``;
    everything else is a plain variable star.  Curve *i* is generated from a
    seed derived from ``(seed, i)``, so any single curve can be regenerated in
    isolation -- which is what makes the pytest suite cheap.
    """

    def __init__(
        self,
        n_curves: int,
        positive_rate: float,
        eclipsing_binary_rate: float,
        *,
        seed: int,
        survey: SurveyConfig | None = None,
        noise: NoiseConfig | None = None,
        star: StarConfig | None = None,
        planet: PlanetConfig | None = None,
        eb: EclipsingBinaryConfig | None = None,
    ) -> None:
        if not 0.0 <= positive_rate <= 1.0:
            raise ValueError("positive_rate must be in [0, 1]")
        if positive_rate + eclipsing_binary_rate > 1.0:
            raise ValueError("positive_rate + eclipsing_binary_rate must not exceed 1")

        self.n_curves = int(n_curves)
        self.positive_rate = float(positive_rate)
        self.eclipsing_binary_rate = float(eclipsing_binary_rate)
        self.seed = int(seed)
        self.survey = survey or SurveyConfig()
        self.noise = noise or NoiseConfig()
        self.star = star or StarConfig()
        self.planet = planet or PlanetConfig()
        self.eb = eb or EclipsingBinaryConfig()
        self._kinds = self._assign_kinds()

    # -- class composition --------------------------------------------------
    def _assign_kinds(self) -> list[CurveKind]:
        """Deterministically assign exactly the requested class counts.

        Drawing each label from a Bernoulli would make the realised positive
        rate a random variable; at a 4% rate and 2400 curves that is a +/- 0.8%
        swing, which is enough to move the headline metric between runs.  We fix
        the counts and shuffle instead.
        """
        n_pos = int(round(self.n_curves * self.positive_rate))
        n_eb = int(round(self.n_curves * self.eclipsing_binary_rate))
        n_noise = self.n_curves - n_pos - n_eb
        if n_noise < 0:
            raise ValueError("class rates exceed the number of curves")
        kinds: list[CurveKind] = (
            ["planet"] * n_pos + ["eclipsing_binary"] * n_eb + ["noise"] * n_noise
        )
        np.random.default_rng(self.seed).shuffle(kinds)  # type: ignore[arg-type]
        return kinds

    @property
    def kinds(self) -> list[CurveKind]:
        """Ground-truth population label for every curve, in order."""
        return list(self._kinds)

    def __len__(self) -> int:
        return self.n_curves

    def __iter__(self) -> Iterator[LightCurve]:
        for index in range(self.n_curves):
            yield self.generate(index)

    # -- single-curve generation -------------------------------------------
    def _curve_rng(self, index: int) -> np.random.Generator:
        """Independent, reproducible stream per curve."""
        return np.random.default_rng([self.seed, index])

    def generate(self, index: int) -> LightCurve:
        """Generate light curve ``index`` (0-based). Deterministic given the seed."""
        kind = self._kinds[index]
        rng = self._curve_rng(index)
        time = sample_time_grid(self.survey, rng)
        n = time.size

        # --- host star -----------------------------------------------------
        tess_mag = float(rng.uniform(*self.noise.tess_mag_range))
        sigma_white = white_noise_sigma(tess_mag, self.noise)
        r_star = float(rng.uniform(*self.star.radius_range_rsun))
        # Main-sequence-ish mass-radius relation, then mean density in g/cm^3.
        m_star = r_star**0.9
        rho_star = RHO_SUN_CGS * m_star / r_star**3

        meta: dict[str, Any] = {
            "kind": kind,
            "tess_mag": tess_mag,
            "sigma_white": sigma_white,
            "r_star_rsun": r_star,
            "rho_star_cgs": rho_star,
        }

        # --- astrophysical + instrumental background -----------------------
        variability, var_meta = stellar_variability(time, self.star, rng)
        meta.update(var_meta)

        red_rms = sigma_white * float(rng.uniform(*self.noise.red_amplitude_range))
        red_alpha = float(rng.uniform(*self.noise.red_alpha_range))
        red = power_law_noise(n, self.survey.cadence_days, red_alpha, red_rms, rng)
        meta.update({"red_rms": red_rms, "red_alpha": red_alpha})

        systematics = np.zeros_like(time)
        if rng.random() < self.noise.ramp_probability:
            systematics += instrumental_ramps(time, sigma_white, rng)
        if rng.random() < self.noise.flare_probability:
            systematics += stellar_flares(time, sigma_white, rng)

        # --- eclipse signal -------------------------------------------------
        if kind == "planet":
            dip, signal_meta = self._planet_signal(time, period_rng=rng, rho_star=rho_star)
        elif kind == "eclipsing_binary":
            dip, signal_meta = self._binary_signal(time, rng, rho_star=rho_star)
        else:
            dip, signal_meta = np.zeros_like(time), {}
        meta.update(signal_meta)

        white = rng.normal(0.0, sigma_white, size=n)
        flux = 1.0 + variability + red + systematics - dip + white
        flux_err = np.full(n, sigma_white)

        # Record the true photometric SNR of the injected event.  This is the
        # quantity recall should be plotted against; depth alone is not enough.
        n_in = int(np.count_nonzero(dip > 0.5 * dip.max())) if dip.max() > 0 else 0
        meta["n_in_transit_cadences"] = n_in
        meta["true_snr"] = (
            float(dip.max() / sigma_white * np.sqrt(n_in)) if n_in > 0 else 0.0
        )

        return LightCurve(
            target_id=f"SYN-{index:06d}",
            time=time,
            flux=flux,
            flux_err=flux_err,
            label=1 if kind == "planet" else 0,
            meta=meta,
        )

    def _planet_signal(
        self,
        time: NDArray[np.float64],
        period_rng: np.random.Generator,
        rho_star: float,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        return planet_signal(time, period_rng, rho_star, self.planet)

    def _binary_signal(
        self,
        time: NDArray[np.float64],
        rng: np.random.Generator,
        rho_star: float,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        return binary_signal(time, rng, rho_star, self.eb)


# --------------------------------------------------------------------------
# Eclipse signals.  Module-level so injection into real photometry
# (transitml.data.injection) draws from exactly the same populations.
# --------------------------------------------------------------------------
def planet_signal(
    time: NDArray[np.float64],
    rng: np.random.Generator,
    rho_star: float,
    cfg: PlanetConfig,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Draw one planet from ``cfg`` around a star of density ``rho_star``.

    Returns the dip profile on ``time`` (non-negative, to be subtracted from
    or multiplied into the flux) and the injected parameters.  Shared by the
    synthetic generator and by injection into real photometry, so both
    populations follow exactly the same physics.
    """
    period = float(10 ** rng.uniform(*np.log10(cfg.period_range_days)))
    k = float(10 ** rng.uniform(*np.log10(cfg.radius_ratio_range)))
    impact = float(rng.uniform(0.0, cfg.impact_parameter_max))
    a_rs = scaled_semi_major_axis(period, rho_star)
    t14, t23 = transit_durations(period, a_rs, k, impact)
    depth = k**2 * cfg.limb_darkening_boost
    epoch = float(time[0] + rng.uniform(0.0, period))

    dip = trapezoid_transit(time, period, epoch, depth, t14, t23)
    meta = {
        "period": period,
        "epoch": epoch,
        "depth": depth,
        "duration_t14": t14,
        "duration_t23": t23,
        "radius_ratio": k,
        "impact_parameter": impact,
        "a_over_rs": a_rs,
        "n_transits_in_window": int(
            np.count_nonzero(np.unique(np.round((time - epoch) / period)) * 0 + 1)
            if t14 > 0
            else 0
        ),
    }
    # Count transits that actually have in-window coverage.
    if t14 > 0:
        epochs = np.round((time - epoch) / period)
        covered = np.unique(epochs[np.abs(time - epoch - epochs * period) < t14 / 2.0])
        meta["n_transits_in_window"] = int(covered.size)
    return dip, meta

def binary_signal(
    time: NDArray[np.float64],
    rng: np.random.Generator,
    rho_star: float,
    cfg: EclipsingBinaryConfig,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Draw one eclipsing binary from ``cfg``; same contract as :func:`planet_signal`."""
    period = float(10 ** rng.uniform(*np.log10(cfg.period_range_days)))
    depth = float(10 ** rng.uniform(*np.log10(cfg.primary_depth_range)))
    grazing = bool(rng.random() < cfg.grazing_probability)

    a_rs = scaled_semi_major_axis(period, rho_star)
    # Effective radius ratio implied by the eclipse depth, for the geometry.
    k = float(np.sqrt(min(depth, 0.9)))
    impact = float(rng.uniform(1.0 - k, 1.0 + k * 0.9)) if grazing else float(
        rng.uniform(0.0, max(1.0 - k, 0.05))
    )
    t14, t23 = transit_durations(period, a_rs, k, impact)
    if t14 <= 0:  # geometry closed off; fall back to a shallow central event
        impact, grazing = 0.3, False
        t14, t23 = transit_durations(period, a_rs, k, impact)

    # Grazing eclipses are shallower than the nominal (R2/R1)^2.
    if grazing:
        depth *= float(rng.uniform(0.15, 0.8))

    secondary = depth * float(rng.uniform(*cfg.secondary_depth_fraction_range))
    odd_even = float(rng.uniform(*cfg.odd_even_fraction_range))
    epoch = float(time[0] + rng.uniform(0.0, period))

    dip = trapezoid_transit(
        time,
        period,
        epoch,
        depth,
        t14,
        t23,
        secondary_depth=secondary,
        odd_even_fraction=odd_even,
    )
    meta = {
        "period": period,
        "epoch": epoch,
        "depth": depth,
        "duration_t14": t14,
        "duration_t23": t23,
        "impact_parameter": impact,
        "grazing": grazing,
        "secondary_depth": secondary,
        "odd_even_fraction": odd_even,
    }
    return dip, meta
