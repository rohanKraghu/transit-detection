"""Central configuration for the transit-detection pipeline.

Every tunable number in the project lives here so that a reviewer can find the
knobs in one place and so that :func:`default_config` fully determines the
run (together with the random seed).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# --- Global reproducibility ------------------------------------------------
SEED: int = 42


@dataclass(frozen=True)
class SurveyConfig:
    """Observing setup: one TESS-like sector of photometry.

    Defaults mimic a TESS 30-minute Full-Frame-Image light curve: ~27.4 days of
    near-continuous coverage interrupted by the mid-sector data downlink.
    """

    baseline_days: float = 27.4
    cadence_minutes: float = 30.0
    #: Mid-sector data downlink gap (TESS pauses to transmit at perigee).
    downlink_gap_days: float = 1.0
    #: Fraction of cadences lost to scattered light / momentum dumps / cosmic rays.
    random_dropout_fraction: float = 0.03

    @property
    def cadence_days(self) -> float:
        return self.cadence_minutes / 1440.0


@dataclass(frozen=True)
class NoiseConfig:
    """Photometric noise budget.

    ``white`` is shot + read noise, drawn from an empirical magnitude-to-scatter
    relation.  ``red`` is the correlated component (pointing jitter, thermal
    drift, imperfect background subtraction) generated as 1/f^alpha noise, which
    is what actually limits shallow-transit detection in real data.
    """

    tess_mag_range: tuple[float, float] = (8.0, 14.0)
    #: Per-cadence scatter in ppm at the bright end of ``tess_mag_range``.
    white_ppm_at_bright_end: float = 60.0
    #: Scatter grows as 10 ** (slope * (mag - mag_bright)).
    white_mag_slope: float = 0.25
    #: Red-noise RMS as a multiple of the white-noise RMS.
    red_amplitude_range: tuple[float, float] = (0.4, 2.5)
    #: Power-law index of the red-noise PSD (1 = flicker, 2 = random walk).
    red_alpha_range: tuple[float, float] = (0.8, 2.0)
    #: Probability a light curve contains instrumental ramps (momentum dumps).
    ramp_probability: float = 0.45
    #: Probability a light curve contains stellar flares.
    flare_probability: float = 0.15


@dataclass(frozen=True)
class StarConfig:
    """Host-star population and its out-of-transit variability."""

    radius_range_rsun: tuple[float, float] = (0.35, 1.7)
    #: Rotational modulation / pulsation amplitude (fractional, peak-to-peak-ish).
    variability_amplitude_range: tuple[float, float] = (1e-4, 8e-3)
    variability_period_range_days: tuple[float, float] = (0.4, 25.0)
    n_variability_harmonics: int = 3


@dataclass(frozen=True)
class PlanetConfig:
    """Injected planetary transits (the positive class)."""

    period_range_days: tuple[float, float] = (1.0, 13.0)
    #: Planet-to-star radius ratio; 0.02 ~ super-Earth, 0.11 ~ hot Jupiter.
    radius_ratio_range: tuple[float, float] = (0.02, 0.11)
    impact_parameter_max: float = 0.95
    #: Extra depth from limb darkening relative to the geometric (Rp/Rs)^2.
    limb_darkening_boost: float = 1.18
    #: The planet's occultation: reflected light at this geometric albedo
    #: (Bond albedo 1.5 times it, a Lambert sphere) plus dayside emission
    #: with this heat-redistribution factor, from 1/4 (heat spread evenly
    #: over the planet) to 2/3 (none).  Typical of hot Jupiters in TESS.
    geometric_albedo: float = 0.1
    heat_redistribution: float = 0.5


@dataclass(frozen=True)
class EclipsingBinaryConfig:
    """Injected eclipsing binaries: astrophysical false positives.

    These are labelled **negative**.  They are the reason odd/even depth
    differences and secondary-eclipse tests exist in real vetting pipelines, and
    they are what makes this a classification problem rather than a
    signal-detection problem.
    """

    period_range_days: tuple[float, float] = (0.6, 13.0)
    primary_depth_range: tuple[float, float] = (2e-3, 6e-2)
    #: Secondary eclipse depth as a fraction of the primary depth.
    secondary_depth_fraction_range: tuple[float, float] = (0.0, 0.45)
    #: Fractional depth difference between odd and even eclipses (period aliasing).
    odd_even_fraction_range: tuple[float, float] = (0.0, 0.25)
    #: Probability the binary is grazing (V-shaped, shallow -- the hard case).
    grazing_probability: float = 0.45


@dataclass(frozen=True)
class DatasetConfig:
    """Size and class composition of the generated dataset."""

    n_curves: int = 2400
    #: Fraction of curves containing a genuine transiting planet (label = 1).
    positive_rate: float = 0.04
    #: Fraction of curves containing an eclipsing binary (label = 0, hard negative).
    eclipsing_binary_rate: float = 0.06
    #: Held-out fraction, stratified on the label.
    test_size: float = 0.35
    #: Stratified folds used on the *training* split for threshold selection.
    n_cv_folds: int = 5


@dataclass(frozen=True)
class PreprocessConfig:
    """Detrending parameters.

    ``knot_spacing_days`` is the load-bearing one.  It must be comfortably
    larger than the longest transit duration we care about (~0.26 d), or the
    spline gains basis functions narrow enough to bend into the transit on the
    first, unweighted IRLS iteration -- after which the robust weights see
    nothing to reject.  See ``transitml.preprocess``.
    """

    # -- robust spline --
    knot_spacing_days: float = 0.75
    spline_degree: int = 3
    #: Split the series at gaps larger than this; each segment gets its own spline.
    gap_threshold_days: float = 0.25
    #: IRLS cycles in the robust (Tukey biweight) fit.
    irls_iterations: int = 6
    #: Biweight tuning constant: residuals beyond this many robust sigma get
    #: zero weight.  4.685 is the textbook value for 95% Gaussian efficiency.
    biweight_tuning: float = 4.685

    # -- rotation (Fourier) term, for variability the spline cannot follow --
    rotation_min_period_days: float = 0.15
    rotation_max_period_days: float = 5.0
    rotation_n_frequencies: int = 1500
    #: Fourier terms per rotation period (spot patterns are not pure sinusoids).
    rotation_harmonics: int = 3
    #: How many distinct periods may be added.  One is enough in practice: the
    #: spline absorbs whatever a second, weaker period would have contributed,
    #: and a second robust refit doubles the cost of the whole stage.
    rotation_max_terms: int = 1
    #: A rotation term is kept only if it shrinks the robust residual scale by
    #: at least this fraction.  Keeps quiet stars spline-only.
    rotation_min_improvement: float = 0.05

    # -- outlier rejection --
    #: Upward-only: downward outliers are the thing we are trying to find.
    upper_clip_sigma: float = 4.0
    clip_iterations: int = 2

    # -- second pass with the strongest signal masked --
    #: Detrend twice: blind, then with the cadences of the strongest BLS signal
    #: kept out of the fit.  The robust weights alone cannot keep a transit out
    #: when it sits against a data gap: the spline is free enough there to bend
    #: into it on the first, unweighted iteration.
    mask_signal: bool = True
    #: Half-width of the window masked around each transit, in BLS durations.
    mask_half_width_durations: float = 1.0
    #: Only a signal the search believes in is masked: the first-pass peak must
    #: reach this ``bls_sde`` (the multi-planet search's threshold).  Masking a
    #: noise peak protects nothing and feeds on itself: with the fit no longer
    #: allowed to follow those cadences, the same peak comes back stronger.
    #: Ungated, the share of plain variable stars clearing SDE 5.5 on the
    #: synthetic run went from 3.7% to 7.8%.
    mask_min_sde: float = 5.5
    #: No second pass when the mask would cover more than this fraction of the
    #: cadences: a long box at a short period is not a transit worth protecting.
    mask_max_fraction: float = 0.25
    #: A spline basis function keeping less than this fraction of its squared
    #: weight outside the mask gets its cadences back.
    mask_min_support: float = 0.05


@dataclass(frozen=True)
class BLSConfig:
    """Box Least Squares search grid."""

    min_period_days: float = 0.5
    #: Capped so at least two transits fit inside the baseline.
    max_period_fraction_of_baseline: float = 0.5
    n_periods: int = 2000
    durations_days: tuple[float, ...] = (0.04, 0.07, 0.11, 0.16, 0.24)


@dataclass(frozen=True)
class MultiPlanetConfig:
    """Iterative search for additional periodic signals (``transitml.search``).

    Only the vetting tool uses this.  The classifier and its headline numbers
    stay on the single strongest BLS peak.
    """

    #: Most signals reported per light curve, the primary included.
    max_signals: int = 3
    #: A peak counts as a candidate only if its ``bls_sde`` reaches this.  On
    #: the synthetic run, 3.7% of variable stars with nothing in them clear 5.5
    #: on their first search, against 62% of planets.
    min_sde: float = 5.5
    #: Cadences within this many transit durations of a found mid-transit time
    #: are masked before the next search.
    mask_half_width_durations: float = 1.5
    #: Stop when fewer cadences than this survive the masking.
    min_cadences: int = 200


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation and operating-point selection."""

    #: Minimum precision an operating point must deliver.  Set by follow-up cost:
    #: at 0.5, at most one wasted follow-up campaign per confirmed planet.
    target_precision: float = 0.50
    #: The floor is applied to a one-sided Wilson lower confidence bound on the
    #: out-of-fold precision, not to its point estimate, with this many sigma.
    #: Picking the deepest point that just clears a floor is an optimisation,
    #: and the point-estimate precision there is biased upward (the original
    #: run promised 0.500 in CV and delivered 0.407 on test).  1.0 is a
    #: one-sided ~84% bound, matching the 68% intervals quoted elsewhere;
    #: 0.0 recovers the point-estimate rule.
    precision_lcb_z: float = 1.0
    #: Size of the "tonight's target list" used for precision@k.
    top_k: int = 20


@dataclass(frozen=True)
class Config:
    """The whole run configuration."""

    seed: int = SEED
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    star: StarConfig = field(default_factory=StarConfig)
    planet: PlanetConfig = field(default_factory=PlanetConfig)
    eb: EclipsingBinaryConfig = field(default_factory=EclipsingBinaryConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    bls: BLSConfig = field(default_factory=BLSConfig)
    multi_planet: MultiPlanetConfig = field(default_factory=MultiPlanetConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy (for the metrics artefact)."""
        return asdict(self)


def default_config() -> Config:
    """Return the configuration used by ``run_pipeline.py``."""
    return Config()
