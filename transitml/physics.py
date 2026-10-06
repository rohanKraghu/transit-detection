"""Transit geometry shared by the generator and the feature extractor.

Keeping these in one module means the "expected duration" feature is computed
with exactly the same physics the injector used -- and, more importantly, that
the physics is written down once where it can be checked.
"""

from __future__ import annotations

import numpy as np

#: Newton's constant, CGS.
G_CGS: float = 6.674e-8
SECONDS_PER_DAY: float = 86400.0
#: Mean density of the Sun in g/cm^3.
RHO_SUN_CGS: float = 1.41
#: The Sun's radius in centimetres.
R_SUN_CM: float = 6.957e10


def scaled_semi_major_axis(period_days: float, stellar_density_cgs: float) -> float:
    """a/R* from Kepler's third law.

    For a circular orbit around a star of mean density ``rho``::

        a / R* = (G * rho * P^2 / (3 pi)) ** (1/3)

    This is the form transit fitters actually use, because a/R* and the transit
    shape constrain stellar density directly without needing the mass.
    """
    p_sec = period_days * SECONDS_PER_DAY
    return float((G_CGS * stellar_density_cgs * p_sec**2 / (3.0 * np.pi)) ** (1.0 / 3.0))


def transit_durations(
    period: float, a_over_rs: float, radius_ratio: float, impact: float
) -> tuple[float, float]:
    """Total (T14) and flat-bottom (T23) durations in days.

    ``T23 == 0`` means the event is grazing, hence V-shaped with no flat bottom.
    Returns ``(0, 0)`` when the geometry admits no eclipse at all.
    """
    outer = (1.0 + radius_ratio) ** 2 - impact**2
    inner = (1.0 - radius_ratio) ** 2 - impact**2
    if outer <= 0 or a_over_rs <= 0:
        return 0.0, 0.0
    t14 = period / np.pi * np.arcsin(min(1.0, np.sqrt(outer) / a_over_rs))
    t23 = (
        period / np.pi * np.arcsin(min(1.0, np.sqrt(inner) / a_over_rs))
        if inner > 0
        else 0.0
    )
    return float(t14), float(t23)


def expected_central_duration(period_days: float, stellar_density_cgs: float) -> float:
    """Duration of a central (b = 0) transit of a small planet, in days.

    Used as a physical yardstick: an event whose measured duration is many times
    longer than this cannot be a planet transit of a main-sequence star at that
    period, whatever its depth.  Real vetting pipelines apply the same test in
    the form of a "fitted stellar density" consistency check.
    """
    a_rs = scaled_semi_major_axis(period_days, stellar_density_cgs)
    if a_rs <= 1.0:
        return float(period_days / 2.0)
    return float(period_days / np.pi * np.arcsin(1.0 / a_rs))


# --------------------------------------------------------------------------
# Occultations.  A hot Jupiter's own secondary eclipse, when the star hides
# the planet's dayside, is a few percent of its transit depth in the TESS
# band: easily significant on a bright star, and not a sign of a binary.
# --------------------------------------------------------------------------
#: Effective temperature of the Sun, in kelvin.
T_SUN_K: float = 5772.0
#: Second radiation constant hc/k, in metre-kelvin.
HC_OVER_K: float = 1.4388e-2
#: Effective wavelength of the TESS band (600-1000 nm), in metres.
TESS_WAVELENGTH_M: float = 800e-9


def main_sequence_teff(stellar_density_cgs: float) -> float:
    """Effective temperature, in kelvin, of a main-sequence star of this density.

    Inverts the mass-radius relation the generators use (M = R^0.9 in solar
    units, so rho / rho_sun = R^-2.1) and takes L = M^4, which gives
    T_eff = T_sun R^0.4: about 3800 K at 0.35 R_sun and 7100 K at 1.7 R_sun.
    That runs a few hundred kelvin warm for M dwarfs, which costs nothing here:
    around a star that dense, even a one-day orbit is too wide for the planet
    to show an occultation.
    """
    radius = (stellar_density_cgs / RHO_SUN_CGS) ** (-1.0 / 2.1)
    return float(T_SUN_K * radius**0.4)


def main_sequence_density(t_eff: float) -> float:
    """Mean density, in g/cm^3, of a main-sequence star at this temperature.

    The inverse of :func:`main_sequence_teff`, for a catalogue star with a
    temperature and nothing else.
    """
    radius = (t_eff / T_SUN_K) ** 2.5
    return float(RHO_SUN_CGS * radius**-2.1)


def density_from_gravity(logg_cgs: float, radius_rsun: float) -> float:
    """Mean density, in g/cm^3, from log g (cgs) and radius: 3 g / (4 pi G R)."""
    return float(3.0 * 10.0**logg_cgs / (4.0 * np.pi * G_CGS * radius_rsun * R_SUN_CM))


def tess_brightness_ratio(t_planet: float, t_star: float) -> float:
    """Planck surface-brightness ratio B(t_planet) / B(t_star) in the TESS band."""
    x_planet = HC_OVER_K / (TESS_WAVELENGTH_M * t_planet)
    x_star = HC_OVER_K / (TESS_WAVELENGTH_M * t_star)
    return float(np.expm1(x_star) / np.expm1(x_planet))


def occultation_depth(
    radius_ratio: float,
    a_over_rs: float,
    t_eff: float,
    *,
    geometric_albedo: float,
    bond_albedo: float,
    redistribution: float,
) -> float:
    """Depth of a planet's occultation (its secondary eclipse) in the TESS band.

    Reflected light, ``A_g (Rp/a)^2``, plus dayside thermal emission,
    ``(Rp/R*)^2 B(T_day) / B(T_eff)``, with the dayside at
    ``T_day = T_eff sqrt(R*/a) [f (1 - A_B)]^(1/4)`` (Cowan & Agol 2011).  The
    redistribution factor ``f`` runs from 1/4 (heat spread evenly over the
    planet) to 2/3 (none: the dayside re-radiates where it is heated).
    """
    t_day = t_eff * np.sqrt(1.0 / a_over_rs) * (redistribution * (1.0 - bond_albedo)) ** 0.25
    thermal = radius_ratio**2 * tess_brightness_ratio(t_day, t_eff)
    reflected = geometric_albedo * (radius_ratio / a_over_rs) ** 2
    return float(thermal + reflected)


# --------------------------------------------------------------------------
# The allowance.  How deep a planet's occultation can be depends on its star:
# the temperature sets how bright the dayside can glow, and the density, with
# the period, how close the planet must orbit.  The secondary-eclipse test
# counts only what is deeper than the deepest occultation a planet could show
# around that star.
# --------------------------------------------------------------------------
#: The planet the allowance is for: no heat redistribution, a Bond albedo of
#: zero and this geometric albedo.  That is hotter and brighter than the hot
#: Jupiters measured so far (the synthetic planets use 0.1 and a
#: redistribution factor of 0.5, see ``PlanetConfig``).
OCCULTATION_LIMIT_ALBEDO: float = 0.3
#: Whatever the star, the allowance stops at this fraction of the transit
#: depth.  KELT-9 b, the hottest planet known, shows about a tenth.  Beyond
#: that the orbit the period and density imply hugs a hot or swollen star so
#: closely that a binary is the likelier reading.
OCCULTATION_LIMIT_FRACTION: float = 0.15


def max_occultation_fraction(
    period_days: float, t_eff: float, stellar_density_cgs: float
) -> float:
    """The deepest occultation a planet on this orbit can show, per unit transit depth.

    Evaluated for the planet described at ``OCCULTATION_LIMIT_ALBEDO`` around
    the given star, and capped at ``OCCULTATION_LIMIT_FRACTION``.  Around a
    Sun-like star that is 3.5% of the transit depth at one day and 0.5% at
    three; around a 7500 K star of density 0.4 g/cm^3, 5.4% at two days and
    the full 15% at one.  NaN when the star's temperature or density is
    unknown.
    """
    if not (period_days > 0 and t_eff > 0 and stellar_density_cgs > 0):
        return float("nan")
    a_rs = max(scaled_semi_major_axis(period_days, stellar_density_cgs), 1.0)
    fraction = occultation_depth(
        1.0,
        a_rs,
        t_eff,
        geometric_albedo=OCCULTATION_LIMIT_ALBEDO,
        bond_albedo=0.0,
        redistribution=2.0 / 3.0,
    )
    return float(min(fraction, OCCULTATION_LIMIT_FRACTION))
