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
