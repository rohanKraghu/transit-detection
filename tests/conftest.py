"""Shared fixtures.

The tests use a deliberately cut-down BLS grid and short light curves: the
point is to check behaviour, not to reproduce the headline numbers, and a
suite that takes two minutes does not get run.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from transitml.config import BLSConfig, Config, default_config
from transitml.data.base import LightCurve
from transitml.data.synthetic import SyntheticTESSSource
from transitml.physics import RHO_SUN_CGS, scaled_semi_major_axis, transit_durations


@pytest.fixture(scope="session")
def config() -> Config:
    """The production configuration, with a coarse BLS grid for speed."""
    base = default_config()
    return replace(base, bls=replace(base.bls, n_periods=600))


@pytest.fixture(scope="session")
def fast_bls() -> BLSConfig:
    return BLSConfig(n_periods=600)


def make_source(config: Config, n_curves: int, positive_rate: float, eb_rate: float, seed: int):
    """Build a :class:`SyntheticTESSSource` from a :class:`Config`."""
    return SyntheticTESSSource(
        n_curves=n_curves,
        positive_rate=positive_rate,
        eclipsing_binary_rate=eb_rate,
        seed=seed,
        survey=config.survey,
        noise=config.noise,
        star=config.star,
        planet=config.planet,
        eb=config.eb,
    )


def clean_transit_curve(
    *,
    period: float = 3.0,
    depth: float = 4e-3,
    duration: float = 0.12,
    sigma: float = 2e-4,
    variability_amplitude: float = 0.0,
    variability_period: float = 4.0,
    seed: int = 0,
    baseline: float = 27.4,
    cadence_minutes: float = 30.0,
) -> tuple[LightCurve, dict[str, float]]:
    """A hand-built light curve with an exactly known box transit.

    Used where a test needs to assert on a number the generator's own physics
    would otherwise make approximate: the depth is exactly ``depth``, the
    duration is exactly ``duration``, and nothing else is going on except the
    noise and variability the caller asks for.
    """
    rng = np.random.default_rng(seed)
    dt = cadence_minutes / 1440.0
    time = np.arange(0.0, baseline, dt)
    epoch = 1.3

    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    in_transit = np.abs(phase) < duration / 2.0

    flux = np.ones_like(time)
    if variability_amplitude > 0:
        flux += variability_amplitude * np.sin(2.0 * np.pi * time / variability_period + 0.7)
    flux[in_transit] -= depth
    flux += rng.normal(0.0, sigma, size=time.size)

    lc = LightCurve(
        target_id="TEST-0001",
        time=time,
        flux=flux,
        flux_err=np.full(time.size, sigma),
        label=1,
        meta={"kind": "planet", "period": period, "epoch": epoch, "depth": depth},
    )
    truth = {
        "period": period,
        "epoch": epoch,
        "depth": depth,
        "duration": duration,
        "sigma": sigma,
        "n_in_transit": int(in_transit.sum()),
        "snr": depth / sigma * np.sqrt(in_transit.sum()),
    }
    return lc, truth


def expected_duration_for(period: float, radius_ratio: float, impact: float) -> float:
    """T14 for a solar-density host, used to sanity-check the generator."""
    a_rs = scaled_semi_major_axis(period, RHO_SUN_CGS)
    return transit_durations(period, a_rs, radius_ratio, impact)[0]


@pytest.fixture(scope="session")
def tiny_model_config() -> Config:
    """A miniature run for tests that need a trained model, not a good one."""
    base = default_config()
    return replace(
        base,
        dataset=replace(base.dataset, n_curves=120, positive_rate=0.15, eclipsing_binary_rate=0.1),
        bls=replace(base.bls, n_periods=400),
    )


@pytest.fixture(scope="session")
def tiny_trained(tiny_model_config):
    """``(split, trained)`` on 120 synthetic curves, 18 of them planets."""
    from transitml.data.loader import build_default_dataset
    from transitml.model import make_split, train

    dataset = build_default_dataset(tiny_model_config, n_jobs=2)
    split = make_split(dataset, test_size=0.35, seed=tiny_model_config.seed)
    trained = train(split, n_folds=3, seed=tiny_model_config.seed, target_precision=0.5)
    return split, trained
