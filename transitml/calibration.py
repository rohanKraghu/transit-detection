"""Calibrated probabilities: Platt scaling fitted on out-of-fold scores.

The classifier's score is not a probability.  It is trained with
``class_weight="balanced"``, which weights the 4% of planets as heavily as the
96% of everything else, so its output answers "planet or not, if both were
equally common": a score of 0.5 sits roughly where the two classes are equally
dense, not where half the stars are planets.  That is fine for ranking and for
the operating threshold, which only need the order, but a number a person reads
as "the chance this is a planet" should be one.

:func:`fit_platt` maps the trees' log-odds ``s`` to a probability with one
logistic regression, ``P(planet) = 1 / (1 + exp(-(a * s + b)))`` (Platt 1999).
It is fitted on the out-of-fold training scores, the same ones the threshold
is chosen from, so neither the test set nor the in-sample scores of the refit
model touch it.  With ``a > 0`` the map is strictly increasing, so ranking,
average precision and which stars clear the threshold are all unchanged; only
the number attached to each star moves.

Why Platt rather than isotonic regression: with about 60 training planets an
isotonic fit is a staircase with a handful of steps, each set by two or three
stars, and it ties scores the trees had kept apart.  Two parameters is what
this much data supports.

The probability is for a star drawn from the training population, where
``train_positive_rate`` of stars host a detectable planet.  For a population
with a different rate, Bayes' rule shifts the log-odds by
``logit(rate) - logit(train_positive_rate)``; :meth:`PlattScaling.log_odds`
does that when given ``positive_rate``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

#: Bin edges for the reliability table.  Log-spaced at the low end, where
#: almost every star sits at a 4% planet rate; equal-width bins would put 90%
#: of the test set in the first one.
RELIABILITY_EDGES: tuple[float, ...] = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0)


@dataclass(frozen=True)
class PlattScaling:
    """``P(planet) = expit(slope * s + intercept)`` for trees' log-odds ``s``."""

    slope: float
    intercept: float
    train_positive_rate: float

    def log_odds(
        self, raw: NDArray[np.float64], positive_rate: float | None = None
    ) -> NDArray[np.float64]:
        """Calibrated log-odds; ``positive_rate`` re-targets them to another population."""
        out = self.slope * np.asarray(raw, dtype=float) + self.intercept
        if positive_rate is not None:
            if not 0.0 < positive_rate < 1.0:
                raise ValueError(f"positive_rate must lie in (0, 1), got {positive_rate}")
            out = out + float(logit(positive_rate) - logit(self.train_positive_rate))
        return out

    def probability(
        self, raw: NDArray[np.float64], positive_rate: float | None = None
    ) -> NDArray[np.float64]:
        """Calibrated ``P(planet)``."""
        return expit(self.log_odds(raw, positive_rate))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_platt(raw: NDArray[np.float64], y: NDArray[np.int_]) -> PlattScaling:
    """Fit Platt scaling to scores ``raw`` (log-odds) and labels ``y``.

    Uses Platt's smoothed targets, ``(N+ + 1) / (N+ + 2)`` for positives and
    ``1 / (N- + 2)`` for negatives, rather than 1 and 0: with few positives an
    unsmoothed fit on a separable set would push the slope to infinity.  The
    slope is bounded below by a small positive number so the map stays
    strictly increasing whatever the data.
    """
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=int)
    if raw.shape != y.shape or raw.ndim != 1:
        raise ValueError("raw and y must be 1-D arrays of the same length")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw scores must be finite")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Platt scaling needs both classes")
    target = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

    def loss(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        z = params[0] * raw + params[1]
        value = float(np.sum(np.logaddexp(0.0, z) - target * z))
        residual = expit(z) - target
        return value, np.array([np.dot(residual, raw), residual.sum()])

    start = np.array([1.0, float(np.log((n_pos + 1.0) / (n_neg + 1.0)))])
    result = minimize(
        loss,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(1e-6, None), (None, None)],
        options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 1000},
    )
    slope, intercept = (float(v) for v in result.x)
    return PlattScaling(slope, intercept, train_positive_rate=n_pos / y.size)


# --------------------------------------------------------------------------
# How good is a probability?  Proper scores and a reliability table
# --------------------------------------------------------------------------
def brier_score(y: NDArray[np.int_], p: NDArray[np.float64]) -> float:
    """Mean squared error of the probabilities; 0 is perfect."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def log_loss(y: NDArray[np.int_], p: NDArray[np.float64]) -> float:
    """Mean negative log-likelihood in nats, with probabilities clipped at 1e-15."""
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def reliability_table(
    y: NDArray[np.int_],
    p: NDArray[np.float64],
    edges: tuple[float, ...] = RELIABILITY_EDGES,
) -> list[dict[str, Any]]:
    """Per probability bin: how many stars, the mean prediction, the observed planet rate.

    A calibrated model has ``mean_predicted`` close to ``observed_rate`` in
    every bin, within the binomial scatter that ``n`` and ``n_planets`` imply.
    Empty bins are kept so tables from different runs line up.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = index == k
        n = int(sel.sum())
        rows.append(
            {
                "p_low": float(lo),
                "p_high": float(hi),
                "n": n,
                "n_planets": int(y[sel].sum()),
                "mean_predicted": float(p[sel].mean()) if n else float("nan"),
                "observed_rate": float(y[sel].mean()) if n else float("nan"),
            }
        )
    return rows


def expected_calibration_error(
    y: NDArray[np.int_],
    p: NDArray[np.float64],
    edges: tuple[float, ...] = RELIABILITY_EDGES,
) -> float:
    """Star-weighted mean of ``|mean predicted - observed rate|`` over the reliability bins."""
    rows = reliability_table(y, p, edges)
    total = sum(r["n"] for r in rows)
    return float(
        sum(r["n"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in rows if r["n"])
        / total
    )


def calibration_summary(
    y: NDArray[np.int_],
    uncalibrated: NDArray[np.float64],
    calibrated: NDArray[np.float64],
    calibrated_log_odds: NDArray[np.float64],
    flagged: NDArray[np.bool_],
    scaling: PlattScaling,
    threshold_probability: float,
) -> dict[str, Any]:
    """Everything the report says about calibration, on one held-out set.

    Three probabilities are scored with the same proper scores: the constant
    training planet rate (the forecast that knows nothing about any one star),
    the classifier's own score read as a probability, and the calibrated one.
    The calibration slope is a logistic fit of the outcome on the calibrated
    log-odds: 1 is ideal, below 1 means the probabilities are too extreme.
    """
    y = np.asarray(y, dtype=int)
    constant = np.full(y.size, scaling.train_positive_rate)
    forecasts = {
        "training_rate": constant,
        "uncalibrated": np.asarray(uncalibrated, dtype=float),
        "calibrated": np.asarray(calibrated, dtype=float),
    }
    scores = {
        name: {
            "brier": brier_score(y, p),
            "log_loss": log_loss(y, p),
            "ece": expected_calibration_error(y, p),
        }
        for name, p in forecasts.items()
    }
    refit = fit_platt(calibrated_log_odds, y) if 0 < y.sum() < y.size else None
    flagged = np.asarray(flagged, dtype=bool)
    return {
        "method": "Platt scaling on out-of-fold training log-odds",
        "scaling": scaling.to_dict(),
        "threshold_as_probability": float(threshold_probability),
        "scores": scores,
        "reliability": {
            "uncalibrated": reliability_table(y, forecasts["uncalibrated"]),
            "calibrated": reliability_table(y, forecasts["calibrated"]),
        },
        "expected_planets": {
            "all": float(forecasts["calibrated"].sum()),
            "all_observed": int(y.sum()),
            "flagged": float(forecasts["calibrated"][flagged].sum()),
            "flagged_observed": int(y[flagged].sum()),
            "n_flagged": int(flagged.sum()),
        },
        "test_calibration_slope": refit.slope if refit else float("nan"),
        "test_calibration_intercept": refit.intercept if refit else float("nan"),
    }
