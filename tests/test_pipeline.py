"""The headline command must run end to end and stay reproducible."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from transitml.config import default_config
from transitml.data.loader import build_default_dataset
from transitml.evaluate import evaluate, format_report
from transitml.features import FEATURE_NAMES
from transitml.model import make_split, train


@pytest.fixture(scope="module")
def small_config():
    """A miniature version of the real run: same code path, 120 curves."""
    base = default_config()
    return replace(
        base,
        dataset=replace(base.dataset, n_curves=120, positive_rate=0.1, eclipsing_binary_rate=0.1),
        bls=replace(base.bls, n_periods=400),
    )


@pytest.fixture(scope="module")
def small_dataset(small_config):
    return build_default_dataset(small_config, n_jobs=2)


def test_dataset_shape_and_composition(small_config, small_dataset):
    assert len(small_dataset) == small_config.dataset.n_curves
    assert list(small_dataset.features.columns) == list(FEATURE_NAMES)
    assert small_dataset.positive_rate == pytest.approx(
        small_config.dataset.positive_rate, abs=1e-9
    )
    assert set(np.unique(small_dataset.y)) <= {0, 1}
    # Eclipsing binaries are present and are negatives.
    kinds = small_dataset.meta["kind"].value_counts().to_dict()
    assert kinds.get("eclipsing_binary", 0) == 12
    binaries = small_dataset.meta["kind"] == "eclipsing_binary"
    assert small_dataset.y[binaries.to_numpy()].sum() == 0


def test_pipeline_is_reproducible(small_config, small_dataset):
    """Same seed, same feature matrix -- exactly."""
    again = build_default_dataset(small_config, n_jobs=2)
    np.testing.assert_array_equal(again.y, small_dataset.y)
    np.testing.assert_allclose(again.X, small_dataset.X, rtol=0, atol=0, equal_nan=True)


def test_end_to_end_produces_a_report(small_config, small_dataset, tmp_path):
    split = make_split(small_dataset, test_size=0.35, seed=small_config.seed)
    model = train(split, n_folds=3, seed=small_config.seed, target_precision=0.5)
    result = evaluate(small_dataset, split, model, top_k=10, seed=small_config.seed)

    assert result.n_test == len(split.y_test)
    assert 0.0 <= result.model.average_precision <= 1.0
    # Even on 120 curves the model should clear chance by a wide margin.
    assert result.model.average_precision > 2 * result.chance_average_precision

    report = format_report(result)
    assert "Average precision" in report
    assert "Confusion matrix" in report
    assert "accuracy" not in report.lower(), "accuracy must not be reported"

    payload = json.dumps(result.to_dict(), default=str)
    (tmp_path / "metrics.json").write_text(payload)
    assert json.loads(payload)["n_test"] == result.n_test


def test_figures_are_written_without_a_display(small_config, small_dataset, tmp_path):
    """Agg backend, PNGs on disk, no window anywhere."""
    import matplotlib

    from transitml.data.loader import sample_light_curves
    from transitml.plots import plot_all

    assert matplotlib.get_backend().lower() == "agg"

    split = make_split(small_dataset, test_size=0.35, seed=small_config.seed)
    model = train(split, n_folds=3, seed=small_config.seed, target_precision=0.5)
    result = evaluate(small_dataset, split, model, top_k=10, seed=small_config.seed)

    kinds = small_dataset.meta["kind"].astype(str).to_numpy()
    picks = [
        int(np.flatnonzero(kinds == kind)[0])
        for kind in ("planet", "eclipsing_binary", "noise")
    ]
    paths = plot_all(
        small_dataset, split, model, result,
        sample_light_curves(small_config, picks), small_config, tmp_path,
    )
    assert len(paths) == 4
    for path in paths:
        assert Path(path).exists() and Path(path).stat().st_size > 10_000
