"""The DR25 CNN, offline on synthetic views.  Skipped when torch is not installed."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitml import cnn, kepler_dr25  # noqa: E402

from .test_kepler_dr25 import synthetic_catalogue  # noqa: E402

SMALL = cnn.CNNConfig(
    global_filters=(4, 8),
    local_filters=(8,),
    dense=32,
    max_epochs=12,
    patience=12,
    batch_size=16,
    n_models=2,
)


@pytest.fixture(scope="module")
def training_set(tmp_path_factory):
    tces, curves = synthetic_catalogue(n_per_class=40)
    patch = pytest.MonkeyPatch()
    patch.setattr(kepler_dr25, "load_or_download_star", lambda kepid, cache: curves.get(kepid))
    try:
        return kepler_dr25.build_training_set(
            tces, tmp_path_factory.mktemp("cache"), catalogue=tces, n_jobs=1
        )
    finally:
        patch.undo()


def test_inputs_are_stacked_and_clipped(training_set):
    ts = training_set.subset(np.arange(len(training_set)))  # a copy: the fixture is shared
    ts.views["global"][0, :5] = 1e6
    inputs = cnn.view_tensors(ts)
    assert inputs["global"].shape == (len(ts), 1, 2001)
    assert inputs["local"].shape == (len(ts), 4, 201)
    assert inputs["scalars"].shape == (len(ts), 2)
    assert inputs["global"].max() <= cnn.VIEW_CLIP
    assert all(v.dtype == np.float32 for v in inputs.values())


def test_network_gives_one_logit_per_tce():
    net = cnn.build_network(SMALL, 2001, 201)
    out = net(torch.zeros(3, 1, 2001), torch.zeros(3, 4, 201), torch.zeros(3, 2))
    assert out.shape == (3,)


def test_cnn_learns_synthetic_classes_on_the_boosting_split(training_set, tmp_path):
    torch.manual_seed(0)
    result, models = cnn.evaluate_cnn(training_set, config=SMALL, seed=0)
    assert len(models) == 2 and len(result.epochs) == 2
    train, test = kepler_dr25.group_split(training_set.kepids, 0.3, 0)
    assert result.n_test == test.size
    assert result.n_train + result.n_valid == train.size
    assert result.cnn_average_precision > result.chance_average_precision + 0.15
    lo, hi = result.difference_ci
    assert lo <= hi and 0.0 <= result.difference_share_positive <= 1.0
    report = cnn.format_cnn_report(result)
    assert "CNN minus boosting" in report
    json.dumps(result.to_dict())
    assert cnn.plot_cnn_pr(result, tmp_path / "pr.png").stat().st_size > 0


def test_cli_writes_report_and_weights(training_set, tmp_path):
    path = training_set.save(tmp_path / "ts.npz")
    code = cnn.main(
        [
            "--training-set", str(path),
            "--results-dir", str(tmp_path / "out"),
            "--figures-dir", str(tmp_path / "fig"),
            "--n-models", "1",
            "--max-epochs", "2",
        ]
    )
    assert code == 0
    assert (tmp_path / "out" / "report.txt").exists()
    assert (tmp_path / "out" / "cnn_models.pt").exists()
    assert (tmp_path / "fig" / "03_cnn_precision_recall.png").exists()
