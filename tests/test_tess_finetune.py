"""Fine-tuning the Kepler models on TOI labels, offline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from transitml import tess_finetune as tf
from transitml.kepler_dr25 import SCALAR_NAMES, TrainingSet
from transitml.views import VIEW_NAMES


def tiny_set(n=40, seed=0, global_bins=2001, local_bins=201):
    rng = np.random.default_rng(seed)
    labels = np.array([i % 3 - 1 for i in range(n)])  # -1, 0, 1: open, FP, planet
    local = rng.normal(0, 0.1, (n, local_bins)).astype(np.float32)
    local[labels == 1, 95:106] -= 1.0
    views = {k: rng.normal(0, 0.1, (n, local_bins)).astype(np.float32) for k in VIEW_NAMES}
    views["local"] = local
    views["global"] = rng.normal(0, 0.1, (n, global_bins)).astype(np.float32)
    return TrainingSet(
        tce_ids=np.array([f"{i}.01" for i in range(n)]),
        kepids=np.arange(n, dtype=np.int64) // 2,
        classes=np.array(["PC"] * n),
        labels=labels.astype(np.int64),
        views=views,
        scalars={k: np.full(n, 3.0) for k in SCALAR_NAMES},
    )


def test_open_candidates_and_test_stars_are_dropped():
    ts = tiny_set()
    kept = tf.drop_stars(tf.labelled(ts), np.array([0, 1]))
    assert (kept.labels >= 0).all()
    assert not np.isin(kept.kepids, [0, 1]).any()
    assert len(kept) == sum(1 for i in range(4, 40) if i % 3 != 0)


def test_tess_rows_weigh_as_much_as_kepler():
    assert tf.tess_weight(34_032, 1_000) == pytest.approx(34.032)
    assert tf.tess_weight(10, 0) == 10


def test_compare_all_pairs_and_operating_point():
    rng = np.random.default_rng(1)
    stars, fine, kep = [], {}, {}
    for tic in range(80):
        label = tic % 2
        stars.append({"target_id": f"TIC {tic}", "disposition": "KP" if label else "FA",
                      "model_score": float(rng.random()), "period_recovered": tic % 4 != 0})
        fine[tic] = 0.3 + 0.4 * label + 0.2 * rng.random()
        kep[tic] = rng.random()
    out = tf.compare_all({"finetuned_cnn": fine, "kepler_cnn": kep}, {"toi_trained": stars})
    assert out["n_stars"] == 80 and out["n_planets"] == 40
    assert out["average_precision"]["finetuned_cnn"] == pytest.approx(1.0)
    d, lo, hi = out["differences"]["finetuned_cnn - kepler_cnn"]
    assert 0 < lo <= d <= hi
    assert "finetuned_cnn - tess_cnn" not in out["differences"]  # absent models are skipped
    assert out["operating_point"]["planets_kept"] == 1.0
    assert out["operating_point"]["false_positives_rejected"] == 1.0
    result = tf.FinetuneResult(
        n_train_tois=10, n_train_stars=8, n_train_planets=5, epochs={"finetuned_cnn": [3]},
        tess_weight=34.0, **out,
    )
    report = tf.format_finetune_report(result)
    assert "fine-tuned on TESS TOIs" in report and "found the TOI period" in report
    json.dumps(result.to_dict(), default=str)


def test_fine_tuning_starts_from_the_given_weights():
    pytest.importorskip("torch")
    from transitml.cnn import CNNConfig, build_network

    ts = tf.labelled(tiny_set(60))
    config = CNNConfig(max_epochs=1, patience=1, learning_rate=0.0, n_models=1,
                       global_filters=(4,), local_filters=(4,), dense=8)
    start = build_network(config, 2001, 201).state_dict()
    nets, epochs = tf.train_cnns(ts, config, seed=0, init_states=[start])
    assert epochs == [1]
    for k, v in nets[0].state_dict().items():
        assert np.allclose(v.numpy(), start[k].numpy())  # zero learning rate: weights untouched
    scores = tf.predict_cnns(nets, ts)
    assert scores.shape == (len(ts),) and ((scores >= 0) & (scores <= 1)).all()
