"""Turn any :class:`LightCurveSource` into a labelled feature matrix.

This is the only place the pipeline touches a data source.  Swap the source and
everything downstream is unchanged -- see :mod:`transitml.data.mast`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from ..config import BLSConfig, Config, PreprocessConfig
from ..features import FEATURE_NAMES, detrend_and_search, extract_features
from .base import LightCurve, LightCurveSource
from .synthetic import SyntheticTESSSource


@dataclass
class Dataset:
    """Feature matrix plus labels plus per-curve ground truth.

    ``meta`` carries the injected parameters for synthetic curves; the
    evaluation uses it to slice recall by true transit SNR and by negative
    subtype (eclipsing binary vs plain variable star).  It is never fed to the
    model -- see :meth:`X`.
    """

    features: pd.DataFrame
    labels: NDArray[np.int_]
    meta: pd.DataFrame

    def __post_init__(self) -> None:
        if len(self.features) != len(self.labels) or len(self.features) != len(self.meta):
            raise ValueError("features/labels/meta length mismatch")
        leaked = set(self.features.columns) & set(self.meta.columns)
        if leaked:
            raise ValueError(f"ground-truth columns leaked into features: {sorted(leaked)}")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def X(self) -> NDArray[np.float64]:
        """Model input: features only, in a fixed column order."""
        return self.features[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)

    @property
    def y(self) -> NDArray[np.int_]:
        return self.labels

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean())

    def save(self, path: str | Path) -> None:
        """Persist to a single ``.npz`` so a run can be re-analysed without re-running BLS."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features.to_numpy(dtype=np.float64),
            feature_names=np.array(self.features.columns, dtype=object),
            labels=self.labels,
            meta_values=self.meta.to_numpy(dtype=object),
            meta_names=np.array(self.meta.columns, dtype=object),
        )


def process_light_curve(
    lc: LightCurve, preprocess: PreprocessConfig, bls: BLSConfig
) -> tuple[dict[str, float], dict[str, Any]]:
    """Detrend one light curve and extract its features.

    Returns ``(features, meta)``.  Kept at module level (not a closure) so it
    pickles cleanly for :class:`joblib.Parallel`.
    """
    lc = lc.finite()
    flat, search = detrend_and_search(lc, preprocess, bls)
    feats = extract_features(flat, bls, search=search)
    meta = dict(lc.meta)
    if lc.label is not None:
        # A real curve's label lives on the curve, not in generator metadata.
        meta.setdefault("label", int(lc.label))
    meta["target_id"] = lc.target_id
    meta["n_cadences"] = lc.n_cadences
    meta["sigma_flat"] = flat.scatter
    return feats, meta


#: Ground-truth / provenance columns, kept strictly out of the feature matrix.
_META_COLUMNS: tuple[str, ...] = (
    "target_id",
    "kind",
    "label",
    "true_snr",
    "period",
    "epoch",
    "depth",
    "duration_t14",
    "duration_t23",
    "radius_ratio",
    "impact_parameter",
    "a_over_rs",
    "n_transits_in_window",
    "n_in_transit_cadences",
    "grazing",
    "secondary_depth",
    "odd_even_fraction",
    "tess_mag",
    "sigma_white",
    "sigma_flat",
    "r_star_rsun",
    "rho_star_cgs",
    "variability_amplitude",
    "variability_period",
    "red_rms",
    "red_alpha",
    "n_cadences",
)


def build_dataset(
    source: LightCurveSource,
    *,
    preprocess: PreprocessConfig,
    bls: BLSConfig,
    n_jobs: int = -1,
    verbose: int = 0,
) -> Dataset:
    """Run the full detrend + feature-extraction pass over a source.

    Parameters
    ----------
    source:
        Any :class:`LightCurveSource`.  Iterated lazily.
    n_jobs:
        Passed to joblib.  BLS is the bottleneck and is embarrassingly parallel
        across light curves.
    """
    results = Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=16)(
        delayed(process_light_curve)(lc, preprocess, bls) for lc in source
    )
    feature_rows = [r[0] for r in results]
    meta_rows = [r[1] for r in results]
    labels = np.array(
        [1 if m.get("kind") == "planet" else int(m.get("label", 0) or 0) for m in meta_rows],
        dtype=int,
    )
    for row, label in zip(meta_rows, labels):
        row["label"] = int(label)

    features = pd.DataFrame(feature_rows)[list(FEATURE_NAMES)]
    meta = pd.DataFrame(meta_rows)
    for column in _META_COLUMNS:
        if column not in meta.columns:
            meta[column] = np.nan
    meta = meta[list(_META_COLUMNS)]
    return Dataset(features=features, labels=labels, meta=meta)


def build_default_dataset(config: Config, n_jobs: int = -1, verbose: int = 0) -> Dataset:
    """Build the demo dataset described by ``config`` from synthetic light curves.

    To run on real photometry instead, replace the two lines below with a
    :class:`~transitml.data.mast.MASTLightCurveSource`; nothing else changes.
    """
    source = SyntheticTESSSource(
        n_curves=config.dataset.n_curves,
        positive_rate=config.dataset.positive_rate,
        eclipsing_binary_rate=config.dataset.eclipsing_binary_rate,
        seed=config.seed,
        survey=config.survey,
        noise=config.noise,
        star=config.star,
        planet=config.planet,
        eb=config.eb,
    )
    return build_dataset(
        source, preprocess=config.preprocess, bls=config.bls, n_jobs=n_jobs, verbose=verbose
    )


def sample_light_curves(config: Config, indices: Sequence[int]) -> list[LightCurve]:
    """Regenerate specific synthetic light curves (for the example figures)."""
    source = SyntheticTESSSource(
        n_curves=config.dataset.n_curves,
        positive_rate=config.dataset.positive_rate,
        eclipsing_binary_rate=config.dataset.eclipsing_binary_rate,
        seed=config.seed,
        survey=config.survey,
        noise=config.noise,
        star=config.star,
        planet=config.planet,
        eb=config.eb,
    )
    return [source.generate(i) for i in indices]
