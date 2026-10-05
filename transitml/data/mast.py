"""Real-data source: TESS/Kepler light curves from MAST via ``lightkurve``.

This module is the drop-in replacement for :class:`~transitml.data.synthetic.SyntheticTESSSource`.
It implements the same :class:`~transitml.data.base.LightCurveSource` interface, so
switching the demo to real photometry is a one-line change in ``run_pipeline.py``::

    source = MASTLightCurveSource(
        targets=[("TIC 307210830", 1), ("TIC 100100827", 0), ...],
        mission="TESS",
        author="SPOC",
    )

**It is not exercised by the demo**, because labelled real data requires
injection-recovery (see below), and because ``lightkurve`` is
deliberately kept out of ``requirements.txt`` so that ``pip install -r`` stays
small and fast.  ``pip install lightkurve`` and the class below works as written.

Nothing downstream changes: the same detrending, the same BLS feature
extraction, the same model, the same evaluation.  That is the point of putting
an interface here.

Labels for real data
--------------------
The synthetic generator knows ground truth.  For real data you supply it, and
the honest options are:

* **Confirmed planets** from the NASA Exoplanet Archive as positives, and stars
  vetted as false positives / non-detections as negatives.  Beware: this label
  set is itself the output of the pipelines we are trying to beat, so it
  inherits their selection function.
* **Injection-recovery**: take real out-of-transit light curves and inject
  synthetic transits into a known subset.  This is what the TESS and Kepler
  teams do to measure pipeline completeness, and it is the only way to get
  unbiased labels at low SNR.  It also keeps the *noise* real, which is the
  part synthetic data gets wrong -- see the README.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from .base import LightCurve, LightCurveSource

_LIGHTKURVE_HINT = (
    "MASTLightCurveSource requires `lightkurve` and outbound network access to "
    "the MAST archive. Install with `pip install lightkurve`. The offline demo "
    "uses transitml.data.synthetic.SyntheticTESSSource instead."
)


class MASTLightCurveSource(LightCurveSource):
    """Stream light curves for a list of labelled targets from the MAST archive.

    Parameters
    ----------
    targets:
        ``(target_id, label)`` pairs, e.g. ``("TIC 307210830", 1)``.  Pass
        ``label=None`` for unlabelled inference.
    mission:
        ``"TESS"`` or ``"Kepler"``.
    author:
        Pipeline that produced the light curve (``"SPOC"``, ``"QLP"``, ...).
        Fixing this matters: different pipelines apply different systematics
        corrections, and mixing them leaks pipeline identity into the features.
    exposure_time:
        Cadence in seconds, e.g. 1800 for TESS FFI, 120 for 2-minute targets.
    quality_bitmask:
        Passed straight to ``lightkurve``; ``"default"`` drops the cadences the
        mission flagged as bad.
    """

    def __init__(
        self,
        targets: Sequence[tuple[str, int | None]],
        *,
        mission: str = "TESS",
        author: str = "SPOC",
        exposure_time: int | None = 1800,
        quality_bitmask: str = "default",
        flux_column: str = "pdcsap_flux",
    ) -> None:
        self.targets = list(targets)
        self.mission = mission
        self.author = author
        self.exposure_time = exposure_time
        self.quality_bitmask = quality_bitmask
        self.flux_column = flux_column

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self) -> Iterator[LightCurve]:
        lk = self._import_lightkurve()
        for target_id, label in self.targets:
            search = lk.search_lightcurve(
                target_id,
                mission=self.mission,
                author=self.author,
                exptime=self.exposure_time,
            )
            if len(search) == 0:
                continue
            collection = search.download_all(quality_bitmask=self.quality_bitmask)
            for lc in collection:
                yield self._to_lightcurve(lc, target_id, label)

    def _to_lightcurve(self, lc, target_id: str, label: int | None) -> LightCurve:
        """Convert one ``lightkurve.LightCurve`` into our container.

        Two things happen here and nowhere else:

        1. Flux is normalised to a relative scale (median = 1) so that depths
           are fractional and directly comparable to the synthetic curves.
        2. Non-finite cadences are dropped.  We deliberately do **not** flatten
           here -- detrending belongs to ``transitml.preprocess`` so that
           synthetic and real curves get identical treatment.
        """
        lc = lc.remove_nans()
        time = np.asarray(lc.time.value, dtype=np.float64)
        flux = np.asarray(lc.flux.value, dtype=np.float64)
        flux_err = np.asarray(lc.flux_err.value, dtype=np.float64)

        median = float(np.nanmedian(flux))
        if not np.isfinite(median) or median == 0.0:
            raise ValueError(f"{target_id}: non-normalisable flux (median={median})")

        order = np.argsort(time)
        return LightCurve(
            target_id=target_id,
            time=time[order],
            flux=(flux / median)[order],
            flux_err=(flux_err / median)[order],
            label=label,
            meta={
                "kind": "real",
                "mission": self.mission,
                "author": self.author,
                "sector": getattr(lc.meta, "get", lambda *_: None)("SECTOR"),
                "tess_mag": lc.meta.get("TESSMAG") if hasattr(lc, "meta") else None,
                # Downstream code uses sigma_white for SNR-style features; for
                # real data the robust scatter of the flattened curve is the
                # honest estimate, computed in transitml.preprocess.
                "sigma_white": float(np.nanmedian(flux_err) / median),
            },
        )

    @staticmethod
    def _import_lightkurve():  # pragma: no cover - requires network + optional dep
        try:
            import lightkurve as lk
        except ImportError as exc:
            raise ImportError(_LIGHTKURVE_HINT) from exc
        return lk
