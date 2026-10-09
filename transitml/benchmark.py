"""Benchmark a trained model against real labels: TOI follow-up dispositions.

Everything else in this project scores the model on labels known by
construction, either fully synthetic or injected into real photometry.  This
module asks the question a user of the vetting tool actually cares about: on
real TESS signals that follow-up observers have since resolved, does the model
keep the planets and reject the false positives?

The model is never retrained here.  It is the one :func:`~transitml.model.train`
fitted on synthetic or injected data, with its operating threshold frozen on
that training split, applied unchanged to the TOI hosts.  Stars the model was
trained on are removed before scoring (:func:`~transitml.data.toi.select_benchmark_targets`).

Read the numbers with the caveats in :mod:`transitml.data.toi` in mind: every
star here is a TOI, so this measures vetting rather than detection, and the
positive rate is set by the catalogue rather than the sky.  For that reason
the report leads with two numbers that do not depend on the class mix:
**recall on confirmed planets** and **the fraction of known false positives
rejected**, both at the frozen threshold.  Average precision is reported too,
against its own chance level.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from .centroid import (
    CentroidConfig,
    centroid_test,
    combine_sector_tests,
    combined_pixel_files,
    sector_combination,
)
from .config import BLSConfig, PreprocessConfig
from .data.base import (
    LightCurve,
    LightCurveSource,
    bin_light_curve,
    stitch_light_curves,
)
from .data.injection import load_curves, save_curves, tic_number
from .data.loader import Dataset, build_dataset
from .data.tic import load_or_fetch_stars, read_star_table, with_star
from .data.toi import (
    FALSE_POSITIVE_REASONS,
    BenchmarkTarget,
    false_positive_reason,
    later_target,
)
from .data.tpf import (
    bin_target_pixels,
    download_tpfs,
    has_sky_matrix,
    load_tpf,
    save_tpf,
)
from .evaluate import (
    N_BOOTSTRAP,
    CurveScores,
    _permutation_importance,
    bootstrap_indices,
    bootstrap_win_rate,
    fast_average_precision,
    period_recovered,
    precision_at_k,
    score_curve,
)
from .features import FEATURE_NAMES
from .model import BASELINES, TrainedModel

#: Bins of the catalogue's TOI SNR.  That SNR comes from every sector the
#: discovery pipeline had, so a single sector here sees less than it says.
TOI_SNR_EDGES: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0, 1e9)

#: Bins of catalogued transit depth, in ppm.  If the planets kept and the false
#: positives kept rise together with depth, the model is ranking by signal
#: strength rather than telling the two apart.
TOI_DEPTH_EDGES_PPM: tuple[float, ...] = (0.0, 1000.0, 3000.0, 6000.0, 10000.0, 1e9)


class CurveListSource(LightCurveSource):
    """A fixed list of already-downloaded light curves."""

    def __init__(self, curves: Sequence[LightCurve], name: str = "TOI hosts") -> None:
        self.curves = list(curves)
        self._name = name

    def __len__(self) -> int:
        return len(self.curves)

    def __iter__(self) -> Iterator[LightCurve]:
        return iter(self.curves)

    @property
    def name(self) -> str:
        return self._name


def fetch_benchmark_curves(
    targets: Sequence[BenchmarkTarget],
    *,
    author: str = "TESS-SPOC",
    exposure_time: int | None = 1800,
    n_workers: int = 8,
) -> list[LightCurve]:
    """Download one light curve per target, in its assigned sector.

    Targets MAST has no matching curve for are skipped (the report counts
    them).  Each curve carries its star's label.  Full-frame-image curves of
    sectors after 26, exposed faster than ``exposure_time``, are fetched at
    their own cadence and averaged to it (:func:`fetch_exposure_seconds`).
    """
    from .data.mast import MASTLightCurveSource, fetch_exposure_seconds

    by_sector: dict[int, list[BenchmarkTarget]] = {}
    for target in targets:
        by_sector.setdefault(target.sector, []).append(target)

    curves: dict[str, LightCurve] = {}
    for sector in sorted(by_sector):
        fetch = fetch_exposure_seconds(author, exposure_time, sector)
        source = MASTLightCurveSource(
            [(t.target_id, t.label) for t in by_sector[sector]],
            mission="TESS",
            author=author,
            exposure_time=fetch,
            sector=sector,
            n_workers=n_workers,
        )
        for lc in source:
            if fetch != exposure_time and exposure_time is not None:
                lc = bin_light_curve(lc, exposure_time)
            curves.setdefault(lc.target_id, lc)
    return [curves[t.target_id] for t in targets if t.target_id in curves]


def load_or_fetch_curves(
    targets: Sequence[BenchmarkTarget],
    cache: str | Path,
    *,
    author: str = "TESS-SPOC",
    exposure_time: int | None = 1800,
    n_workers: int = 8,
) -> list[LightCurve]:
    """One curve per target, from ``cache`` where possible and MAST otherwise.

    Only targets the cache has never tried are downloaded, so a rerun with
    the same TOI table needs no network, and widening the sector list only
    fetches the new stars.  Targets MAST had nothing for are remembered in a
    ``.tried.json`` file beside the cache so they are not retried every run.
    Labels always come from ``targets``, so a disposition that changed since
    the download is honoured.
    """
    cache = Path(cache)
    tried_path = cache.with_suffix(".tried.json")
    cached = load_curves(cache) if cache.exists() else []
    tried = set(json.loads(tried_path.read_text())) if tried_path.exists() else set()

    def key(target_id: str, sector: Any) -> str:
        return f"{target_id}:{sector}"

    have = {key(lc.target_id, lc.meta.get("sector")): lc for lc in cached}
    tried |= set(have)
    missing = [t for t in targets if key(t.target_id, t.sector) not in tried]
    if missing:
        # A star asked for in several sectors is fetched in rounds, one sector
        # of it per round, since a fetch returns one curve per star.
        rounds: list[list[BenchmarkTarget]] = []
        asked: Counter[str] = Counter()
        for target in missing:
            n = asked[target.target_id]
            if n == len(rounds):
                rounds.append([])
            rounds[n].append(target)
            asked[target.target_id] += 1
        for batch in rounds:
            fetched = fetch_benchmark_curves(
                batch, author=author, exposure_time=exposure_time, n_workers=n_workers
            )
            # Keyed by the sector asked for: that is the sector MAST was searched in.
            sector_of = {t.target_id: t.sector for t in batch}
            for lc in fetched:
                have[key(lc.target_id, sector_of[lc.target_id])] = lc
        tried |= {key(t.target_id, t.sector) for t in missing}
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_curves(list(have.values()), cache)
        tried_path.write_text(json.dumps(sorted(tried)))

    out: list[LightCurve] = []
    for target in targets:
        lc = have.get(key(target.target_id, target.sector))
        if lc is not None:
            out.append(replace(lc, label=target.label))
    return out


def load_or_fetch_stitched(
    targets: Sequence[BenchmarkTarget],
    sectors: Sequence[int],
    cache: str | Path,
    *,
    author: str = "TESS-SPOC",
    exposure_time: int | None = 1800,
    n_workers: int = 8,
) -> tuple[list[BenchmarkTarget], list[LightCurve]]:
    """Every one of ``sectors`` each target was observed in, joined into one curve.

    Each star's curves come from ``cache`` or MAST as :func:`load_or_fetch_curves`
    gets them, one per sector, and are joined with
    :func:`~transitml.data.base.stitch_light_curves` (each sector normalised to
    its own median, the gaps between sectors kept).  Returns the targets, each
    moved to the first of its sectors that had a curve (whose pixel file the
    centroid test then reads; unchanged when none had one), and one curve per
    star that had any, both in the order given.
    """
    order = {s: i for i, s in enumerate(sectors)}
    per_sector = [
        replace(target, sector=s)
        for target in targets
        for s in sorted(
            {s for toi in target.tois for s in toi.sectors if s in order}, key=order.get
        )
    ]
    found: dict[str, list[LightCurve]] = {}
    for lc in load_or_fetch_curves(
        per_sector, cache, author=author, exposure_time=exposure_time, n_workers=n_workers
    ):
        found.setdefault(lc.target_id, []).append(lc)
    moved: list[BenchmarkTarget] = []
    curves: list[LightCurve] = []
    for target in targets:
        star = found.get(target.target_id)
        if not star:
            moved.append(target)
            continue
        first = min(star, key=lambda lc: order[int(lc.meta["sector"])])
        moved.append(replace(target, sector=int(first.meta["sector"])))
        joined = stitch_light_curves(star) if len(star) > 1 else star[0]
        curves.append(replace(joined, meta={**joined.meta, "sector": first.meta["sector"]}))
    return moved, curves


def fetch_with_fallback(
    targets: Sequence[BenchmarkTarget],
    sectors: Sequence[int],
    fetch: Callable[[list[BenchmarkTarget]], list[LightCurve]],
) -> tuple[list[BenchmarkTarget], list[LightCurve]]:
    """Curves for ``targets``, each from the first of its ``sectors`` that has one.

    A target ``fetch`` finds no curve for is asked for again in the next of
    ``sectors`` it was observed in (:func:`~transitml.data.toi.later_target`),
    until one is found or its sectors run out.  This is for a training set,
    which loses nothing by taking a star from another of its sectors; a
    benchmark star keeps its first.  Returns the targets as last asked for (a
    moved one carries its new sector) and the curves found, both in the order
    given.
    """
    curves = fetch(list(targets))
    found = {lc.target_id: lc for lc in curves}
    current = {t.target_id: t for t in targets}
    missing = [t for t in targets if t.target_id not in found]
    while missing := [m for t in missing if (m := later_target(t, sectors)) is not None]:
        current.update((t.target_id, t) for t in missing)
        found.update((lc.target_id, lc) for lc in fetch(missing))
        missing = [t for t in missing if t.target_id not in found]
    last = [current[t.target_id] for t in targets]
    return last, [found[t.target_id] for t in last if t.target_id in found]


def with_tic_stars(curves: Sequence[LightCurve], path: str | Path) -> list[LightCurve]:
    """The curves with their hosts' TIC temperature and density in ``meta``.

    The secondary-eclipse test sizes its allowance for a planet's own
    occultation from them.  Stars come from the table at ``path``, and only
    those it lacks are looked up at MAST
    (:func:`~transitml.data.tic.load_or_fetch_stars`).  If that lookup fails,
    the stars the table already has are used and the rest go without, which
    leaves their secondary test as it was before the allowance.
    """
    tics = [tic for lc in curves if (tic := tic_number(lc.target_id)) is not None]
    try:
        stars = load_or_fetch_stars(tics, path)
    except Exception as exc:  # noqa: BLE001 - network and catalogue errors vary
        print(f"  TIC lookup failed ({exc}); using only the stars already in {path}")
        stars = read_star_table(path) if Path(path).exists() else {}
    return [with_star(lc, stars) for lc in curves]


def tpf_cache_path(cache_dir: str | Path, target_id: str, sector: int) -> Path:
    """Where one star's target pixel file for one sector is kept."""
    return Path(cache_dir) / f"{target_id.replace(' ', '_')}_s{int(sector):04d}.npz"


def _fetch_tpf(
    target_id: str, sector: int, author: str, exposure_time: int | None, dest: Path
) -> bool:
    """One star's pixels in one sector, averaged to ``exposure_time`` when exposed faster."""
    from .data.mast import fetch_exposure_seconds

    fetch = fetch_exposure_seconds(author, exposure_time, sector)
    tpfs = download_tpfs(target_id, author=author, exposure_time=fetch, sector=sector)
    if not tpfs:
        return False
    tpf = tpfs[0]
    if exposure_time is not None and fetch != exposure_time:
        tpf = bin_target_pixels(tpf, exposure_time)
    save_tpf(tpf, dest)
    return True


def load_or_fetch_tpfs(
    targets: Sequence[BenchmarkTarget],
    cache_dir: str | Path,
    *,
    author: str = "TESS-SPOC",
    exposure_time: int | None = 1800,
    n_workers: int = 8,
) -> dict[str, Path]:
    """Each target's pixel file in its benchmark sector, downloading only what is missing.

    Files are kept one per star and sector (:func:`tpf_cache_path`), so a
    rerun needs no network, and stars MAST had no file for are listed in
    ``tried.json`` beside them and not asked for again.  Returns
    ``{target_id: path}`` rather than the files: 750 stamps of 11 by 11
    pixels over a sector hold about 2 GB in memory.
    """
    found = _load_or_fetch_tpf_files(
        [(t.target_id, t.sector) for t in targets], cache_dir, author, exposure_time, n_workers
    )
    return {
        t.target_id: found[t.target_id, t.sector]
        for t in targets
        if (t.target_id, t.sector) in found
    }


def load_or_fetch_sector_tpfs(
    sectors: Mapping[str, Sequence[int]],
    cache_dir: str | Path,
    *,
    author: str = "TESS-SPOC",
    exposure_time: int | None = 1800,
    n_workers: int = 8,
    sky: bool = False,
) -> dict[str, list[Path]]:
    """Each star's pixel files in every one of its ``sectors``, downloading only what is missing.

    For stars searched on several sectors joined.  Kept and remembered as
    :func:`load_or_fetch_tpfs` keeps them; returns ``{target_id: [path, ...]}``
    in the order of each star's sectors, without the sectors MAST has no file
    for and the stars it has none for at all.  With ``sky``, a file cached
    before pixel files kept their orientation on the sky (format 1) is
    fetched again; if that fails it is kept as it is.
    """
    pairs = [(target_id, int(s)) for target_id, star in sectors.items() for s in star]
    found = _load_or_fetch_tpf_files(
        pairs, cache_dir, author, exposure_time, n_workers, refetch_without_sky=sky
    )
    out: dict[str, list[Path]] = {}
    for pair in pairs:
        if pair in found:
            out.setdefault(pair[0], []).append(found[pair])
    return out


def _load_or_fetch_tpf_files(
    pairs: Sequence[tuple[str, int]],
    cache_dir: str | Path,
    author: str,
    exposure_time: int | None,
    n_workers: int,
    refetch_without_sky: bool = False,
) -> dict[tuple[str, int], Path]:
    """``{(target_id, sector): path}`` for every pair with a pixel file, fetching the new ones."""
    cache_dir = Path(cache_dir)
    tried_path = cache_dir / "tried.json"
    tried = set(json.loads(tried_path.read_text())) if tried_path.exists() else set()
    paths = {pair: tpf_cache_path(cache_dir, *pair) for pair in pairs}
    missing = [
        pair
        for pair, path in paths.items()
        if (not path.exists() and f"{pair[0]}:{pair[1]}" not in tried)
        or (refetch_without_sky and path.exists() and not has_sky_matrix(path))
    ]
    if missing:
        cache_dir.mkdir(parents=True, exist_ok=True)
        args = (
            [target_id for target_id, _ in missing],
            [sector for _, sector in missing],
            repeat(author),
            repeat(exposure_time),
            [paths[pair] for pair in missing],
        )
        if n_workers <= 1:
            list(map(_fetch_tpf, *args))
        else:
            # Processes, not threads: lightkurve's FITS reading is not thread-safe.
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(_fetch_tpf, *args, chunksize=4))
        tried |= {f"{target_id}:{sector}" for target_id, sector in missing}
        tried_path.write_text(json.dumps(sorted(tried)))
    return {pair: path for pair, path in paths.items() if path.exists()}


#: What the benchmark keeps of each centroid test.
CENTROID_FIELDS: tuple[str, ...] = (
    "status",
    "significant",
    "offset_distance_pixels",
    "offset_arcsec",
    "offset_sigma",
    "difference_snr",
    "n_transits",
)


#: What :func:`~transitml.centroid.combine_sector_tests` needs of each sector's test.
SECTOR_CENTROID_FIELDS: tuple[str, ...] = (
    *CENTROID_FIELDS,
    "offset_error_pixels",
    "offset_sky_pixels",
    "offset_sky_covariance",
)


def _centroid_one(
    path: Path,
    period: float,
    epoch: float,
    duration: float,
    config: CentroidConfig | None,
    fields: tuple[str, ...] = CENTROID_FIELDS,
) -> dict[str, Any]:
    result = centroid_test(load_tpf(path), period, epoch, duration, config)
    return {name: getattr(result, name) for name in fields}


def centroid_tests(
    dataset: Dataset,
    tpf_paths: Mapping[str, Path | Sequence[Path]],
    *,
    config: CentroidConfig | None = None,
    n_jobs: int = -1,
    sky: bool = False,
) -> list[dict[str, Any] | None]:
    """The centroid test of every star in ``dataset``, on its own search ephemeris.

    The ephemeris is the BLS peak the model was scored on (the dataset's
    ``search_period``, ``search_epoch`` and ``search_duration``), not the catalogue's,
    so each star gets the test ``vet --centroids`` would give it.  ``None``
    for a star without a pixel file.  A star given a list of pixel files (one
    per sector, for a star searched on its sectors joined) is tested on each
    and the tests combined by :func:`~transitml.centroid.combine_sector_tests`
    (with ``sky``, as offsets on the sky), which also says how many sectors
    were tested and how many placed the dip.
    """
    meta = dataset.meta
    ids = meta["target_id"].astype(str).tolist()
    ephemeris = meta[["search_period", "search_epoch", "search_duration"]].to_numpy(dtype=float)
    jobs: list[tuple[int, Path, bool]] = []
    for i, target_id in enumerate(ids):
        paths = tpf_paths.get(target_id)
        if isinstance(paths, (str, Path)):
            jobs.append((i, Path(paths), False))
        elif paths is not None:
            jobs.extend((i, Path(path), True) for path in paths)
    done = Parallel(n_jobs=n_jobs)(
        delayed(_centroid_one)(
            path, *ephemeris[i], config, SECTOR_CENTROID_FIELDS if several else CENTROID_FIELDS
        )
        for i, path, several in jobs
    )
    out: list[dict[str, Any] | None] = [None] * len(ids)
    sectors: dict[int, list[dict[str, Any]]] = {}
    for (i, _, several), result in zip(jobs, done):
        if several:
            sectors.setdefault(i, []).append(result)
        else:
            out[i] = result
    for i, tests in sectors.items():
        out[i] = combine_sector_tests(tests, config, sky=sky)
    return out


#: The centroid test as model inputs, for a model trained with pixel features:
#: how many sigma and how many pixels the dip sits from the target, and how
#: clearly the difference image shows it.
CENTROID_FEATURE_NAMES: tuple[str, ...] = (
    "centroid_offset_sigma",
    "centroid_offset_pixels",
    "centroid_difference_snr",
)


def centroid_features(tests: Sequence[dict[str, Any] | None]) -> pd.DataFrame:
    """One row of :data:`CENTROID_FEATURE_NAMES` per :func:`centroid_tests` entry.

    The offset is NaN unless the test placed the dip (status ``"ok"``), and
    everything is NaN for a star without a pixel file; the classifier learns
    a direction for missing values rather than having them imputed.
    """
    rows = []
    for test in tests:
        placed = test is not None and test["status"] == "ok"
        rows.append(
            {
                "centroid_offset_sigma": test["offset_sigma"] if placed else np.nan,
                "centroid_offset_pixels": test["offset_distance_pixels"] if placed else np.nan,
                "centroid_difference_snr": test["difference_snr"] if test is not None else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=list(CENTROID_FEATURE_NAMES), dtype=float)


def with_centroid_features(dataset: Dataset, tests: Sequence[dict[str, Any] | None]) -> Dataset:
    """``dataset`` with the :data:`CENTROID_FEATURE_NAMES` columns added beside its features."""
    if len(tests) != len(dataset):
        raise ValueError(f"{len(tests)} centroid results for {len(dataset)} stars")
    extra = centroid_features(tests)
    extra.index = dataset.features.index
    return Dataset(
        features=pd.concat([dataset.features, extra], axis=1),
        labels=dataset.labels,
        meta=dataset.meta,
    )


#: Offset floors, in pixels, the report shows the flag rates at: the default
#: is :attr:`CentroidConfig.min_offset_pixels`, and these say how much it matters.
CENTROID_FLOORS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)


@dataclass
class CentroidVeto:
    """The centroid test on the benchmark stars, and the model with it as a veto.

    The veto ranks every star whose dip is flagged as off target below every
    star that is not, keeping the model's order within each group, so it can
    only move flagged stars down.  Stars without a pixel file, or whose dip
    the difference image does not detect, are never flagged.
    """

    min_offset_pixels: float
    n_with_pixels: int
    status: dict[str, int]
    #: ``{group: {"n", "with_pixels", "flagged"}}`` for planets, false
    #: positives, each disposition and each false-positive reason.
    flagged: dict[str, dict[str, int]]
    model: CurveScores
    ap_gain: float
    ap_gain_low: float
    ap_gain_high: float
    win_rate: float
    planet_recall: float
    false_positive_rejection: float
    #: Planets and false positives flagged at other offset floors.
    floors: list[dict[str, Any]]
    #: Pixel files tested, when each star's sectors were tested and combined.
    n_pixel_files: int | None = None
    #: How they were combined (``"stouffer"`` or ``"sky"``), when they were.
    combination: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_offset_pixels": self.min_offset_pixels,
            "n_with_pixels": self.n_with_pixels,
            **({"n_pixel_files": self.n_pixel_files} if self.n_pixel_files is not None else {}),
            **({"combination": self.combination} if self.combination is not None else {}),
            "status": self.status,
            "flagged": self.flagged,
            "model": self.model.to_dict(),
            "average_precision_gain": self.ap_gain,
            "average_precision_gain_ci68": [self.ap_gain_low, self.ap_gain_high],
            "bootstrap_win_rate_vs_model": self.win_rate,
            "planet_recall": self.planet_recall,
            "false_positive_rejection": self.false_positive_rejection,
            "flagged_by_floor": self.floors,
        }


def _centroid_veto(
    tests: Sequence[dict[str, Any] | None],
    y: NDArray[np.int_],
    model: CurveScores,
    kept: NDArray[np.bool_],
    dispositions: NDArray[np.str_],
    reasons: Sequence[str | None],
    resamples: NDArray[np.int_] | None,
    config: CentroidConfig,
) -> CentroidVeto:
    has = np.array([t is not None for t in tests])
    flagged = np.array([t is not None and bool(t["significant"]) for t in tests])
    conclusive = np.array([t is not None and t["status"] == "ok" for t in tests])
    sigma = np.array([t["offset_sigma"] if t else np.nan for t in tests], dtype=float)
    distance = np.array([t["offset_distance_pixels"] if t else np.nan for t in tests], dtype=float)
    planets, negatives = y == 1, y == 0
    scores = model.scores
    assert scores is not None

    def count(sel: NDArray[np.bool_]) -> dict[str, int]:
        return {
            "n": int(sel.sum()),
            "with_pixels": int((sel & has).sum()),
            "flagged": int((sel & flagged).sum()),
        }

    groups = {"planets (CP/KP)": planets, "false positives (FP/FA)": negatives}
    for d in sorted(set(dispositions.tolist())):
        groups[str(d)] = dispositions == d
    reason = np.array([r or "" for r in reasons])
    if any(reasons):
        for name in FALSE_POSITIVE_REASONS:
            groups[f"FP, {name}"] = (dispositions == "FP") & (reason == name)

    vetoed = np.where(flagged, scores - 1.0, scores)
    curve = score_curve(
        "gradient_boosting_centroid_veto",
        "the model, with stars whose dip is off target ranked last",
        y,
        vetoed,
        resamples,
    )
    gains = [
        fast_average_precision(y[idx], vetoed[idx]) - fast_average_precision(y[idx], scores[idx])
        for idx in (resamples if resamples is not None else [])
        if y[idx].sum() >= 2
    ]
    low, high = (float(v) for v in np.percentile(gains, [16, 84])) if gains else (np.nan, np.nan)
    win = (
        bootstrap_win_rate(y, vetoed, scores, resamples) if resamples is not None else float("nan")
    )
    floors = []
    for floor in CENTROID_FLOORS:
        at = conclusive & (sigma >= config.significance_sigma) & (distance >= floor)
        floors.append(
            {
                "min_offset_pixels": floor,
                "planets_flagged": _rate(at, planets & has),
                "false_positives_flagged": _rate(at, negatives & has),
            }
        )
    kept_after = kept & ~flagged
    return CentroidVeto(
        min_offset_pixels=config.min_offset_pixels,
        n_with_pixels=int(has.sum()),
        status=dict(Counter(str(t["status"]) for t in tests if t is not None)),
        flagged={name: count(sel) for name, sel in groups.items()},
        model=curve,
        ap_gain=curve.average_precision - model.average_precision,
        ap_gain_low=low,
        ap_gain_high=high,
        win_rate=win,
        planet_recall=_rate(kept_after, planets),
        false_positive_rejection=_rate(~kept_after, negatives),
        floors=floors,
        n_pixel_files=combined_pixel_files(tests),
        combination=sector_combination(tests),
    )


@dataclass
class BenchmarkResult:
    """Everything the TOI benchmark reports."""

    sectors: list[int]
    selection: dict[str, int]
    n_without_curve: int
    n_stars: int
    n_planets: int
    chance_average_precision: float
    model: CurveScores
    baselines: list[CurveScores]
    win_rate_vs_best_baseline: float
    threshold: float
    threshold_rule: str
    confusion: dict[str, int]
    planet_recall: float
    false_positive_rejection: float
    precision_at_catalogue_mix: float
    precision_at_k: dict[str, float]
    rejection_by_disposition: dict[str, dict[str, float]]
    search_recovery: dict[str, float]
    planet_recall_given_search: dict[str, float]
    recall_by_toi_snr: list[dict[str, Any]]
    by_toi_depth: list[dict[str, Any]]
    missed_planets: list[dict[str, Any]] = field(default_factory=list)
    accepted_false_positives: list[dict[str, Any]] = field(default_factory=list)
    labels: NDArray[np.int_] | None = None
    stars: list[dict[str, Any]] = field(default_factory=list)
    #: The centroid test and the model with it as a veto, when pixels were given.
    centroid: CentroidVeto | None = None
    #: The columns the model read: the light-curve features, plus the centroid
    #: test's for a model trained with pixel features.
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    #: Average precision lost when each feature is shuffled across these stars,
    #: when asked for (``importance_repeats``).
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    #: Each star searched on every one of ``sectors`` it was observed in,
    #: joined, rather than on the first.
    stitched: bool = False

    @property
    def positive_rate(self) -> float:
        return self.n_planets / self.n_stars if self.n_stars else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sectors": self.sectors,
            "stitched": self.stitched,
            "selection": self.selection,
            "n_without_curve": self.n_without_curve,
            "n_stars": self.n_stars,
            "n_planets": self.n_planets,
            "n_false_positives": self.n_stars - self.n_planets,
            "positive_rate": self.positive_rate,
            "chance_average_precision": self.chance_average_precision,
            "model": self.model.to_dict(),
            "baselines": [b.to_dict() for b in self.baselines],
            "bootstrap_win_rate_vs_best_baseline": self.win_rate_vs_best_baseline,
            "operating_point": {
                "threshold": self.threshold,
                "rule": self.threshold_rule,
                "planet_recall": self.planet_recall,
                "false_positive_rejection": self.false_positive_rejection,
                "precision_at_catalogue_mix": self.precision_at_catalogue_mix,
                "confusion_matrix": self.confusion,
            },
            "precision_at_k": self.precision_at_k,
            "rejection_by_disposition": self.rejection_by_disposition,
            "search_recovery": self.search_recovery,
            "planet_recall_given_search": self.planet_recall_given_search,
            "recall_by_toi_snr": self.recall_by_toi_snr,
            "by_toi_depth": self.by_toi_depth,
            "missed_planets": self.missed_planets,
            "accepted_false_positives": self.accepted_false_positives,
            **({"centroid_veto": self.centroid.to_dict()} if self.centroid else {}),
            **(
                {"model_features": self.feature_names}
                if tuple(self.feature_names) != FEATURE_NAMES
                else {}
            ),
            **(
                {"feature_importance": self.feature_importance}
                if self.feature_importance
                else {}
            ),
            "stars": self.stars,
        }


def _rate(mask: NDArray[np.bool_], within: NDArray[np.bool_]) -> float:
    n = int(within.sum())
    return float((mask & within).sum() / n) if n else float("nan")


def _row(
    target: BenchmarkTarget,
    score: float,
    bls_period: float,
    recovered: bool,
    features: Any,
) -> dict[str, Any]:
    ref = target.reference
    return {
        "odd_even_sigma": float(features["odd_even_sigma"]),
        "secondary_sigma": float(features["secondary_sigma"]),
        "red_noise_beta": float(features["red_noise_beta"]),
        "target_id": target.target_id,
        "toi": ref.toi,
        "disposition": ref.disposition,
        "sector": target.sector,
        "toi_period_days": ref.period,
        "toi_depth_ppm": ref.depth_ppm,
        "toi_snr": ref.snr,
        "tess_mag": ref.tess_mag,
        "bls_period_days": bls_period,
        "period_recovered": recovered,
        "model_score": score,
    }


def benchmark(
    dataset: Dataset,
    targets: Sequence[BenchmarkTarget],
    trained: TrainedModel,
    *,
    sectors: Sequence[int],
    selection: dict[str, int],
    n_without_curve: int = 0,
    top_k: int = 20,
    seed: int = 42,
    n_bootstrap: int = N_BOOTSTRAP,
    centroids: Sequence[dict[str, Any] | None] | None = None,
    centroid_config: CentroidConfig | None = None,
    comments: Mapping[str, str] | None = None,
    importance_repeats: int = 0,
    stitched: bool = False,
) -> BenchmarkResult:
    """Score the TOI hosts in ``dataset`` with ``trained`` at its frozen threshold.

    ``dataset`` rows are matched to ``targets`` by target ID, so the dataset
    may hold fewer stars than ``targets`` (those MAST had no curve for).  The
    model reads the columns it was trained on (``trained.feature_names``), so
    a model with pixel features needs a dataset with them
    (:func:`with_centroid_features`); the baselines read the light-curve ones.

    With ``centroids`` (one :func:`centroid_tests` entry per dataset row,
    run with ``centroid_config``), the result also scores the model with the
    centroid test as a veto.  With ``comments`` (``{TOI: ExoFOP comment}``),
    the false positives are also counted by the reason they were retired.
    With ``importance_repeats``, it also measures how much average precision
    each feature carries on these stars, by shuffling it that many times.
    """
    by_id = {t.target_id: t for t in targets}
    ids = dataset.meta["target_id"].astype(str).tolist()
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} curve(s) match no benchmark target, e.g. {missing[0]}")
    rows = [by_id[i] for i in ids]
    y = np.array([t.label for t in rows], dtype=int)
    if not np.array_equal(y, dataset.y):
        raise ValueError("dataset labels disagree with the TOI dispositions")
    if y.min() == y.max():
        raise ValueError("the benchmark needs both planets and false positives")

    X = dataset.X
    names = tuple(trained.feature_names)
    inputs = dataset.inputs(names)
    scores = trained.score(inputs)
    resamples = bootstrap_indices(len(y), n_bootstrap, seed) if n_bootstrap > 0 else None
    model_curve = score_curve(
        "gradient_boosting", "the trained model, unchanged", y, scores, resamples
    )
    baseline_curves = [
        score_curve(b.name, b.description, y, b.score(X), resamples) for b in BASELINES
    ]
    strongest = max(baseline_curves, key=lambda c: c.average_precision)
    win_rate = (
        bootstrap_win_rate(y, model_curve.scores, strongest.scores, resamples)
        if resamples is not None
        else float("nan")
    )

    kept = scores >= trained.threshold
    planets = y == 1
    negatives = ~planets
    tp = int((kept & planets).sum())
    fp = int((kept & negatives).sum())
    fn = int((~kept & planets).sum())
    tn = int((~kept & negatives).sum())

    bls_period = 10.0 ** dataset.features["log_period"].to_numpy(dtype=float)
    toi_period = np.array([t.reference.period for t in rows], dtype=float)
    recovered = period_recovered(bls_period, toi_period)
    has_period = np.isfinite(toi_period)

    dispositions = np.array([t.reference.disposition for t in rows])
    rejection = {
        str(d): {
            "n": int((dispositions == d).sum()),
            "rejected": _rate(~kept, (dispositions == d)),
        }
        for d in sorted(set(dispositions[negatives]))
    }

    reasons: list[str | None] = [
        false_positive_reason(comments.get(t.reference.toi, ""))
        if comments is not None and t.reference.disposition == "FP"
        else None
        for t in rows
    ]
    veto = None
    if centroids is not None:
        if len(centroids) != len(rows):
            raise ValueError(f"{len(centroids)} centroid results for {len(rows)} stars")
        veto = _centroid_veto(
            centroids,
            y,
            model_curve,
            kept,
            dispositions,
            reasons,
            resamples,
            centroid_config or CentroidConfig(),
        )

    toi_snr = np.array([t.reference.snr for t in rows], dtype=float)
    snr_rows: list[dict[str, Any]] = []
    for lo, hi in zip(TOI_SNR_EDGES[:-1], TOI_SNR_EDGES[1:]):
        sel = planets & (toi_snr >= lo) & (toi_snr < hi)
        snr_rows.append(
            {
                "snr_low": lo,
                "snr_high": hi,
                "n_planets": int(sel.sum()),
                "recall": _rate(kept, sel),
                "search_recovery": _rate(recovered, sel & has_period),
            }
        )

    toi_depth = np.array([t.reference.depth_ppm for t in rows], dtype=float)
    depth_rows: list[dict[str, Any]] = []
    for lo, hi in zip(TOI_DEPTH_EDGES_PPM[:-1], TOI_DEPTH_EDGES_PPM[1:]):
        in_bin = (toi_depth >= lo) & (toi_depth < hi)
        depth_rows.append(
            {
                "depth_low_ppm": lo,
                "depth_high_ppm": hi,
                "n_planets": int((in_bin & planets).sum()),
                "planets_kept": _rate(kept, in_bin & planets),
                "n_false_positives": int((in_bin & negatives).sum()),
                "false_positives_kept": _rate(kept, in_bin & negatives),
            }
        )

    feats = dataset.features

    def row(i: int) -> dict[str, Any]:
        out = _row(
            rows[i], float(scores[i]), float(bls_period[i]), bool(recovered[i]), feats.iloc[i]
        )
        if reasons[i] is not None:
            out["false_positive_reason"] = reasons[i]
        if centroids is not None:
            out["centroid"] = centroids[i]
        return out

    missed = [row(i) for i in np.flatnonzero(planets & ~kept)]
    missed.sort(key=lambda r: -np.nan_to_num(r["toi_snr"], nan=-1.0))
    accepted = [row(i) for i in np.flatnonzero(negatives & kept)]
    accepted.sort(key=lambda r: -r["model_score"])

    return BenchmarkResult(
        stitched=stitched,
        sectors=list(sectors),
        selection=dict(selection),
        n_without_curve=int(n_without_curve),
        n_stars=len(y),
        n_planets=int(planets.sum()),
        chance_average_precision=float(y.mean()),
        model=model_curve,
        baselines=baseline_curves,
        win_rate_vs_best_baseline=win_rate,
        threshold=trained.threshold,
        threshold_rule=trained.threshold_rule,
        confusion={
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        planet_recall=_rate(kept, planets),
        false_positive_rejection=_rate(~kept, negatives),
        precision_at_catalogue_mix=float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        precision_at_k={
            f"model_top{top_k}": precision_at_k(y, scores, top_k),
            **{f"{b.name}_top{top_k}": precision_at_k(y, b.score(X), top_k) for b in BASELINES},
        },
        rejection_by_disposition=rejection,
        search_recovery={
            "planets": _rate(recovered, planets & has_period),
            "false_positives": _rate(recovered, negatives & has_period),
        },
        planet_recall_given_search={
            "period_recovered": _rate(kept, planets & has_period & recovered),
            "period_not_recovered": _rate(kept, planets & has_period & ~recovered),
        },
        recall_by_toi_snr=snr_rows,
        by_toi_depth=depth_rows,
        missed_planets=missed,
        accepted_false_positives=accepted,
        labels=y,
        stars=[{**row(i), "kept": bool(kept[i])} for i in range(len(rows))],
        centroid=veto,
        feature_names=list(names),
        feature_importance=(
            _permutation_importance(
                trained, inputs, y, seed=seed, n_repeats=importance_repeats, feature_names=names
            )
            if importance_repeats > 0
            else []
        ),
    )


def build_benchmark_dataset(
    curves: Sequence[LightCurve],
    *,
    preprocess: PreprocessConfig,
    bls: BLSConfig,
    n_jobs: int = -1,
) -> Dataset:
    """Detrend, search and featurise the TOI host curves exactly as for training."""
    return build_dataset(CurveListSource(curves), preprocess=preprocess, bls=bls, n_jobs=n_jobs)


def _fmt(value: float, spec: str = ".3f") -> str:
    return format(value, spec) if np.isfinite(value) else "n/a"


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Human-readable summary of the TOI benchmark."""
    lines: list[str] = []
    add = lines.append
    sel = result.selection
    add("=" * 72)
    add("REAL-LABEL BENCHMARK: TOI HOSTS WITH FOLLOW-UP DISPOSITIONS")
    add("=" * 72)
    per_star = "every sector of each star, joined" if result.stitched else "one sector per star"
    add(f"sectors: {', '.join(str(s) for s in result.sectors)} ({per_star})")
    add(
        f"stars in TOI table: {sel.get('stars_in_table', 0)}   unlabelled (PC/APC): "
        f"{sel.get('unlabelled', 0)}   not observed in these sectors: "
        f"{sel.get('not_in_sectors', 0)}   in the training set: {sel.get('in_training_set', 0)}"
    )
    add(
        f"selected: {sel.get('selected', 0)}   no light curve at MAST: {result.n_without_curve}   "
        f"scored: {result.n_stars}"
    )
    add(
        f"scored stars: {result.n_planets} planets (CP/KP), "
        f"{result.n_stars - result.n_planets} false positives (FP/FA), "
        f"positive rate {result.positive_rate:.1%}"
    )
    extra = [name for name in result.feature_names if name not in FEATURE_NAMES]
    if extra:
        add(f"model inputs: the {len(FEATURE_NAMES)} light-curve features and {', '.join(extra)}")
    add("")
    add("At the frozen operating threshold (chosen on the training split only)")
    add("-" * 72)
    add(f"  threshold: {result.threshold:.4f}")
    add(f"  confirmed planets kept (recall):       {_fmt(result.planet_recall)}")
    add(f"  known false positives rejected:        {_fmt(result.false_positive_rejection)}")
    for disposition, row in result.rejection_by_disposition.items():
        add(f"    {disposition:<4s} n = {row['n']:<5d} rejected {_fmt(row['rejected'])}")
    add(
        f"  precision at this catalogue's mix:     {_fmt(result.precision_at_catalogue_mix)}"
        "   (not a survey precision; see below)"
    )
    c = result.confusion
    add("")
    add("  Confusion matrix (rows = disposition, cols = prediction)")
    add("                          pred: no planet    pred: planet")
    add(f"    FP / FA                  {c['true_negative']:>10d}      {c['false_positive']:>10d}")
    add(f"    CP / KP                  {c['false_negative']:>10d}      {c['true_positive']:>10d}")
    add("")
    add("Ranking (average precision; chance = positive rate)")
    add("-" * 72)
    add(f"  {'random ranking (chance)':<34s} AP = {result.chance_average_precision:.3f}")
    for b in result.baselines:
        add(
            f"  {'baseline: ' + b.name:<34s} AP = {b.average_precision:.3f}"
            f"  [{_fmt(b.ap_low)}, {_fmt(b.ap_high)}]   (ROC-AUC {b.roc_auc:.3f})"
        )
    m = result.model
    add(
        f"  {'model: gradient boosting':<34s} AP = {m.average_precision:.3f}"
        f"  [{_fmt(m.ap_low)}, {_fmt(m.ap_high)}]   (ROC-AUC {m.roc_auc:.3f})"
    )
    add("  (brackets are 68% bootstrap intervals)")
    add(
        f"  the model beats the best baseline in {_fmt(result.win_rate_vs_best_baseline, '.1%')}"
        " of paired bootstrap resamples"
    )
    for key, value in result.precision_at_k.items():
        add(f"  precision of {key:<30s} {_fmt(value)}")
    if result.centroid is not None:
        add("")
        _centroid_section(add, result)
    if result.feature_importance:
        add("")
        add("What the model leans on here: average precision lost when one feature is")
        add("shuffled across these stars (mean and spread of the shuffles)")
        add("-" * 72)
        for row in result.feature_importance[:10]:
            add(f"  {row['feature']:<30s} {row['importance']:+.3f} +/- {row['std']:.3f}")
    add("")
    add("Did the search find the catalogued signal?")
    add("-" * 72)
    add(
        "  BLS period matches the TOI period (or a low-order alias): planets "
        f"{_fmt(result.search_recovery['planets'], '.2f')}, false positives "
        f"{_fmt(result.search_recovery['false_positives'], '.2f')}"
    )
    given = result.planet_recall_given_search
    add(
        f"  planet recall when the period was found: {_fmt(given['period_recovered'], '.2f')}"
        f"   when it was not: {_fmt(given['period_not_recovered'], '.2f')}"
    )
    add("")
    scored = "these sectors are" if result.stitched else "one sector is"
    add(f"  recall by catalogue TOI SNR (all sectors; {scored} scored here)")
    add(f"  {'SNR bin':<16s}{'n':>5s}{'recall':>9s}{'search':>9s}")
    for row in result.recall_by_toi_snr:
        hi = "inf" if row["snr_high"] > 1e8 else f"{row['snr_high']:.0f}"
        label = f"{row['snr_low']:.0f} - {hi}"
        add(
            f"  {label:<16s}{row['n_planets']:>5d}"
            f"{_fmt(row['recall'], '9.2f')}{_fmt(row['search_recovery'], '9.2f')}"
        )
    add("")
    add("Kept at the threshold, by catalogued depth: does the model separate the")
    add("classes, or rank both by signal strength?")
    add(f"  {'depth (ppm)':<16s}{'planets':>8s}{'kept':>7s}{'FPs':>7s}{'kept':>7s}")
    for row in result.by_toi_depth:
        hi = "inf" if row["depth_high_ppm"] > 1e8 else f"{row['depth_high_ppm']:.0f}"
        label = f"{row['depth_low_ppm']:.0f} - {hi}"
        add(
            f"  {label:<16s}{row['n_planets']:>8d}{_fmt(row['planets_kept'], '7.2f')}"
            f"{row['n_false_positives']:>7d}{_fmt(row['false_positives_kept'], '7.2f')}"
        )
    add("")
    add("Highest-SNR confirmed planets the model rejected")
    add("-" * 72)
    _rows(add, result.missed_planets)
    add("")
    add("Highest-scoring false positives the model kept")
    add("-" * 72)
    _rows(add, result.accepted_false_positives)
    add("")
    add("Reading these numbers")
    add("-" * 72)
    add("  Every star here is a TOI: both classes already passed a TESS pipeline's")
    add("  detection and vetting, so the false positives are the hard ones. The")
    add("  catalogue sets the positive rate, so precision here is not survey")
    add("  precision; recall and false-positive rejection are the transferable numbers.")
    add("=" * 72)
    return "\n".join(lines)


def _centroid_section(add, result: BenchmarkResult) -> None:
    veto = result.centroid
    assert veto is not None
    add("Centroid veto: is the dip on the target?")
    add("-" * 72)
    status = ", ".join(f"{k} {v}" for k, v in sorted(veto.status.items()))
    add(f"  stars with a pixel file: {veto.n_with_pixels} of {result.n_stars}   ({status})")
    if veto.n_pixel_files is not None:
        add(
            f"  each tested in every sector it was joined from ({veto.n_pixel_files} pixel "
            "files) and the tests combined:"
        )
        if veto.combination == "sky":
            add(
                "  each sector's offset turned onto the sky with its file's WCS and the "
                "offsets added as"
            )
            add("  vectors (Stouffer's method in two dimensions), SNR in quadrature,")
        else:
            add(
                "  significance by Stouffer's method, offset length weighted by its error, "
                "SNR in quadrature,"
            )
        add("  a sector whose centroid falls outside its window left out")
    add(
        "  flagged as off target (difference-image centroid at least 3 sigma and "
        f"{veto.min_offset_pixels:g} pixel"
    )
    add("  from the catalogue position, on the search's own ephemeris):")
    for name, row in veto.flagged.items():
        share = row["flagged"] / row["with_pixels"] if row["with_pixels"] else float("nan")
        add(
            f"    {name:<26s} n = {row['n']:<4d} with pixels {row['with_pixels']:<4d} "
            f"flagged {row['flagged']:<4d} ({_fmt(share, '.0%')})"
        )
    m = veto.model
    add(
        f"  {'model with the centroid veto':<34s} AP = {m.average_precision:.3f}"
        f"  [{_fmt(m.ap_low)}, {_fmt(m.ap_high)}]   (ROC-AUC {m.roc_auc:.3f})"
    )
    add(
        f"  gain over the model alone: {veto.ap_gain:+.3f} [{veto.ap_gain_low:+.3f}, "
        f"{veto.ap_gain_high:+.3f}]; the veto wins in {_fmt(veto.win_rate, '.1%')} "
        "of paired resamples"
    )
    add(
        f"  at the frozen threshold: planets kept {_fmt(result.planet_recall)} -> "
        f"{_fmt(veto.planet_recall)}, false positives rejected "
        f"{_fmt(result.false_positive_rejection)} -> {_fmt(veto.false_positive_rejection)}"
    )
    add("  flagged at other offset floors (planets / false positives, of those with pixels):")
    for row in veto.floors:
        add(
            f"    {row['min_offset_pixels']:.2f} pixel   "
            f"{_fmt(row['planets_flagged'], '.1%')} / {_fmt(row['false_positives_flagged'], '.1%')}"
        )


def _rows(add, rows: list[dict[str, Any]], limit: int = 10) -> None:
    if not rows:
        add("  none")
        return
    for row in rows[:limit]:
        add(
            f"  {row['target_id']:<15s} TOI {row['toi']:<8s} {row['disposition']:<3s} "
            f"S{row['sector']:<3d} SNR {_fmt(row['toi_snr'], '6.1f')}  "
            f"P {_fmt(row['toi_period_days'], '7.2f')} d  BLS {_fmt(row['bls_period_days'], '7.2f')} d"
            f"  score {row['model_score']:.3f}"
        )
        add(
            f"  {'':<15s} odd/even {_fmt(row['odd_even_sigma'], '.1f')} sigma, "
            f"secondary {_fmt(row['secondary_sigma'], '.1f')} sigma, "
            f"red-noise beta {_fmt(row['red_noise_beta'], '.2f')}"
            + ("" if row["period_recovered"] else ", period not recovered")
        )
    if len(rows) > limit:
        add(f"  ... and {len(rows) - limit} more")


def plot_benchmark(result: BenchmarkResult, path: Path) -> Path:
    """Precision-recall on the TOI hosts, and where each class's scores fall."""
    import matplotlib.pyplot as plt

    from .plots import INK_SOFT, NEUTRAL, SERIES, _save, _style

    _style()
    fig, (ax_pr, ax_hist) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    entries = [(result.model, SERIES[0], "-", "gradient boosting")]
    if result.centroid is not None:
        entries.append((result.centroid.model, SERIES[0], ":", "with the centroid veto"))
    entries += [
        (b, colour, "-", f"baseline: {b.name}") for b, colour in zip(result.baselines, SERIES[1:])
    ]
    for curve, colour, style, label in entries:
        ax_pr.plot(
            curve.recall, curve.precision, lw=2.0, color=colour, ls=style,
            label=f"{label}  (AP = {curve.average_precision:.3f})",
        )
    chance = result.chance_average_precision
    ax_pr.axhline(chance, lw=1.4, ls="--", color=NEUTRAL)
    ax_pr.text(
        0.985, chance - 0.02, f"random ranking (AP = {chance:.3f})",
        ha="right", va="top", color=INK_SOFT, fontsize=8.5,
    )
    ax_pr.set_xlim(0, 1.02)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.set_xlabel("recall (fraction of confirmed planets kept)")
    ax_pr.set_ylabel("precision at this catalogue's mix")
    ax_pr.set_title(
        f"{result.n_stars} TOI hosts ({result.n_planets} CP/KP, "
        f"{result.n_stars - result.n_planets} FP/FA)",
        loc="left",
    )
    ax_pr.legend(loc="best")

    scores, labels = result.model.scores, result.labels
    if scores is not None and labels is not None:
        bins = np.linspace(0.0, 1.0, 26)
        for label, colour, name in ((1, SERIES[0], "CP / KP"), (0, SERIES[1], "FP / FA")):
            ax_hist.hist(
                scores[labels == label], bins=bins, color=colour,
                histtype="step", lw=2.0, label=name,
            )
        ax_hist.axvline(result.threshold, color=NEUTRAL, lw=1.4, ls="--")
        ax_hist.text(
            result.threshold, ax_hist.get_ylim()[1] * 0.55,
            f" threshold {result.threshold:.2f}",
            ha="left", va="top", color=INK_SOFT, fontsize=8.5,
        )
        ax_hist.set_xlabel("model score")
        ax_hist.set_ylabel("stars")
        ax_hist.set_title(
            f"kept {result.planet_recall:.0%} of planets, "
            f"rejected {result.false_positive_rejection:.0%} of false positives",
            loc="left",
        )
        ax_hist.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))
    fig.tight_layout()
    return _save(fig, path)
