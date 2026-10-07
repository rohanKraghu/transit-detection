"""Kepler DR25: labelled threshold-crossing events and their light curves.

The Kepler pipeline's final run over all 17 quarters (Data Release 25) flagged
34,032 **threshold-crossing events** (TCEs): periodic dips strong enough to be
worth a look, each with a period, an epoch, a duration and a depth.  The
Robovetter (Thompson et al. 2018) then sorted every one of them, and the
sorted list is the largest uniformly labelled set of transit-like signals
there is.  That is what this module turns into a training set.

Labels
------
Every TCE gets one of three classes, the same three the Kepler team used:

=====  ===============================================  =====
class  meaning                                          label
=====  ===============================================  =====
PC     planet candidate: a KOI the Robovetter passed    1
AFP    astrophysical false positive: a KOI it failed    0
       for a reason other than "not transit-like"
       (eclipsing binary, centroid offset, ephemeris
       match)
NTP    non-transiting phenomenon: a KOI failed as not   0
       transit-like, or a TCE never made a KOI at all
       (instrumental noise, rolling band, variability)
=====  ===============================================  =====

The class comes from the DR25 KOI table, matched to the TCE on ``kepid`` and
``koi_tce_plnt_num``.  The Robovetter's own call (``koi_pdisposition``) is
used rather than the archive disposition, because it is the one made the same
way for every TCE; the archive disposition (``CONFIRMED`` for a planet a later
paper confirmed) is kept beside it as ``archive_disposition``.

Two caveats travel with these labels:

* **They are the Robovetter's output.**  A model trained on them learns to
  agree with the Robovetter, mistakes included.  DR24 had a hand-vetted
  training subset (``av_training_set``, the one AstroNet used); in DR25 that
  column is empty for all 34,032 rows, so there is no human-only subset to
  fall back on.
* **About one TCE in eight is a planet candidate.**  Most TCEs are
  instrumental: 26,973 of the 34,032 are NTP.  Average precision is reported
  against that chance level, not against 0.5.

Light curves
------------
Long-cadence (29.4 min) PDCSAP flux for each star, one FITS file per quarter,
straight from the MAST archive with the standard library and astropy; no
``lightkurve`` needed.  Each star is cached as one small ``.npz`` (time, flux,
error, quarter), so a rerun touches the network only for stars it has never
seen.
"""

from __future__ import annotations

import csv
import io
import json
import time as _time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import LightCurve, LightCurveSource

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"

TCE_TABLE = "q1_q17_dr25_tce"
KOI_TABLE = "q1_q17_dr25_koi"

#: Classes and the binary label each one maps to.
CLASSES: tuple[str, ...] = ("PC", "AFP", "NTP")
CLASS_LABEL: dict[str, int] = {"PC": 1, "AFP": 0, "NTP": 0}

#: Quality bits dropped on top of the cadences PDC already set to NaN:
#: attitude tweak (1), safe mode (2), coarse point (4), Earth point (8),
#: reaction-wheel desaturation (32) and manual exclude (128).  Each marks a
#: cadence where the spacecraft, not the star, moved the flux.
QUALITY_BITMASK = 1 | 2 | 4 | 8 | 32 | 128

_TCE_COLUMNS = (
    "kepid",
    "tce_plnt_num",
    "tce_period",
    "tce_time0bk",
    "tce_duration",
    "tce_depth",
    "tce_max_mult_ev",
    "tce_model_snr",
    "tce_num_transits",
    "tce_prad",
    "tce_impact",
)
_KOI_COLUMNS = (
    "kepid",
    "koi_tce_plnt_num",
    "kepoi_name",
    "koi_pdisposition",
    "koi_disposition",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_score",
)

#: Columns of the catalogue CSV written by :func:`write_catalogue`.
CATALOGUE_COLUMNS: tuple[str, ...] = (
    "kepid",
    "tce_plnt_num",
    "period_days",
    "epoch_bkjd",
    "duration_hours",
    "depth_ppm",
    "mes",
    "model_snr",
    "n_transits",
    "planet_radius_rearth",
    "impact",
    "kepoi_name",
    "robovetter_disposition",
    "archive_disposition",
    "fpflag_nt",
    "fpflag_ss",
    "fpflag_co",
    "fpflag_ec",
    "koi_score",
    "tce_class",
    "label",
)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
def _http(
    url: str,
    *,
    params: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
    timeout: float = 120.0,
    retries: int = 4,
) -> bytes:
    """GET (or POST when ``data`` is given) with exponential backoff."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body), timeout=timeout
            ) as response:
                return response.read()
        except OSError:
            if attempt == retries:
                raise
            _time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def query_tap(query: str, *, timeout: float = 300.0) -> list[dict[str, str]]:
    """Run an ADQL query against the NASA Exoplanet Archive; rows as dicts."""
    raw = _http(TAP_URL, params={"query": query, "format": "csv"}, timeout=timeout)
    return list(csv.DictReader(io.StringIO(raw.decode())))


def fetch_tables() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Download the DR25 TCE and KOI tables (the columns the catalogue uses)."""
    tces = query_tap(f"select {','.join(_TCE_COLUMNS)} from {TCE_TABLE}")
    kois = query_tap(f"select {','.join(_KOI_COLUMNS)} from {KOI_TABLE}")
    return tces, kois


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class KeplerTCE:
    """One DR25 threshold-crossing event and its label."""

    kepid: int
    tce_plnt_num: int
    period_days: float
    epoch_bkjd: float
    duration_hours: float
    depth_ppm: float
    tce_class: str
    mes: float = float("nan")
    model_snr: float = float("nan")
    n_transits: float = float("nan")
    planet_radius_rearth: float = float("nan")
    impact: float = float("nan")
    kepoi_name: str = ""
    robovetter_disposition: str = ""
    archive_disposition: str = ""
    fpflag_nt: int = 0
    fpflag_ss: int = 0
    fpflag_co: int = 0
    fpflag_ec: int = 0
    koi_score: float = float("nan")

    @property
    def label(self) -> int:
        return CLASS_LABEL[self.tce_class]

    @property
    def tce_id(self) -> str:
        return f"{self.kepid:09d}-{self.tce_plnt_num:02d}"

    @property
    def target_id(self) -> str:
        return f"KIC {self.kepid}"

    @property
    def duration_days(self) -> float:
        return self.duration_hours / 24.0

    def as_row(self) -> dict[str, Any]:
        row = {name: getattr(self, name) for name in CATALOGUE_COLUMNS if name != "label"}
        row["label"] = self.label
        return row


def _float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def _int(value: str | None, default: int = 0) -> int:
    number = _float(value)
    return int(number) if np.isfinite(number) else default


def classify(koi: dict[str, str] | None) -> str:
    """The class of a TCE, from its matched KOI row (``None`` when it has none)."""
    if koi is None:
        return "NTP"
    if koi.get("koi_pdisposition", "").strip().upper() == "CANDIDATE":
        return "PC"
    return "NTP" if _int(koi.get("koi_fpflag_nt")) else "AFP"


def build_catalogue(
    tces: Iterable[dict[str, str]], kois: Iterable[dict[str, str]]
) -> list[KeplerTCE]:
    """Join TCE rows to KOI rows and label every TCE.

    Raises
    ------
    ValueError
        If two KOIs claim the same TCE, which would make its label ambiguous.
    """
    by_tce: dict[tuple[int, int], dict[str, str]] = {}
    for koi in kois:
        key = (_int(koi["kepid"]), _int(koi["koi_tce_plnt_num"], -1))
        if key in by_tce:
            raise ValueError(f"two KOIs match TCE {key}")
        by_tce[key] = koi

    catalogue = []
    for row in tces:
        kepid, plnt = _int(row["kepid"]), _int(row["tce_plnt_num"])
        koi = by_tce.get((kepid, plnt))
        catalogue.append(
            KeplerTCE(
                kepid=kepid,
                tce_plnt_num=plnt,
                period_days=_float(row["tce_period"]),
                epoch_bkjd=_float(row["tce_time0bk"]),
                duration_hours=_float(row["tce_duration"]),
                depth_ppm=_float(row["tce_depth"]),
                mes=_float(row.get("tce_max_mult_ev")),
                model_snr=_float(row.get("tce_model_snr")),
                n_transits=_float(row.get("tce_num_transits")),
                planet_radius_rearth=_float(row.get("tce_prad")),
                impact=_float(row.get("tce_impact")),
                tce_class=classify(koi),
                kepoi_name=(koi or {}).get("kepoi_name", ""),
                robovetter_disposition=(koi or {}).get("koi_pdisposition", ""),
                archive_disposition=(koi or {}).get("koi_disposition", ""),
                fpflag_nt=_int((koi or {}).get("koi_fpflag_nt")),
                fpflag_ss=_int((koi or {}).get("koi_fpflag_ss")),
                fpflag_co=_int((koi or {}).get("koi_fpflag_co")),
                fpflag_ec=_int((koi or {}).get("koi_fpflag_ec")),
                koi_score=_float((koi or {}).get("koi_score")),
            )
        )
    catalogue.sort(key=lambda t: (t.kepid, t.tce_plnt_num))
    return catalogue


def write_catalogue(catalogue: Sequence[KeplerTCE], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOGUE_COLUMNS)
        writer.writeheader()
        for tce in catalogue:
            writer.writerow(
                {
                    k: (f"{v:.10g}" if isinstance(v, float) else v)
                    for k, v in tce.as_row().items()
                }
            )
    return path


def read_catalogue(path: str | Path) -> list[KeplerTCE]:
    """Read a catalogue CSV written by :func:`write_catalogue`."""
    out = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(
                KeplerTCE(
                    kepid=int(row["kepid"]),
                    tce_plnt_num=int(row["tce_plnt_num"]),
                    period_days=_float(row["period_days"]),
                    epoch_bkjd=_float(row["epoch_bkjd"]),
                    duration_hours=_float(row["duration_hours"]),
                    depth_ppm=_float(row["depth_ppm"]),
                    tce_class=row["tce_class"],
                    mes=_float(row["mes"]),
                    model_snr=_float(row["model_snr"]),
                    n_transits=_float(row["n_transits"]),
                    planet_radius_rearth=_float(row["planet_radius_rearth"]),
                    impact=_float(row["impact"]),
                    kepoi_name=row["kepoi_name"],
                    robovetter_disposition=row["robovetter_disposition"],
                    archive_disposition=row["archive_disposition"],
                    fpflag_nt=_int(row["fpflag_nt"]),
                    fpflag_ss=_int(row["fpflag_ss"]),
                    fpflag_co=_int(row["fpflag_co"]),
                    fpflag_ec=_int(row["fpflag_ec"]),
                    koi_score=_float(row["koi_score"]),
                )
            )
    return out


def load_or_fetch_catalogue(path: str | Path) -> list[KeplerTCE]:
    """The labelled catalogue from ``path``, downloading and writing it if absent."""
    path = Path(path)
    if path.exists():
        return read_catalogue(path)
    catalogue = build_catalogue(*fetch_tables())
    write_catalogue(catalogue, path)
    return catalogue


def class_counts(catalogue: Iterable[KeplerTCE]) -> dict[str, int]:
    counts = {name: 0 for name in CLASSES}
    for tce in catalogue:
        counts[tce.tce_class] += 1
    return counts


def stratified_sample(
    catalogue: Sequence[KeplerTCE], per_class: int, seed: int
) -> list[KeplerTCE]:
    """Up to ``per_class`` TCEs of each class, drawn at random, in catalogue order.

    Balanced classes make a small sample informative about the rare ones
    (there are 995 + 3,025 KOI false positives against 25,978 TCEs that never
    became KOIs).  Reproducible for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    chosen: list[KeplerTCE] = []
    for name in CLASSES:
        members = [t for t in catalogue if t.tce_class == name]
        order = rng.permutation(len(members))
        chosen.extend(members[i] for i in order[:per_class])
    chosen.sort(key=lambda t: (t.kepid, t.tce_plnt_num))
    return chosen


# --------------------------------------------------------------------------
# Light curves
# --------------------------------------------------------------------------
def _mast(request: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _http(MAST_INVOKE_URL, data={"request": json.dumps(request)}, timeout=120.0)
    reply = json.loads(raw)
    if reply.get("status") not in ("COMPLETE", None):
        raise RuntimeError(f"MAST: {reply.get('status')} {reply.get('msg')}")
    return list(reply.get("data", []))


def long_cadence_uris(kepid: int) -> list[str]:
    """MAST URIs of every long-cadence light-curve file of one star."""
    observations = _mast(
        {
            "service": "Mast.Caom.Filtered",
            "format": "json",
            "params": {
                "columns": "obsid,t_exptime",
                "filters": [
                    {"paramName": "obs_collection", "values": ["Kepler"]},
                    {"paramName": "target_name", "values": [f"kplr{kepid:09d}"]},
                ],
            },
        }
    )
    uris: list[str] = []
    for obs in observations:
        if float(obs.get("t_exptime") or 0) < 1000:  # short cadence
            continue
        products = _mast(
            {
                "service": "Mast.Caom.Products",
                "format": "json",
                "params": {"obsid": str(obs["obsid"])},
            }
        )
        uris.extend(
            p["dataURI"]
            for p in products
            if str(p.get("productFilename", "")).endswith("_llc.fits")
        )
    return sorted(set(uris))


def read_llc_fits(data: bytes | str | Path) -> dict[str, NDArray]:
    """Time (BKJD), PDCSAP flux and error, and quarter from one ``_llc.fits``.

    Cadences with non-finite time or flux, or a quality bit in
    :data:`QUALITY_BITMASK`, are dropped.
    """
    from astropy.io import fits

    source = io.BytesIO(data) if isinstance(data, bytes) else data
    with fits.open(source, memmap=False) as hdul:
        quarter = int(hdul[0].header.get("QUARTER", -1))
        table = hdul[1].data
        time = np.asarray(table["TIME"], dtype=np.float64)
        flux = np.asarray(table["PDCSAP_FLUX"], dtype=np.float64)
        flux_err = np.asarray(table["PDCSAP_FLUX_ERR"], dtype=np.float64)
        quality = np.asarray(table["SAP_QUALITY"], dtype=np.int64)
    keep = (
        np.isfinite(time)
        & np.isfinite(flux)
        & np.isfinite(flux_err)
        & ((quality & QUALITY_BITMASK) == 0)
    )
    return {
        "time": time[keep],
        "flux": flux[keep],
        "flux_err": flux_err[keep],
        "quarter": np.full(int(keep.sum()), quarter, dtype=np.int16),
    }


def download_star(kepid: int) -> dict[str, NDArray]:
    """Every long-cadence quarter of one star, each normalised to its median.

    Quarters are normalised separately because Kepler rolls every quarter and
    the star lands on a different CCD with a different aperture.
    """
    parts = []
    for uri in long_cadence_uris(kepid):
        quarter = read_llc_fits(_http(MAST_DOWNLOAD_URL, params={"uri": uri}))
        if quarter["flux"].size == 0:
            continue
        median = float(np.median(quarter["flux"]))
        if not np.isfinite(median) or median <= 0:
            continue
        quarter["flux"] = quarter["flux"] / median
        quarter["flux_err"] = quarter["flux_err"] / median
        parts.append(quarter)
    if not parts:
        return {
            "time": np.empty(0),
            "flux": np.empty(0),
            "flux_err": np.empty(0),
            "quarter": np.empty(0, dtype=np.int16),
        }
    merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    order = np.argsort(merged["time"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    # A cadence served twice (overlapping files) is kept once.
    keep = np.concatenate([[True], np.diff(merged["time"]) > 0])
    return {key: value[keep] for key, value in merged.items()}


def curve_path(cache_dir: str | Path, kepid: int) -> Path:
    return Path(cache_dir) / f"kplr{kepid:09d}.npz"


def load_or_download_star(kepid: int, cache_dir: str | Path) -> LightCurve | None:
    """One star's stitched curve, from the cache or MAST.  ``None`` if MAST has none.

    An empty download is cached too, so a star MAST has nothing for is not
    asked for again.
    """
    path = curve_path(cache_dir, kepid)
    if path.exists():
        with np.load(path) as stored:
            arrays = {key: stored[key] for key in stored.files}
    else:
        arrays = download_star(kepid)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            time=arrays["time"],
            flux=arrays["flux"].astype(np.float32),
            flux_err=arrays["flux_err"].astype(np.float32),
            quarter=arrays["quarter"],
        )
        tmp.replace(path)
    if arrays["time"].size == 0:
        return None
    return LightCurve(
        target_id=f"KIC {kepid}",
        time=np.asarray(arrays["time"], dtype=np.float64),
        flux=np.asarray(arrays["flux"], dtype=np.float64),
        flux_err=np.asarray(arrays["flux_err"], dtype=np.float64),
        meta={
            "kind": "real",
            "mission": "Kepler",
            "kepid": kepid,
            "n_quarters": int(np.unique(arrays["quarter"]).size),
        },
    )


class KeplerLightCurveSource(LightCurveSource):
    """Stitched long-cadence Kepler curves for a list of KIC IDs, cached on disk."""

    def __init__(
        self,
        kepids: Sequence[int],
        cache_dir: str | Path,
        labels: dict[int, int] | None = None,
    ) -> None:
        self.kepids = list(kepids)
        self.cache_dir = Path(cache_dir)
        self.labels = labels or {}

    def __len__(self) -> int:
        return len(self.kepids)

    def __iter__(self) -> Iterator[LightCurve]:
        for kepid in self.kepids:
            lc = load_or_download_star(kepid, self.cache_dir)
            if lc is not None:
                yield replace(lc, label=self.labels.get(kepid))

    @property
    def name(self) -> str:
        return "Kepler long cadence (MAST)"
