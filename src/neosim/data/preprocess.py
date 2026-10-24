"""Build training windows from PreMo, PICS and the legacy NICU simulator CSVs.

The neural generator works on 120-minute windows at one-minute resolution.  The
public datasets are heterogeneous:

* PreMo provides continuous HR and SpO2 for very-low-birth-weight infants plus
  REDCap clinical events.
* PICS provides ECG-derived HR and respiration-rate annotations for ten preterm
  infants.
* The legacy repository contains one eight-hour HR/RR trace (infant 2).

We deliberately keep that heterogeneity: missing channels are represented by a
mask and the single conditional model learns all three signals across datasets.
"""

from __future__ import annotations

import json
import math
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import median_filter
from tqdm import tqdm

from neosim.config import (
    EVENT_INDEX,
    EVENT_TYPES,
    PROCESSED_DIR,
    RAW_DIR,
    SIGNALS,
)

T_WINDOW = 120
STRIDE = 30
MIN_COVERAGE = 0.85
MAX_INTERP_MIN = 5

_PREMO_META: dict[int, PatientMeta] | None = None


@dataclass(slots=True)
class PatientMeta:
    patient_id: int
    ga_weeks: float
    birth_weight_g: float
    male: float
    site: int
    sepsis_times: tuple[float, ...] = ()
    nec_times: tuple[float, ...] = ()
    intubation_times: tuple[float, ...] = ()
    extubation_times: tuple[float, ...] = ()


# ---------------------------------------------------------------------------
# Metadata / REDCap events
# ---------------------------------------------------------------------------
def _us_to_minutes(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds() / 60.0)
    try:
        return float(value) / 60_000_000.0
    except (TypeError, ValueError):
        return None


def build_premo_metadata(root: Path = RAW_DIR / "premo") -> dict[int, PatientMeta]:
    redcap = root / "redcap"
    demo = pq.read_table(redcap / "redcap.Demographic and Perinatal data.parquet").to_pandas()
    demo = demo.drop_duplicates("subject_number_deid", keep="first")
    ga = demo["demograph_birth_ga_weeks"].fillna(30) * 7 + demo["demograph_birth_ga_days"].fillna(0)
    weight = demo["delivery_birth_weight"].fillna(demo["delivery_birth_weight"].median())
    meta: dict[int, PatientMeta] = {}
    for i, row in demo.iterrows():
        pid = int(row["subject_number_deid"])
        meta[pid] = PatientMeta(
            patient_id=pid,
            ga_weeks=float(ga.iloc[i] / 7.0),
            birth_weight_g=float(weight.iloc[i]),
            male=1.0 if str(row.get("demograph_gender", "")).lower() == "male" else 0.0,
            site=int(row.get("site_id", 0) or 0),
        )

    sep = pq.read_table(redcap / "redcap.Sepsis or NEC eval data.parquet").to_pandas()
    for pid, group in sep.groupby("subject_number_deid"):
        pid = int(pid)
        if pid not in meta:
            continue
        sepsis: list[float] = []
        nec: list[float] = []
        for _, row in group.iterrows():
            if (
                str(row.get("sep_ev1", "")).lower() == "yes"
                or str(row.get("culture_yn", "")).lower() == "yes"
            ):
                t = _us_to_minutes(row.get("culture_age_us"))
                if t is not None:
                    sepsis.append(t)
            if str(row.get("nec_surgery", "")).lower() == "yes":
                t = _us_to_minutes(row.get("nec_surg_age_us"))
                if t is not None:
                    nec.append(t)
        meta[pid].sepsis_times = tuple(sepsis)
        meta[pid].nec_times = tuple(nec)

    intub = pq.read_table(redcap / "redcap.Intubation eval data.parquet").to_pandas()
    for pid, group in intub.groupby("subject_number_deid"):
        pid = int(pid)
        if pid not in meta:
            continue
        ints: list[float] = []
        exts: list[float] = []
        for _, row in group.iterrows():
            if str(row.get("intubation_yn", "")).lower() == "yes":
                t = _us_to_minutes(row.get("intubation_age_us"))
                if t is not None:
                    ints.append(t)
                t = _us_to_minutes(row.get("extubation_age_us"))
                if t is not None:
                    exts.append(t)
        meta[pid].intubation_times = tuple(ints)
        meta[pid].extubation_times = tuple(exts)
    return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interp_limit(series: pd.Series, limit: int = MAX_INTERP_MIN) -> pd.Series:
    return series.interpolate(method="linear", limit=limit, limit_area="inside")


def _coverage(mask: np.ndarray, length: int) -> np.ndarray:
    if len(mask) < length:
        return np.empty(0, dtype=np.float32)
    kernel = np.ones(length, dtype=np.float32)
    return np.convolve(mask.astype(np.float32), kernel, mode="valid") / length


def _window_starts(valid: np.ndarray, max_windows: int, rng: np.random.Generator) -> np.ndarray:
    n = len(valid)
    if n < T_WINDOW:
        return np.empty(0, dtype=int)
    starts = np.arange(0, n - T_WINDOW + 1, STRIDE)
    candidates = starts[valid[starts]]
    if len(candidates) <= max_windows:
        return candidates
    return np.sort(rng.choice(candidates, size=max_windows, replace=False))


def _event_times_near(
    times: tuple[float, ...], start: float, end: float, before: float, after: float
) -> bool:
    return any(start - before <= t <= end + after for t in times)


def _static_vector(
    ga_weeks: float,
    weight_g: float,
    age_days: float,
    male: float,
    dataset_pics: float,
    brady: bool,
    tachy: bool,
    apnea: bool,
    rds: bool,
    sepsis: bool,
    nec: bool,
    intubation: bool,
    severity: float,
) -> np.ndarray:
    return np.array(
        [
            ga_weeks,
            weight_g / 1000.0,
            math.log1p(max(age_days, 0.0)),
            male,
            dataset_pics,
            float(brady),
            float(tachy),
            float(apnea),
            float(rds),
            float(sepsis),
            float(nec),
            float(intubation),
            severity,
        ],
        dtype=np.float32,
    )


def _condition_label(brady: bool, desat: bool, apnea: bool, sepsis: bool, intubation: bool) -> int:
    if sepsis:
        return 4
    if intubation:
        return 5
    if apnea:
        return 3
    if brady:
        return 1
    if desat:
        return 2
    return 0


# ---------------------------------------------------------------------------
# PreMo
# ---------------------------------------------------------------------------
def _init_premo_worker(meta: dict[int, PatientMeta]) -> None:
    global _PREMO_META
    _PREMO_META = meta


def process_premo_file(path_str: str, max_windows: int, seed: int) -> dict[str, np.ndarray] | None:
    assert _PREMO_META is not None
    path = Path(path_str)
    pid = int(path.stem)
    meta = _PREMO_META.get(pid)
    if meta is None:
        return None
    table = pq.read_table(path, columns=["age_us", "HR", "SPO2"]).to_pandas()
    if len(table) < 100:
        return None
    age_series = table["age_us"]
    valid_age = age_series.notna()
    age_us = age_series[valid_age].to_numpy(dtype=np.int64, copy=False)
    if len(age_us) == 0:
        return None
    minute_raw = np.floor_divide(age_us, 60_000_000)
    # Some PreMo records use a negative or large absolute clock.  Shift to zero
    # so a sparse early timestamp does not allocate an enormous leading array.
    minute = minute_raw - minute_raw.min()
    table = table.loc[valid_age].assign(minute=minute)
    grouped = table.groupby("minute", sort=True)
    hr = grouped["HR"].mean()
    spo2 = grouped["SPO2"].mean()
    n_min = int(minute.max()) + 1
    if n_min <= 0 or n_min > 180 * 24 * 60:
        return None
    idx = np.arange(n_min, dtype=np.int64)
    hr = hr.reindex(idx)
    spo2 = spo2.reindex(idx)

    hr = hr.where((hr >= 30) & (hr <= 300))
    spo2 = spo2.where((spo2 >= 20) & (spo2 <= 100))
    hr = _interp_limit(hr)
    spo2 = _interp_limit(spo2)
    hr_np = np.array(hr.to_numpy(dtype=np.float32), copy=True)
    spo2_np = np.array(spo2.to_numpy(dtype=np.float32), copy=True)
    # A light median filter removes isolated monitor artefacts without erasing
    # the multi-minute bradycardia/desaturation events we want to model.
    hr_valid = np.isfinite(hr_np)
    spo2_valid = np.isfinite(spo2_np)
    if hr_valid.sum() > 5:
        hr_np[hr_valid] = median_filter(hr_np[hr_valid], size=3, mode="nearest")
    if spo2_valid.sum() > 5:
        spo2_np[spo2_valid] = median_filter(spo2_np[spo2_valid], size=3, mode="nearest")

    mask = np.zeros((n_min, len(SIGNALS)), dtype=np.float32)
    mask[:, 0] = np.isfinite(hr_np)
    mask[:, 2] = np.isfinite(spo2_np)
    x = np.zeros((n_min, len(SIGNALS)), dtype=np.float32)
    x[:, 0] = np.nan_to_num(hr_np, nan=0.0)
    x[:, 2] = np.nan_to_num(spo2_np, nan=0.0)

    events = np.zeros((n_min, len(EVENT_TYPES)), dtype=np.float32)
    events[hr_np < 100, EVENT_INDEX["bradycardia"]] = 1
    events[hr_np < 80, EVENT_INDEX["severe_bradycardia"]] = 1
    events[spo2_np < 85, EVENT_INDEX["desaturation"]] = 1
    events[spo2_np < 80, EVENT_INDEX["severe_desaturation"]] = 1
    events[hr_np > 200, EVENT_INDEX["tachycardia"]] = 1

    cov_hr = _coverage(mask[:, 0], T_WINDOW)
    cov_spo2 = _coverage(mask[:, 2], T_WINDOW)
    valid = (cov_hr >= MIN_COVERAGE) & (cov_spo2 >= MIN_COVERAGE)
    rng = np.random.default_rng(seed + pid)
    starts = _window_starts(valid, max_windows, rng)
    if len(starts) == 0:
        return None

    n = len(starts)
    xs = np.empty((n, T_WINDOW, len(SIGNALS)), dtype=np.float32)
    masks = np.empty((n, T_WINDOW, len(SIGNALS)), dtype=np.float32)
    evs = np.empty((n, T_WINDOW, len(EVENT_TYPES)), dtype=np.float32)
    stat = np.empty((n, 13), dtype=np.float32)
    times = np.empty((n, T_WINDOW, 1), dtype=np.float32)
    conds = np.empty(n, dtype=np.int64)
    for j, s in enumerate(starts):
        sl = slice(s, s + T_WINDOW)
        xs[j] = x[sl]
        masks[j] = mask[sl]
        evs[j] = events[sl]
        times[j, :, 0] = np.arange(T_WINDOW, dtype=np.float32) / 60.0 / 24.0

        brady = bool(evs[j, :, EVENT_INDEX["bradycardia"]].any())
        desat = bool(evs[j, :, EVENT_INDEX["desaturation"]].any())
        tachy = bool(evs[j, :, EVENT_INDEX["tachycardia"]].any())
        apnea = False
        start_min = float(s)
        end_min = float(s + T_WINDOW)
        sepsis = _event_times_near(meta.sepsis_times, start_min, end_min, 720, 2880)
        nec = _event_times_near(meta.nec_times, start_min, end_min, 720, 2880)
        intub = _event_times_near(meta.intubation_times, start_min, end_min, 1440, 4320)
        severity = float(
            np.clip(
                0.5 * evs[j, :, EVENT_INDEX["bradycardia"]].mean() * 10
                + 0.5 * evs[j, :, EVENT_INDEX["desaturation"]].mean() * 4,
                0,
                1,
            )
        )
        stat[j] = _static_vector(
            meta.ga_weeks,
            meta.birth_weight_g,
            start_min / 1440.0,
            meta.male,
            0.0,
            brady,
            tachy,
            apnea,
            intub,
            sepsis,
            nec,
            intub,
            severity,
        )
        conds[j] = _condition_label(brady, desat, apnea, sepsis, intub)

    return {
        "x": xs,
        "mask": masks,
        "events": evs,
        "static": stat,
        "time": times,
        "patient": np.full(n, pid, dtype=np.int64),
        "start_min": starts.astype(np.int64),
        "condition": conds,
    }


def preprocess_premo(
    max_windows: int = 32,
    workers: int = 8,
    limit: int | None = None,
    output_prefix: str = "premo",
) -> None:
    meta = build_premo_metadata()
    files = sorted((RAW_DIR / "premo" / "vitals").glob("*.parquet"))
    if limit:
        files = files[:limit]
    print(f"PreMo: processing {len(files)} patients with {workers} workers")
    splits: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_premo_worker, initargs=(meta,)) as pool:
        futures = {}
        for path in files:
            pid = int(path.stem)
            split = (
                "train"
                if zlib.crc32(str(pid).encode()) % 100 < 80
                else ("val" if zlib.crc32(str(pid).encode()) % 100 < 90 else "test")
            )
            n_windows = max_windows if split == "train" else max(1, max_windows // 2)
            futures[pool.submit(process_premo_file, str(path), n_windows, 1234)] = (split, path.name)
        failed = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc="PreMo"):
            split, name = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # one malformed record must not kill the run
                failed += 1
                if failed <= 5:
                    tqdm.write(f"  skipped {name}: {type(exc).__name__}: {exc}")
                continue
            if result is not None:
                splits[split].append(result)
        if failed:
            print(f"  skipped {failed} malformed/unreadable PreMo records")
    for split, parts in splits.items():
        if not parts:
            continue
        merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}
        out = PROCESSED_DIR / f"{output_prefix}_{split}.npz"
        np.savez_compressed(out, **merged)
        print(f"  {split}: {len(merged['x'])} windows -> {out}")


# ---------------------------------------------------------------------------
# PICS
# ---------------------------------------------------------------------------
#: The PICS release contains only cohort-level demographics.  These values are
#: sampled deterministically inside the published ranges and are used only as
#: weak conditioning; the model also receives a dataset indicator.
_PICS_META = {
    1: (31.0, 1.30),
    2: (30.0, 1.10),
    3: (31.5, 1.35),
    4: (32.6, 1.65),
    5: (30.7, 1.20),
    6: (33.2, 1.85),
    7: (29.9, 0.90),
    8: (34.3, 2.05),
    9: (31.9, 1.50),
    10: (32.0, 1.40),
}


def _parse_header(path: Path) -> tuple[float, int]:
    line = path.read_text().splitlines()[0].split()
    return float(line[2]), int(line[3])


def _refractory_peaks(samples: np.ndarray, fs: float, refractory_s: float = 0.8) -> np.ndarray:
    """Remove duplicate respiration detections closer than a physiological breath."""
    if len(samples) < 2:
        return samples
    keep = [int(samples[0])]
    for sample in samples[1:]:
        if (int(sample) - keep[-1]) / fs >= refractory_s:
            keep.append(int(sample))
    return np.asarray(keep, dtype=np.int64)


def _short_zero_runs_to_missing(values: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Keep short zero runs as apnea; treat long annotation gaps as missing."""
    out = values.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == 0:
            j = i
            while j < n and out[j] == 0:
                j += 1
            if j - i > max_gap:
                out[i:j] = np.nan
            i = j
        else:
            i += 1
    return out


def process_pics_infant(infant: int) -> dict[str, np.ndarray] | None:
    import wfdb

    root = RAW_DIR / "pics"
    ecg_hea = root / f"infant{infant}_ecg.hea"
    fs_ecg, n_ecg = _parse_header(ecg_hea)
    _, n_resp = _parse_header(root / f"infant{infant}_resp.hea")
    fs_resp_line = (root / f"infant{infant}_resp.hea").read_text().splitlines()[0].split()
    fs_resp = float(fs_resp_line[2])

    try:
        rpeaks = wfdb.rdann(str(root / f"infant{infant}_ecg"), "qrsc").sample
    except Exception:
        return None
    try:
        resp = wfdb.rdann(str(root / f"infant{infant}_resp"), "resp").sample
    except Exception:
        resp = np.empty(0, dtype=np.int64)

    duration_min = int(max(n_ecg / fs_ecg, n_resp / fs_resp) // 60) + 1
    hr_minute = np.full(duration_min, np.nan, dtype=np.float32)
    rr_minute = np.full(duration_min, np.nan, dtype=np.float32)

    if len(rpeaks) > 2:
        rr = np.diff(rpeaks) / fs_ecg
        good = (rr >= 0.30) & (rr <= 1.50)
        hr = 60.0 / np.clip(rr, 0.30, 1.50)
        minute = (rpeaks[1:] / fs_ecg // 60).astype(np.int64)
        minute = np.clip(minute, 0, duration_min - 1)
        for m in np.unique(minute):
            sel = (minute == m) & good
            if sel.any():
                hr_minute[m] = float(np.nanmedian(hr[sel]))

    if len(resp) > 1:
        resp = _refractory_peaks(resp, fs_resp)
        minute = (resp / fs_resp // 60).astype(np.int64)
        minute = np.clip(minute, 0, duration_min - 1)
        counts = np.bincount(minute, minlength=duration_min).astype(np.float32)
        rr_minute[: len(counts)] = counts
        rr_minute = _short_zero_runs_to_missing(rr_minute, max_gap=3)
        rr_minute[(rr_minute < 0) | (rr_minute > 100)] = np.nan

    hr_series = _interp_limit(pd.Series(hr_minute))
    rr_series = _interp_limit(pd.Series(rr_minute))
    hr_np = np.array(hr_series.to_numpy(dtype=np.float32), copy=True)
    rr_np = np.array(rr_series.to_numpy(dtype=np.float32), copy=True)
    hr_valid = np.isfinite(hr_np)
    rr_valid = np.isfinite(rr_np)
    if hr_valid.sum() > 5:
        hr_np[hr_valid] = median_filter(hr_np[hr_valid], size=3, mode="nearest")
    if rr_valid.sum() > 5:
        rr_np[rr_valid] = median_filter(rr_np[rr_valid], size=3, mode="nearest")

    n_min = len(hr_np)
    mask = np.zeros((n_min, len(SIGNALS)), dtype=np.float32)
    mask[:, 0] = np.isfinite(hr_np)
    mask[:, 1] = np.isfinite(rr_np)
    x = np.zeros((n_min, len(SIGNALS)), dtype=np.float32)
    x[:, 0] = np.nan_to_num(hr_np, nan=0.0)
    x[:, 1] = np.nan_to_num(rr_np, nan=0.0)
    events = np.zeros((n_min, len(EVENT_TYPES)), dtype=np.float32)
    events[hr_np < 100, EVENT_INDEX["bradycardia"]] = 1
    events[hr_np < 80, EVENT_INDEX["severe_bradycardia"]] = 1
    events[rr_np < 10, EVENT_INDEX["apnea"]] = 1
    events[hr_np > 200, EVENT_INDEX["tachycardia"]] = 1

    cov_hr = _coverage(mask[:, 0], T_WINDOW)
    cov_rr = _coverage(mask[:, 1], T_WINDOW)
    valid = (cov_hr >= MIN_COVERAGE) & (cov_rr >= MIN_COVERAGE)
    rng = np.random.default_rng(1000 + infant)
    starts = _window_starts(valid, 64, rng)
    if len(starts) == 0:
        return None
    n = len(starts)
    xs = np.empty((n, T_WINDOW, len(SIGNALS)), dtype=np.float32)
    masks = np.empty((n, T_WINDOW, len(SIGNALS)), dtype=np.float32)
    evs = np.empty((n, T_WINDOW, len(EVENT_TYPES)), dtype=np.float32)
    stat = np.empty((n, 13), dtype=np.float32)
    times = np.empty((n, T_WINDOW, 1), dtype=np.float32)
    conds = np.empty(n, dtype=np.int64)
    ga, weight = _PICS_META[infant]
    for j, s in enumerate(starts):
        sl = slice(s, s + T_WINDOW)
        xs[j] = x[sl]
        masks[j] = mask[sl]
        evs[j] = events[sl]
        times[j, :, 0] = np.arange(T_WINDOW, dtype=np.float32) / 60.0 / 24.0
        brady = bool(evs[j, :, EVENT_INDEX["bradycardia"]].any())
        apnea = bool(evs[j, :, EVENT_INDEX["apnea"]].any())
        tachy = bool(evs[j, :, EVENT_INDEX["tachycardia"]].any())
        severity = float(
            np.clip(
                0.4 * evs[j, :, EVENT_INDEX["bradycardia"]].mean() * 10
                + 0.6 * evs[j, :, EVENT_INDEX["apnea"]].mean() * 3,
                0,
                1,
            )
        )
        stat[j] = _static_vector(
            ga,
            weight * 1000,
            0.0,
            0.5,
            1.0,
            brady,
            tachy,
            apnea,
            False,
            False,
            False,
            False,
            severity,
        )
        conds[j] = _condition_label(brady, False, apnea, False, False)

    return {
        "x": xs,
        "mask": masks,
        "events": evs,
        "static": stat,
        "time": times,
        "patient": np.full(n, -infant, dtype=np.int64),
        "start_min": starts.astype(np.int64),
        "condition": conds,
    }


def preprocess_pics(max_windows: int = 64) -> None:
    print("PICS: processing 10 infants")
    splits: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    for infant in range(1, 11):
        result = process_pics_infant(infant)
        if result is None:
            print(f"  infant {infant}: no usable windows")
            continue
        split = "train" if infant <= 8 else ("val" if infant == 9 else "test")
        splits[split].append(result)
        print(f"  infant {infant}: {len(result['x'])} windows -> {split}")
    for split, parts in splits.items():
        if not parts:
            continue
        merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}
        out = PROCESSED_DIR / f"pics_{split}.npz"
        np.savez_compressed(out, **merged)
        print(f"  {split}: {len(merged['x'])} windows -> {out}")


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
def compute_norm_stats() -> dict:
    arrays = []
    for name in ("premo_train.npz", "pics_train.npz"):
        path = PROCESSED_DIR / name
        if path.exists():
            arrays.append(np.load(path))
    if not arrays:
        raise FileNotFoundError("no processed training data found")
    xs = np.concatenate([a["x"] for a in arrays], axis=0)
    ms = np.concatenate([a["mask"] for a in arrays], axis=0)
    statics = np.concatenate([a["static"] for a in arrays], axis=0)
    channel_stats = {}
    for i, name in enumerate(SIGNALS):
        values = xs[:, :, i][ms[:, :, i] > 0]
        channel_stats[name] = {
            "mean": float(values.mean()),
            "std": float(values.std() + 1e-6),
            "min": float(values.min()),
            "max": float(values.max()),
            "q01": float(np.quantile(values, 0.01)),
            "q99": float(np.quantile(values, 0.99)),
        }
    static_stats = {
        "mean": statics.mean(axis=0).tolist(),
        "std": (statics.std(axis=0) + 1e-6).tolist(),
    }
    stats = {"channels": channel_stats, "static": static_stats, "n_windows": int(len(xs))}
    path = PROCESSED_DIR / "norm_stats.json"
    path.write_text(json.dumps(stats, indent=2))
    print(f"Saved {path}")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["premo", "pics", "stats", "all"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=32)
    args = parser.parse_args()
    if args.dataset in ("pics", "all"):
        preprocess_pics()
    if args.dataset in ("premo", "all"):
        preprocess_premo(max_windows=args.max_windows, workers=args.workers, limit=args.limit)
    if args.dataset in ("stats", "all"):
        compute_norm_stats()
