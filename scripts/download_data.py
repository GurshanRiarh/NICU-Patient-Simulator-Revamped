#!/usr/bin/env python
"""Download the public datasets used to train and validate NeoSim 2.0.

Usage
-----
python scripts/download_data.py premo --workers 24
python scripts/download_data.py pics
python scripts/download_data.py legacy

PreMo is distributed by the University of Virginia Dataverse under the ODC-By
license.  Only the de-identified vital-sign parquet files and the REDCap event
tables are downloaded; the licence and README are stored in docs/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PREMO_INDEX = ROOT / "scripts" / "premo_file_index.json"
DATAVERSE = "https://dataverse.lib.virginia.edu/api/access/datafile/{file_id}"
PHYSIONET = "https://physionet.org/files/picsdb/1.0.0/{name}"

USER_AGENT = "NeoSim/2.0 (academic research; contact: GurshanRiarh@cmail.carleton.ca)"


def _download(url: str, dest: Path, expected: int | None = None, retries: int = 6) -> tuple[Path, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return dest, dest.stat().st_size
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                if expected is not None and tmp.stat().st_size != expected:
                    raise OSError(f"size mismatch: got {tmp.stat().st_size}, expected {expected}")
                tmp.replace(dest)
                return dest, dest.stat().st_size
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def download_premo(workers: int, limit: int | None) -> None:
    files = json.loads(PREMO_INDEX.read_text())
    if limit:
        files = files[:limit]
    root = RAW / "premo"
    jobs = []
    for file_id, name, size in files:
        if name.startswith("redcap."):
            dest = root / "redcap" / name
        elif name.endswith(".parquet"):
            dest = root / "vitals" / name
        else:
            continue
        jobs.append((DATAVERSE.format(file_id=file_id), dest, size))

    total = sum(j[2] for j in jobs)
    print(f"PreMo: {len(jobs)} files, {total / 1e9:.2f} GB -> {root}")
    done = 0
    downloaded = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download, url, dest, size) for url, dest, size in jobs]
        for fut in as_completed(futures):
            path, nbytes = fut.result()
            done += 1
            downloaded += nbytes
            if done % 100 == 0 or done == len(jobs):
                rate = downloaded / max(time.time() - t0, 1e-6) / 1e6
                print(
                    f"  [{done:>4}/{len(jobs)}] {downloaded / 1e9:6.2f} GB "
                    f"({rate:5.1f} MB/s) last={path.name}",
                    flush=True,
                )


def download_pics() -> None:
    root = RAW / "pics"
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(1, 11):
        names += [
            f"infant{i}_ecg.hea",
            f"infant{i}_ecg.atr",
            f"infant{i}_ecg.qrsc",
            f"infant{i}_resp.hea",
            f"infant{i}_resp.resp",
        ]
    names.append("RECORDS")
    print(f"PICS: {len(names)} annotation files -> {root}")
    for name in names:
        _download(PHYSIONET.format(name=name), root / name)
    print("PICS annotation download complete.")


def download_legacy() -> None:
    """The two CSVs from the original repository are already vendored."""
    root = ROOT / "data" / "legacy"
    expected = [
        "infant_2_8h_heart_rate_outlierRem.csv",
        "infant_2_8h_respiration_rate_outlierRem.csv",
    ]
    missing = [name for name in expected if not (root / name).exists()]
    if missing:
        print(f"Missing vendored legacy files: {missing}", file=sys.stderr)
        sys.exit(1)
    print("Legacy CSVs present.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["premo", "pics", "legacy", "all"])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="download only the first N PreMo files")
    args = parser.parse_args()
    if args.dataset in ("premo", "all"):
        download_premo(args.workers, args.limit)
    if args.dataset in ("pics", "all"):
        download_pics()
    if args.dataset in ("legacy", "all"):
        download_legacy()


if __name__ == "__main__":
    main()
