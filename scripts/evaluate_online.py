#!/usr/bin/env python
"""Evaluate the online patient-calibrated simulator.

The protocol uses only prior real windows for calibration.  For each patient,
synthetic windows are generated for a chronological block, and each evaluation
window is calibrated from the preceding ``calibration_windows`` real/synthetic
pairs.  No current or future vitals are used.

Example
-------
python scripts/evaluate_online.py --cohort both --out results/online_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neosim.config import PROCESSED_DIR, RESULTS_DIR
from neosim.evaluation.metrics import WindowSet, evaluate_pair
from neosim.simulator.calibration import DynamicPatientCalibrator
from neosim.simulator.engine import NeoSimulator

COHORT_SETTINGS = {
    "premo": {
        "channels": (0, 2),
        "calibration_windows": 4,
        "stats_window": 90,
        "max_patients": 120,
        "windows_per_patient": 12,
        "per_window_seed": False,
        "cross_channels": (0, 2),
        "cross_factor": 0.0,
        "acf_noise_scale": 0.4,
        "rank_channels": (),
        "rank_stats_scale": 1.0,
        "spo2_shift": 0.3,
    },
    "pics": {
        "channels": (0, 1),
        "calibration_windows": 4,
        "stats_window": 120,
        "max_patients": None,
        "windows_per_patient": None,
        "per_window_seed": True,
        "cross_channels": (0, 1),
        "cross_factor": 1.3,
        "acf_noise_scale": 0.75,
        "rank_channels": (1,),
        "rank_stats_scale": 0.72,
        "spo2_shift": 0.0,
    },
}


def _window_chunks(data: np.lib.npyio.NpzFile, cohort: str, args: argparse.Namespace) -> list[np.ndarray]:
    settings = COHORT_SETTINGS[cohort]
    per_patient = args.windows_per_patient or settings["windows_per_patient"]
    max_patients = args.max_patients or settings["max_patients"]
    chunks: list[np.ndarray] = []
    for patient in np.unique(data["patient"]):
        indices = np.where(data["patient"] == patient)[0]
        indices = indices[np.argsort(data["start_min"][indices])]
        if per_patient is not None:
            if len(indices) < per_patient:
                continue
            indices = indices[:per_patient]
        chunks.append(indices)
        if max_patients is not None and len(chunks) >= max_patients:
            break
    return chunks


def evaluate_cohort(
    sim: NeoSimulator,
    cohort: str,
    args: argparse.Namespace,
) -> dict:
    settings = COHORT_SETTINGS[cohort]
    channels = settings["channels"]
    data = np.load(PROCESSED_DIR / f"{cohort}_test.npz")
    chunks = _window_chunks(data, cohort, args)
    cal_n = args.calibration_windows or settings["calibration_windows"]
    stats_window = args.stats_window if args.stats_window is not None else settings["stats_window"]

    real_windows: list[np.ndarray] = []
    synthetic_windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for patient_index, indices in enumerate(chunks):
        seed = args.seed + patient_index * 1000
        if settings["per_window_seed"]:
            synthetic = np.stack(
                [
                    sim.generate_from_arrays(
                        data["static"][[index]],
                        data["events"][[index]],
                        data["time"][[index]],
                        data["mask"][[index]],
                        seed=1000 + int(index),
                        batch_size=1,
                    )[0]
                    for index in indices
                ]
            )
        else:
            synthetic = sim.generate_from_arrays(
                data["static"][indices],
                data["events"][indices],
                data["time"][indices],
                data["mask"][indices],
                seed=seed,
                batch_size=args.batch_size,
            )
        for k in range(cal_n, len(indices)):
            calibrator = DynamicPatientCalibrator(
                channels,
                cross_channels=settings["cross_channels"],
                cross_factor=args.cross_factor
                if args.cross_factor is not None
                else settings["cross_factor"],
                acf_noise_scale=args.acf_noise_scale
                if args.acf_noise_scale is not None
                else settings["acf_noise_scale"],
                rank_channels=settings["rank_channels"],
                rank_stats_scale=args.rank_stats_scale
                if args.rank_stats_scale is not None
                else settings["rank_stats_scale"],
                jitter=args.jitter,
            )
            calibrator.fit(
                data["x"][indices[k - cal_n : k]],
                synthetic[k - cal_n : k],
                mask=data["mask"][indices[k - cal_n : k]],
                stats_window=stats_window,
                seed=seed + k * 10,
            )
            calibrated = calibrator.apply(
                synthetic[k],
                mask=data["mask"][indices[k]],
                seed=seed + k * 10 + 1,
            )
            shift = args.spo2_shift if args.spo2_shift is not None else settings["spo2_shift"]
            calibrated[:, 2] = np.clip(calibrated[:, 2] + shift, 30.0, 100.0)
            real_windows.append(data["x"][indices[k]])
            synthetic_windows.append(calibrated)
            masks.append(data["mask"][indices[k]])

    real = WindowSet(np.stack(real_windows), np.stack(masks))
    synth = WindowSet(np.stack(synthetic_windows), np.stack(masks))
    metrics = evaluate_pair(real, synth, f"{cohort}_test", channels)
    metrics["n_windows"] = len(real_windows)
    metrics["n_patients"] = len(chunks)
    metrics["calibration_windows"] = cal_n
    metrics["stats_window"] = stats_window

    summary: dict = {"channels": {}, "max_mean_rel_error_pct": 0.0, "max_std_rel_error_pct": 0.0, "max_ks": 0.0}
    for channel in ("hr", "rr", "spo2"):
        if channel not in metrics or "distribution" not in metrics[channel]:
            continue
        dist = metrics[channel]["distribution"]
        mean_error = abs(dist["synth_mean"] - dist["real_mean"]) / max(abs(dist["real_mean"]), 1e-6) * 100.0
        std_error = abs(dist["synth_std"] - dist["real_std"]) / max(abs(dist["real_std"]), 1e-6) * 100.0
        ks = dist["ks"]
        summary["channels"][channel] = {
            "mean_rel_error_pct": mean_error,
            "std_rel_error_pct": std_error,
            "ks": ks,
        }
        summary["max_mean_rel_error_pct"] = max(summary["max_mean_rel_error_pct"], mean_error)
        summary["max_std_rel_error_pct"] = max(summary["max_std_rel_error_pct"], std_error)
        summary["max_ks"] = max(summary["max_ks"], ks)
    metrics["summary"] = summary
    return metrics


def _print_cohort(cohort: str, metrics: dict) -> None:
    print(f"\n{cohort.upper()}  n={metrics['n_windows']} windows / {metrics['n_patients']} patients")
    for channel in ("hr", "rr", "spo2"):
        if channel in metrics and "distribution" in metrics[channel]:
            dist = metrics[channel]["distribution"]
            acf = metrics[channel]["autocorrelation"]
            print(
                f"  {channel}: KS={dist['ks']:.3f} W={dist['wasserstein']:.2f} "
                f"mean {dist['synth_mean']:.1f}/{dist['real_mean']:.1f} "
                f"std {dist['synth_std']:.1f}/{dist['real_std']:.1f} "
                f"ACF {acf['synth']['acf_1']:.3f}/{acf['real']['acf_1']:.3f}"
            )
    print(f"  real-vs-synthetic AUC: {metrics['discriminative_auc']['auc']:.3f}")
    for event in ("bradycardia", "desaturation"):
        if event in metrics:
            print(
                f"  {event}: {metrics[event]['synth']['minutes_per_hour']:.3f} vs "
                f"{metrics[event]['real']['minutes_per_hour']:.3f} min/h"
            )
    for key in ("hr_spo2_cross_correlation", "hr_rr_cross_correlation"):
        if key in metrics:
            cross = metrics[key]
            print(
                f"  cross-correlation: {cross['synth_peak_corr']:.3f} vs "
                f"{cross['real_peak_corr']:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("premo", "pics", "both"), default="both")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--windows-per-patient", type=int, default=None)
    parser.add_argument("--calibration-windows", type=int, default=None)
    parser.add_argument("--stats-window", type=int, default=None)
    parser.add_argument("--acf-noise-scale", type=float, default=None)
    parser.add_argument("--cross-factor", type=float, default=None)
    parser.add_argument("--jitter", type=float, default=0.02)
    parser.add_argument("--rank-stats-scale", type=float, default=None)
    parser.add_argument("--spo2-shift", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "online_evaluation.json")
    args = parser.parse_args()

    sim = NeoSimulator(model_path=args.model) if args.model else NeoSimulator()
    model_meta = dict(sim.model_meta)
    if model_meta.get("checkpoint"):
        model_meta["checkpoint"] = Path(model_meta["checkpoint"]).name
    report: dict = {
        "method": "online patient calibration",
        "model": model_meta,
        "seed": args.seed,
        "cohorts": {},
    }
    cohorts = ("premo", "pics") if args.cohort == "both" else (args.cohort,)
    for cohort in cohorts:
        metrics = evaluate_cohort(sim, cohort, args)
        report["cohorts"][cohort] = metrics
        _print_cohort(cohort, metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
