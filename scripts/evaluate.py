#!/usr/bin/env python
"""Evaluate NeoFlow against held-out PreMo and PICS windows.

Example
-------
python scripts/evaluate.py --n 80 --out results/evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from neosim.config import PROCESSED_DIR, RESULTS_DIR, SimulationConfig
from neosim.evaluation.metrics import WindowSet, evaluate_pair
from neosim.simulator.engine import NeoSimulator


def load_real(name: str) -> WindowSet:
    data = np.load(PROCESSED_DIR / f"{name}.npz")
    return WindowSet(data["x"], data["mask"])


def sample_windows(
    sim: NeoSimulator,
    cohort: str,
    n: int,
    seed: int,
    sampling: dict[str, float] | None = None,
) -> WindowSet:
    rng = np.random.default_rng(seed)
    conditions = [
        "stable_preterm",
        "bradycardia",
        "apnea_of_prematurity",
        "rds",
        "sepsis",
        "tachycardia",
    ]
    prevalence = [0.65, 0.08, 0.08, 0.08, 0.04, 0.07]
    xs = np.zeros((n, 120, 3), dtype=np.float32)
    masks = np.ones((n, 120, 3), dtype=np.float32)
    for i in tqdm(range(n), desc=f"synthesise {cohort}"):
        if cohort == "premo":
            ga = float(rng.uniform(24.0, 32.0))
            weight = float(rng.uniform(600.0, 1500.0))
            age = float(rng.uniform(0.0, 7.0))
        else:
            ga = float(rng.uniform(29.5, 34.0))
            weight = float(rng.uniform(900.0, 2050.0))
            age = 0.0
        config = SimulationConfig(
            patient_id=f"EVAL-{cohort}-{i:04d}",
            gestational_age_weeks=ga,
            birth_weight_g=weight,
            postnatal_age_days=age,
            sex="Male" if rng.random() > 0.5 else "Female",
            condition=str(rng.choice(conditions, p=prevalence)),
            severity=float(np.clip(rng.beta(1.5, 2.5), 0.05, 0.95)),
            shift_hours=2.0,
            interval_minutes=1.0,
            cohort=cohort,
            seed=int(rng.integers(0, 2**31 - 1)),
            care_events=False,
            closed_loop=False,
            **(sampling or {}),
        )
        result = sim.simulate(config)
        xs[i, :, 0] = result.arrays["hr"]
        xs[i, :, 1] = result.arrays["rr"]
        xs[i, :, 2] = result.arrays["spo2"]
    return WindowSet(xs, masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "evaluation.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--noise-phi", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--noise-phi-hr", type=float, default=None)
    parser.add_argument("--noise-phi-rr", type=float, default=None)
    parser.add_argument("--noise-phi-spo2", type=float, default=None)
    parser.add_argument("--noise-scale-hr", type=float, default=None)
    parser.add_argument("--noise-scale-rr", type=float, default=None)
    parser.add_argument("--noise-scale-spo2", type=float, default=None)
    args = parser.parse_args()

    sim = NeoSimulator(model_path=args.model) if args.model else NeoSimulator()
    report: dict = {"model": sim.model_meta, "n_per_dataset": args.n}

    real_premo = load_real("premo_test")
    real_pics = load_real("pics_test")
    sampling = {
        key: value
        for key, value in {
            "sampling_steps": args.sampling_steps,
            "cfg_scale": args.cfg_scale,
            "noise_phi": args.noise_phi,
            "noise_scale": args.noise_scale,
            "noise_phi_hr": args.noise_phi_hr,
            "noise_phi_rr": args.noise_phi_rr,
            "noise_phi_spo2": args.noise_phi_spo2,
            "noise_scale_hr": args.noise_scale_hr,
            "noise_scale_rr": args.noise_scale_rr,
            "noise_scale_spo2": args.noise_scale_spo2,
        }.items()
        if value is not None
    }
    synth_premo = sample_windows(sim, "premo", args.n, args.seed, sampling)
    synth_pics = sample_windows(sim, "pics", args.n, args.seed + 1, sampling)

    report["premo"] = evaluate_pair(real_premo, synth_premo, "premo_test", (0, 2))
    report["pics"] = evaluate_pair(real_pics, synth_pics, "pics_test", (0, 1))

    if not args.skip_save:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        np.savez_compressed(
            RESULTS_DIR / "synthetic_eval.npz",
            premo_x=synth_premo.x,
            premo_mask=synth_premo.mask,
            pics_x=synth_pics.x,
            pics_mask=synth_pics.mask,
        )
        print(f"wrote {args.out}")

    for dataset in ("premo", "pics"):
        print(f"\n{dataset.upper()}")
        for channel, metrics in report[dataset].items():
            if isinstance(metrics, dict) and "distribution" in metrics:
                dist = metrics["distribution"]
                print(
                    f"  {channel}: KS={dist['ks']:.3f} W={dist['wasserstein']:.2f} "
                    f"mean {dist['real_mean']:.1f}->{dist['synth_mean']:.1f}"
                )
        auc = report[dataset].get("discriminative_auc", {})
        dcr = report[dataset].get("distance_to_closest", {})
        print(f"  discriminative AUC: {auc.get('auc')}")
        print(f"  DCR ratio: {dcr.get('ratio')}")


if __name__ == "__main__":
    main()
