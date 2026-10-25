#!/usr/bin/env python
"""Fit a validation-based affine calibration for the neural baseline.

The calibration is per cohort and per available channel.  It is fitted on
held-out validation windows and applied before care/event enforcement, so it
corrects baseline mean and spread without shrinking enforced clinical events.

Example
-------
python scripts/calibrate.py --n 100 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_real, sample_windows  # noqa: E402
from neosim.config import BASELINE_CALIBRATION_FILE  # noqa: E402
from neosim.evaluation.metrics import CHANNELS  # noqa: E402
from neosim.simulator.engine import NeoSimulator  # noqa: E402


def _stats(window_set, channel: int) -> tuple[float, float]:
    values = window_set.values(channel)
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=BASELINE_CALIBRATION_FILE)
    args = parser.parse_args()

    sim = NeoSimulator(model_path=args.model) if args.model else NeoSimulator()
    model_meta = dict(sim.model_meta)
    if model_meta.get("checkpoint"):
        model_meta["checkpoint"] = Path(model_meta["checkpoint"]).name
    calibration: dict = {
        "model": model_meta,
        "n": args.n,
        "seed": args.seed,
        "premo": {},
        "pics": {},
    }
    # PICS has only one validation infant, so a validation-fitted calibration
    # would not generalise.  PreMo has hundreds of validation patients.
    for cohort, channels in (("premo", (0, 2)),):
        real = load_real(f"{cohort}_val")
        synth = sample_windows(sim, cohort, args.n, args.seed)
        for channel in channels:
            real_mean, real_std = _stats(real, channel)
            synth_mean, synth_std = _stats(synth, channel)
            scale = float(np.clip(real_std / max(synth_std, 1e-6), 0.5, 1.5))
            shift = float(real_mean - scale * synth_mean)
            calibration[cohort][CHANNELS[channel]] = {
                "scale": scale,
                "shift": shift,
                "real_mean": real_mean,
                "real_std": real_std,
                "synth_mean": synth_mean,
                "synth_std": synth_std,
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calibration, indent=2))
    print(f"wrote {args.out}")
    for cohort in ("premo", "pics"):
        print(cohort)
        for name, values in calibration[cohort].items():
            print(
                f"  {name}: mean {values['synth_mean']:.1f}->{values['real_mean']:.1f}, "
                f"std {values['synth_std']:.1f}->{values['real_std']:.1f}, "
                f"scale={values['scale']:.3f}, shift={values['shift']:.2f}"
            )


if __name__ == "__main__":
    main()
