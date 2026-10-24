"""Command line interface for NeoSim 2.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neosim.config import DEFAULT_MODEL, OUTPUT_DIR, SimulationConfig
from neosim.simulator.engine import NeoSimulator


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--patient-id", default="SIM-0001")
    parser.add_argument("--name", default="Baby Doe")
    parser.add_argument("--ga", type=float, default=30.0, help="gestational age in weeks")
    parser.add_argument("--weight-g", type=float, default=1250.0)
    parser.add_argument("--age-days", type=float, default=14.0)
    parser.add_argument("--sex", default="Female", choices=["Female", "Male"])
    parser.add_argument("--condition", default="stable_preterm")
    parser.add_argument("--severity", type=float, default=0.35)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--start", default=None, help="HH:MM or HH:MM:SS")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cohort", default="auto", choices=["auto", "premo", "pics"])
    parser.add_argument("--interventions", nargs="*", default=None)
    parser.add_argument("--no-care-events", action="store_true")
    parser.add_argument("--no-closed-loop", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)


def _config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        patient_id=args.patient_id,
        name=args.name,
        gestational_age_weeks=args.ga,
        birth_weight_g=args.weight_g,
        postnatal_age_days=args.age_days,
        sex=args.sex,
        condition=args.condition,
        severity=args.severity,
        shift_hours=args.hours,
        interval_minutes=args.interval,
        start_time=args.start,
        seed=args.seed,
        cohort=args.cohort,
        interventions=args.interventions or [],
        care_events=not args.no_care_events,
        closed_loop=not args.no_closed_loop,
    )


def cmd_simulate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    sim = NeoSimulator(model_path=args.model, allow_fallback=True)
    result = sim.simulate(config)
    out_dir = (
        Path(args.out) if args.out else OUTPUT_DIR / f"{config.patient_id}_{result.times[0]:%Y%m%d_%H%M}"
    )
    formats = {"json", "csv", "fhir"} if args.format == "all" else {args.format}
    out_dir.mkdir(parents=True, exist_ok=True)
    if "json" in formats:
        payload = sim.to_legacy_json(result)
        (out_dir / "simulation.json").write_text(json.dumps(payload, indent=2))
    if "csv" in formats:
        (out_dir / "dashboard.json").write_text(
            json.dumps(
                {
                    "summary": result.summary(),
                    "times": [t.isoformat() for t in result.times],
                    "vitals": {k: v.tolist() for k, v in result.arrays.items()},
                    "events": result.events,
                    "interventions": result.interventions,
                    "pain": result.pain_labels,
                    "stability": result.stability.tolist(),
                    "sepsis_risk": result.sepsis_risk.tolist(),
                },
                indent=2,
            )
        )
        sim.to_csv(result, out_dir)
    if "fhir" in formats:
        (out_dir / "fhir_bundle.json").write_text(json.dumps(sim.to_fhir(result), indent=2))
    summary = result.summary()
    if args.json:
        print(json.dumps({"summary": summary, "path": str(out_dir)}, indent=2))
    else:
        print(
            f"{summary['patient_id']} | {summary['condition']} | {summary['shift_hours']} h | "
            f"HR {summary['hr_mean']:.0f} bpm | SpO2 {summary['spo2_mean']:.1f}% | "
            f"{summary['n_interventions']} interventions"
        )
        print(f"saved {', '.join(sorted(formats))} -> {out_dir}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("neosim.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB, sm{props.major}{props.minor})")
    else:
        print("GPU: unavailable")
    print(f"model: {args.model} ({'found' if args.model.exists() else 'not trained yet'})")
    print(f"torch: {torch.__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neosim", description="NeoSim 2.0 NICU patient simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="generate a simulated NICU shift")
    _add_common(sim)
    sim.add_argument("--out", default=None)
    sim.add_argument("--format", choices=["json", "csv", "fhir", "all"], default="json")
    sim.add_argument("--json", action="store_true", help="print machine-readable summary")
    sim.set_defaults(func=cmd_simulate)

    serve = sub.add_parser("serve", help="run the FastAPI dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    info = sub.add_parser("info", help="show runtime and model information")
    info.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    info.set_defaults(func=cmd_info)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
