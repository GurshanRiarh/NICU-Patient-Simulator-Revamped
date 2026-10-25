import json
from pathlib import Path

import numpy as np

from neosim.config import SimulationConfig
from neosim.simulator.engine import NeoSimulator
from neosim.simulator.events import ClinicalEvent, EventSchedule
from neosim.simulator.physiology import VitalArrays, apply_care_effects, enforce_events


def _simulator(dummy_norm) -> NeoSimulator:
    return NeoSimulator(model_path=Path("/nonexistent/model.pt"), norm=dummy_norm, allow_fallback=True)


def test_fallback_simulation_and_exports(dummy_norm, tmp_path):
    sim = _simulator(dummy_norm)
    result = sim.simulate(
        SimulationConfig(condition="rds", shift_hours=1, interval_minutes=5, seed=3, severity=0.7)
    )
    assert len(result.times) == 12
    assert all(len(result.arrays[name]) == 12 for name in ("hr", "rr", "spo2", "temp"))
    assert result.events
    assert result.interventions
    assert result.summary()["n_interventions"] == len(result.interventions)

    legacy = sim.to_legacy_json(result)
    assert {"patient_id", "Sex", "Age", "Weight", "Conditions", "data"} <= set(legacy)
    assert len(legacy["data"]) == 12
    assert "vital_signs" in legacy["data"][0]
    assert "pain_level" in legacy["data"][0]

    fhir = sim.to_fhir(result)
    assert fhir["resourceType"] == "Bundle"
    assert len(fhir["entry"]) > 40

    paths = sim.to_csv(result, tmp_path)
    assert all(path.exists() for path in paths.values())
    assert json.loads((tmp_path / "summary.json").read_text())["patient_id"] == "SIM-0001"


def test_simulation_is_reproducible(dummy_norm):
    sim = _simulator(dummy_norm)
    config = SimulationConfig(shift_hours=2, interval_minutes=5, seed=99)
    first = sim.simulate(config).arrays["hr"]
    second = sim.simulate(config).arrays["hr"]
    assert np.allclose(first, second)


def test_enforced_events_reach_targets():
    n = 120
    arrays = VitalArrays(
        hr=np.full(n, 150.0),
        rr=np.full(n, 40.0),
        spo2=np.full(n, 96.0),
        temp=np.full(n, 37.0),
    )
    schedule = EventSchedule(shift_minutes=n)
    schedule.add(
        ClinicalEvent(30.0, 1.0, "bradycardia", 0.8, meta={"associated_desat": False})
    )
    enforce_events(arrays, schedule, SimulationConfig())
    assert arrays.hr.min() < 100
    assert np.allclose(arrays.spo2, 96.0)


def test_closed_loop_responds_to_measured_episodes(dummy_norm):
    sim = _simulator(dummy_norm)
    n = 120
    arrays = VitalArrays(
        hr=np.full(n, 150.0),
        rr=np.full(n, 40.0),
        spo2=np.full(n, 96.0),
        temp=np.full(n, 37.0),
    )
    arrays.spo2[30:36] = 75.0
    arrays.hr[70:74] = 80.0
    arrays.rr[95:99] = 2.0
    schedule = EventSchedule(shift_minutes=n)
    responses = sim._closed_loop_responses(arrays, schedule, np.random.default_rng(1))

    triggers = {event.meta["trigger"] for event in responses}
    assert triggers == {"desaturation", "bradycardia", "apnea"}
    assert all(event.meta.get("closed_loop") for event in responses)

    before = arrays.spo2.copy()
    oxygen = [event for event in responses if event.type == "oxygen"]
    apply_care_effects(arrays, schedule, SimulationConfig(), events=oxygen)
    assert np.any(arrays.spo2 > before)
