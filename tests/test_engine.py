import json
from pathlib import Path

import numpy as np

from neosim.config import SimulationConfig
from neosim.simulator.engine import NeoSimulator


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


def test_enforced_events_reach_targets(dummy_norm):
    sim = _simulator(dummy_norm)
    result = sim.simulate(
        SimulationConfig(condition="bradycardia", shift_hours=4, interval_minutes=1, seed=12, severity=0.9)
    )
    assert result.arrays["hr"].min() < 110
