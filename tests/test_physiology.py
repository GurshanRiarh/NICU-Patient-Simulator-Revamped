import numpy as np

from neosim.config import SimulationConfig
from neosim.simulator.events import ClinicalEvent, EventSchedule
from neosim.simulator.physiology import VitalArrays, enforce_events


def _arrays(n: int = 60) -> VitalArrays:
    return VitalArrays(
        hr=np.full(n, 150.0, dtype=np.float32),
        rr=np.full(n, 40.0, dtype=np.float32),
        spo2=np.full(n, 96.0, dtype=np.float32),
        temp=np.full(n, 37.0, dtype=np.float32),
    )


def test_apnea_enforcement_only_changes_respiration():
    arrays = _arrays()
    schedule = EventSchedule(shift_minutes=len(arrays.hr))
    schedule.add(ClinicalEvent(20.0, 1.0, "apnea", 0.8))
    enforce_events(arrays, schedule, SimulationConfig())
    assert arrays.rr.min() < 5
    assert np.allclose(arrays.spo2, 96.0)
    assert np.allclose(arrays.hr, 150.0)


def test_isolated_bradycardia_does_not_lower_spo2():
    arrays = _arrays()
    schedule = EventSchedule(shift_minutes=len(arrays.hr))
    schedule.add(
        ClinicalEvent(20.0, 1.0, "bradycardia", 0.8, meta={"associated_desat": False})
    )
    enforce_events(arrays, schedule, SimulationConfig())
    assert arrays.hr.min() < 100
    assert np.allclose(arrays.spo2, 96.0)


def test_bradycardia_with_desaturation_lowers_both():
    arrays = _arrays()
    schedule = EventSchedule(shift_minutes=len(arrays.hr))
    schedule.add(
        ClinicalEvent(20.0, 1.0, "bradycardia", 0.8, meta={"associated_desat": True})
    )
    enforce_events(arrays, schedule, SimulationConfig())
    assert arrays.hr.min() < 100
    assert arrays.spo2.min() < 96.0


def test_subminute_event_is_visible_at_monitor_resolution():
    arrays = _arrays()
    schedule = EventSchedule(shift_minutes=len(arrays.hr))
    schedule.add(ClinicalEvent(20.4, 0.2, "bradycardia", 0.5))
    enforce_events(arrays, schedule, SimulationConfig())
    assert arrays.hr.min() < 100
