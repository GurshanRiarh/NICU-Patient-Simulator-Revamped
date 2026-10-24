from neosim.config import EVENT_TYPES, SimulationConfig
from neosim.simulator.events import ClinicalEvent, EventSchedule, EventScheduler


def test_schedule_is_bounded_and_has_care_events():
    schedule = EventScheduler(SimulationConfig(shift_hours=8, seed=4)).build()
    assert schedule.events
    assert all(0 <= event.start < 480 for event in schedule.events)
    assert all(event.duration > 0 for event in schedule.events)
    assert any(event.type == "feeding" for event in schedule.events)
    assert any(event.type in {"bradycardia", "severe_bradycardia"} for event in schedule.events)


def test_event_matrix_shape_and_alignment():
    schedule = EventScheduler(SimulationConfig(shift_hours=2, seed=2)).build()
    matrix = schedule.matrix(120)
    assert matrix.shape == (120, len(EVENT_TYPES))
    for event in schedule.events:
        if event.type in EVENT_TYPES:
            index = EVENT_TYPES.index(event.type)
            assert matrix[:, index].sum() >= 1


def test_intervention_filter():
    schedule = EventScheduler(SimulationConfig(shift_hours=2, seed=3, interventions=["feeding"])).build()
    types = {event.type for event in schedule.events}
    assert "feeding" in types
    assert "handling" not in types


def test_apnea_couples_to_desaturation_and_bradycardia():
    scheduler = EventScheduler(SimulationConfig(shift_hours=2, seed=7, severity=0.9))
    schedule = EventSchedule(shift_minutes=120)
    for start in range(5, 120, 10):
        schedule.add(ClinicalEvent(float(start), 1.0, "apnea", 1.0))
    scheduler._couple_spontaneous(schedule)

    linked = [event for event in schedule.events if event.meta.get("trigger") == "apnea"]
    assert any(event.type in {"desaturation", "severe_desaturation"} for event in linked)
    assert any(event.type in {"bradycardia", "severe_bradycardia"} for event in linked)


def test_caffeine_attenuates_later_apnea():
    scheduler = EventScheduler(SimulationConfig(shift_hours=2, seed=8))
    schedule = EventSchedule(shift_minutes=120)
    schedule.add(ClinicalEvent(20, 10.0, "medication", 0.5, "Caffeine citrate", {"medication": "caffeine"}))
    apnea = ClinicalEvent(40, 1.0, "apnea", 0.9)
    schedule.add(apnea)
    scheduler._medication_interactions(schedule)
    assert apnea.severity < 0.9
    assert apnea.meta["modified_by"] == "caffeine"
