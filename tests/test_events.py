from neosim.config import EVENT_TYPES, SimulationConfig
from neosim.simulator.events import EventScheduler


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
