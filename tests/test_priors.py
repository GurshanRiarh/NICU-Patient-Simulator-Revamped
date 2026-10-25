import numpy as np

from neosim.config import SimulationConfig
from neosim.simulator import priors
from neosim.simulator.events import ClinicalEvent, EventSchedule, EventScheduler


def test_bradycardia_probability_increases_with_apnea_duration():
    assert priors.bradycardia_given_apnea(12.0) == 0.10
    assert 0.10 < priors.bradycardia_given_apnea(17.0) < 0.34
    assert abs(priors.bradycardia_given_apnea(20.0) - 0.34) < 1e-9
    assert priors.bradycardia_given_apnea(60.0) == 0.75
    assert priors.bradycardia_given_apnea(120.0) == 0.75


def test_desaturation_probability_is_bounded():
    for duration in (10.0, 20.0, 40.0, 90.0):
        for severity in (0.0, 0.5, 1.0):
            probability = priors.desaturation_given_apnea(duration, severity)
            assert 0.55 <= probability <= 0.95


def test_desaturation_bradycardia_probability_is_low():
    assert priors.desaturation_bradycardia_probability(0.0) == 0.015
    assert priors.desaturation_bradycardia_probability(1.0) == 0.025


def test_feeding_severity_matches_reported_split():
    rng = np.random.default_rng(0)
    draws = np.array([priors.feeding_desaturation_severity(rng) for _ in range(20_000)])
    mild = ((draws >= 0.16) & (draws <= 0.32)).mean()
    severe = (draws >= 0.52).mean()
    assert 0.56 < mild < 0.62
    assert 0.19 < severe < 0.23


def test_preoxygenation_reduces_suction_risk_and_effect():
    assert priors.suction_risk_scale(True) < priors.suction_risk_scale(False)
    assert priors.suction_effect_scale(True) < priors.suction_effect_scale(False)


def test_spontaneous_bradycardias_are_usually_coupled_to_desaturation():
    scheduler = EventScheduler(SimulationConfig(shift_hours=24, seed=3, severity=0.5))
    schedule = EventSchedule(shift_minutes=1440)
    for start in range(10, 1400, 20):
        schedule.add(ClinicalEvent(float(start), 1.0, "bradycardia", 0.5))
    scheduler._couple_spontaneous(schedule)
    brady = [
        event
        for event in schedule.events
        if event.type in {"bradycardia", "severe_bradycardia"}
    ]
    associated = sum(bool(event.meta.get("associated_desat")) for event in brady)
    assert 0.75 < associated / len(brady) < 0.95
