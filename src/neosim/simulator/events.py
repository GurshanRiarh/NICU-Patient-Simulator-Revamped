"""Clinical event scheduling for the NICU simulator.

The scheduler creates two families of events:

1. **Spontaneous physiology** - bradycardia, desaturations, apnoea and
   tachycardia.  Timestamps and durations are passed to the neural generator as
   conditioning flags.
2. **Care events** - feeding, nappy changes, medication, handling, oxygen,
   suction, family visits, etc.  These are exported as discrete interventions
   for downstream summarisation and also drive explicit physiological response
   kernels in :mod:`neosim.simulator.physiology`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from neosim.config import EVENT_INDEX, EVENT_TYPES, SimulationConfig

SPONTANEOUS = {
    "bradycardia",
    "severe_bradycardia",
    "desaturation",
    "severe_desaturation",
    "apnea",
    "tachycardia",
}


@dataclass(slots=True)
class ClinicalEvent:
    start: float
    duration: float
    type: str
    severity: float = 0.5
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def end(self) -> float:
        return self.start + self.duration

    def overlaps(self, other: ClinicalEvent, pad: float = 0.0) -> bool:
        return not (self.end + pad <= other.start or other.end + pad <= self.start)


@dataclass
class EventSchedule:
    events: list[ClinicalEvent] = field(default_factory=list)
    shift_minutes: float = 480.0

    def add(self, event: ClinicalEvent) -> None:
        if 0 <= event.start < self.shift_minutes:
            event.duration = min(event.duration, self.shift_minutes - event.start)
            self.events.append(event)

    def matrix(self, n_minutes: int | None = None) -> np.ndarray:
        n = int(n_minutes or self.shift_minutes)
        out = np.zeros((n, len(EVENT_TYPES)), dtype=np.float32)
        for event in self.events:
            if event.type not in EVENT_INDEX:
                continue
            lo = max(0, int(math.floor(event.start)))
            hi = min(n, int(math.ceil(event.end)))
            if hi > lo:
                out[lo:hi, EVENT_INDEX[event.type]] = 1.0
        return out

    def care_events(self) -> list[ClinicalEvent]:
        return [e for e in self.events if e.type not in SPONTANEOUS]

    def sorted(self) -> list[ClinicalEvent]:
        return sorted(self.events, key=lambda e: (e.start, e.type))

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.type] = counts.get(event.type, 0) + 1
        return counts


class EventScheduler:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed if config.seed is not None else random.randrange(2**31))
        self.shift = int(config.shift_hours * 60)
        self.severity = float(np.clip(config.severity, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Rates
    # ------------------------------------------------------------------
    def _rates(self) -> tuple[float, float, float, float]:
        """Return brady, desat, apnoea and tachycardia rates per hour."""
        condition = self.config.condition
        ga = self.config.gestational_age_weeks
        weight = self.config.birth_weight_g
        immaturity = float(np.clip((34.0 - ga) / 10.0, 0.0, 1.0))
        low_weight = float(np.clip((2500.0 - weight) / 2000.0, 0.0, 1.0))
        base = 0.06 + 0.22 * immaturity + 0.12 * low_weight + 0.30 * self.severity
        brady = base * 0.35
        desat = base * 1.6
        apnea = 0.0
        tachy = 0.0
        if condition in {"bradycardia"}:
            brady *= 3.0
        if condition in {"apnea_of_prematurity", "rds"}:
            apnea = 0.06 + 0.35 * immaturity + 0.35 * self.severity
            desat *= 1.5
        if condition == "rds":
            desat += 0.35
        if condition in {"sepsis", "nec"}:
            desat *= 2.0
            brady *= 1.4
            tachy = 0.2 + 0.8 * self.severity
        if condition == "tachycardia":
            tachy = 1.0 + 1.5 * self.severity
        return brady, desat, apnea, tachy

    def _poisson_times(self, rate_per_hour: float, seed_offset: int = 0) -> list[float]:
        if rate_per_hour <= 0:
            return []
        expected = rate_per_hour * self.config.shift_hours
        count = int(self.rng.poisson(expected))
        if count == 0:
            return []
        times = np.sort(self.rng.uniform(0, self.config.shift_hours, count)) * 60.0
        return [float(t) for t in times]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> EventSchedule:
        schedule = EventSchedule(shift_minutes=self.shift)
        self._spontaneous(schedule)
        if self.config.care_events:
            self._care(schedule)
        self._closed_loop(schedule)
        self._dedupe(schedule)
        return schedule

    def _spontaneous(self, schedule: EventSchedule) -> None:
        brady_rate, desat_rate, apnea_rate, tachy_rate = self._rates()
        for t in self._poisson_times(brady_rate):
            severity = float(np.clip(self.rng.beta(2.0, 2.5) + 0.15 * self.severity, 0.1, 1.0))
            duration = float(self.rng.uniform(0.5, 2.2))
            kind = "severe_bradycardia" if severity > 0.72 else "bradycardia"
            schedule.add(ClinicalEvent(t, duration, kind, severity, "Spontaneous bradycardia"))
        for t in self._poisson_times(desat_rate):
            severity = float(np.clip(self.rng.beta(2.0, 2.0) + 0.1 * self.severity, 0.1, 1.0))
            duration = float(self.rng.uniform(1.0, 4.5))
            kind = "severe_desaturation" if severity > 0.70 else "desaturation"
            schedule.add(ClinicalEvent(t, duration, kind, severity, "Spontaneous desaturation"))
        for t in self._poisson_times(apnea_rate):
            severity = float(np.clip(self.rng.beta(2.0, 2.0) + 0.15 * self.severity, 0.2, 1.0))
            duration = float(self.rng.uniform(0.5, 1.8))
            schedule.add(ClinicalEvent(t, duration, "apnea", severity, "Apnoea of prematurity"))
        for t in self._poisson_times(tachy_rate):
            severity = float(np.clip(self.rng.beta(2.0, 2.0), 0.2, 1.0))
            duration = float(self.rng.uniform(3.0, 15.0))
            schedule.add(ClinicalEvent(t, duration, "tachycardia", severity, "Tachycardia episode"))

    def _selected(self, name: str) -> bool:
        return not self.config.interventions or name in self.config.interventions

    def _care(self, schedule: EventSchedule) -> None:
        cfg = self.config
        if self._selected("feeding"):
            # Feeding every three hours; slightly jittered as in real care.
            t = float(self.rng.uniform(0, 60))
            while t < self.shift:
                schedule.add(
                    ClinicalEvent(
                        t, 25.0, "feeding", 0.5, "Enteral feed", {"volume_ml": int(self.rng.integers(5, 35))}
                    )
                )
                t += 180.0 + float(self.rng.normal(0, 10))
        if self._selected("diaper change"):
            t = float(self.rng.uniform(20, 100))
            while t < self.shift:
                schedule.add(ClinicalEvent(t, 6.0, "handling", 0.35, "Nappy change"))
                t += 180.0 + float(self.rng.normal(0, 12))
        if self._selected("position change"):
            t = float(self.rng.uniform(10, 90))
            while t < self.shift:
                schedule.add(ClinicalEvent(t, 5.0, "position_change", 0.3, "Position change"))
                t += 240.0 + float(self.rng.normal(0, 15))
        if self._selected("medication"):
            if cfg.condition in {"sepsis", "nec"}:
                t = float(self.rng.uniform(0, 120))
                while t < self.shift:
                    schedule.add(
                        ClinicalEvent(
                            t, 15.0, "medication", 0.6, "Antibiotic dose", {"medication": "antibiotic"}
                        )
                    )
                    t += 720.0
            if cfg.condition == "apnea_of_prematurity":
                t = float(self.rng.uniform(0, 180))
                while t < self.shift:
                    schedule.add(
                        ClinicalEvent(
                            t, 10.0, "medication", 0.5, "Caffeine citrate", {"medication": "caffeine"}
                        )
                    )
                    t += 1440.0
            if cfg.condition in {"rds"}:
                schedule.add(
                    ClinicalEvent(
                        float(self.rng.uniform(0, 30)),
                        10.0,
                        "medication",
                        0.6,
                        "Surfactant",
                        {"medication": "surfactant"},
                    )
                )
        if self._selected("oxygen administration") and (
            cfg.condition in {"rds", "apnea_of_prematurity", "sepsis"} or self.severity > 0.55
        ):
            t = float(self.rng.uniform(0, 90))
            while t < self.shift:
                schedule.add(
                    ClinicalEvent(t, 12.0, "oxygen", 0.5, "Increase FiO2 / flow", {"delta_fio2": 0.05})
                )
                t += 240.0 + float(self.rng.normal(0, 20))
        if self._selected("suction") and cfg.condition in {"rds"}:
            t = float(self.rng.uniform(20, 120))
            while t < self.shift:
                schedule.add(ClinicalEvent(t, 3.0, "suction", 0.6, "Endotracheal / nasal suction"))
                t += 240.0 + float(self.rng.normal(0, 25))
        if self._selected("family visitation"):
            for _ in range(int(self.rng.integers(0, 3))):
                t = float(self.rng.uniform(0, self.shift))
                schedule.add(ClinicalEvent(t, 45.0, "family_visit", 0.2, "Family visit / kangaroo care"))
        if self._selected("lab test"):
            t = float(self.rng.uniform(0.15, 0.55) * self.shift)
            schedule.add(ClinicalEvent(t, 4.0, "handling", 0.55, "Heel prick / blood gas"))
        if self._selected("imaging"):
            t = float(self.rng.uniform(0.2, 0.8) * self.shift)
            schedule.add(ClinicalEvent(t, 10.0, "handling", 0.4, "Chest X-ray / ultrasound"))
        if self._selected("pain management"):
            # PRN comfort measures after the most painful handling events.
            for event in list(schedule.events):
                if event.type == "handling" and event.severity > 0.45:
                    schedule.add(
                        ClinicalEvent(
                            event.start + 1.0, 8.0, "pain", 0.3, "Non-nutritive sucking / swaddling"
                        )
                    )
        if self._selected("lighting adjustment"):
            t = float(self.rng.uniform(0.2, 0.8) * self.shift)
            schedule.add(ClinicalEvent(t, 10.0, "family_visit", 0.1, "Dim lights / reduce noise"))

    def _closed_loop(self, schedule: EventSchedule) -> None:
        if not self.config.closed_loop:
            return
        for event in list(schedule.events):
            if event.type in {"severe_desaturation", "severe_bradycardia", "apnea"}:
                delay = float(self.rng.uniform(0.5, 2.0))
                schedule.add(
                    ClinicalEvent(
                        event.start + delay,
                        8.0,
                        "oxygen",
                        0.55,
                        "Closed-loop response: stimulation / oxygen",
                    )
                )

    @staticmethod
    def _dedupe(schedule: EventSchedule) -> None:
        """Remove exact duplicate care events created by overlapping rules."""
        seen: set[tuple[str, int]] = set()
        unique: list[ClinicalEvent] = []
        for event in schedule.sorted():
            key = (event.type, int(event.start))
            if key in seen:
                continue
            seen.add(key)
            unique.append(event)
        schedule.events = unique
