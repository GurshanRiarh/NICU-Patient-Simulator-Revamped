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
from neosim.simulator import priors

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
        # The absolute bradycardia rate is calibrated to the held-out PreMo
        # cohort.  Conditional bradycardia-desaturation coupling is handled
        # separately from the literature priors.
        brady = base * 0.09
        desat = base * 1.35
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
        self._couple_spontaneous(schedule)
        if self.config.care_events:
            self._care(schedule)
            self._couple_care(schedule)
            self._medication_interactions(schedule)
        self._dedupe(schedule)
        return schedule

    def _spontaneous(self, schedule: EventSchedule) -> None:
        brady_rate, desat_rate, apnea_rate, tachy_rate = self._rates()
        for t in self._poisson_times(brady_rate):
            severity = float(np.clip(self.rng.beta(2.0, 2.5) + 0.15 * self.severity, 0.1, 1.0))
            # Bradycardias are defined by a fall of at least 33% for at least
            # 4 s (Poets 1993), so most are much shorter than a minute.
            duration = float(self.rng.uniform(0.1, 0.9))
            kind = "severe_bradycardia" if severity > 0.72 else "bradycardia"
            schedule.add(ClinicalEvent(t, duration, kind, severity, "Spontaneous bradycardia"))
        for t in self._poisson_times(desat_rate):
            severity = float(np.clip(self.rng.beta(2.0, 2.0) + 0.1 * self.severity, 0.1, 1.0))
            # A 1-4 min desaturation is a plausible monitor-level episode; the
            # exact duration is not well standardised across studies.
            duration = float(self.rng.uniform(1.0, 4.0))
            kind = "severe_desaturation" if severity > 0.70 else "desaturation"
            schedule.add(ClinicalEvent(t, duration, kind, severity, "Spontaneous desaturation"))
        for t in self._poisson_times(apnea_rate):
            severity = float(np.clip(self.rng.beta(2.0, 2.0) + 0.15 * self.severity, 0.2, 1.0))
            # Apnoea duration is drawn from the literature mixture (10-90 s),
            # which drives the duration-dependent bradycardia probability.
            duration = priors.sample_apnea_duration(self.rng) / 60.0
            schedule.add(ClinicalEvent(t, duration, "apnea", severity, "Apnoea of prematurity"))
        for t in self._poisson_times(tachy_rate):
            severity = float(np.clip(self.rng.beta(2.0, 2.0), 0.2, 1.0))
            duration = float(self.rng.uniform(3.0, 15.0))
            schedule.add(ClinicalEvent(t, duration, "tachycardia", severity, "Tachycardia episode"))

    @staticmethod
    def _has_nearby(schedule: EventSchedule, types: set[str], start: float, pad: float = 2.0) -> bool:
        return any(event.type in types and abs(event.start - start) <= pad for event in schedule.events)

    @staticmethod
    def _linked(
        start: float,
        duration: float,
        kind: str,
        severity: float,
        label: str,
        trigger: str,
    ) -> ClinicalEvent:
        return ClinicalEvent(
            start,
            duration,
            kind,
            float(np.clip(severity, 0.1, 1.0)),
            label,
            {"trigger": trigger},
        )

    def _couple_spontaneous(self, schedule: EventSchedule) -> None:
        """Add the events that physiologically travel together.

        The conditional probabilities and delays come from the neonatal
        literature (see :mod:`neosim.simulator.priors`):

        * bradycardia risk rises with apnoea duration: 10% at 10-14 s, 34% at
          15-20 s and 75% above 20 s (Zhao 2011);
        * desaturation typically starts within seconds of apnoea onset and
          precedes bradycardia (Poets 1993);
        * 86% of bradycardias are associated with a desaturation (Poets 1993);
          the reverse probability is calibrated to PreMo and is much lower than
          the 2020 cohort's combined-event fraction.
        """
        brady_types = {"bradycardia", "severe_bradycardia"}
        desat_types = {"desaturation", "severe_desaturation"}
        for event in list(schedule.events):
            if event.type == "apnea":
                duration_s = event.duration * 60.0
                event.meta["associated_desat"] = False
                if self.rng.random() < priors.desaturation_given_apnea(duration_s, event.severity):
                    start = event.start + float(self.rng.uniform(*priors.APNEA_DESAT_DELAY_MIN))
                    severity = float(
                        np.clip(0.15 + 0.9 * event.severity + self.rng.normal(0, 0.05), 0.2, 1.0)
                    )
                    kind = "severe_desaturation" if severity > 0.70 else "desaturation"
                    event.meta["associated_desat"] = True
                    if not self._has_nearby(schedule, desat_types, start):
                        schedule.add(
                            self._linked(
                                start,
                                float(self.rng.uniform(0.5, 3.0)),
                                kind,
                                severity,
                                "Apnoea-associated desaturation",
                                "apnea",
                            )
                        )
                event.meta["associated_brady"] = False
                if self.rng.random() < priors.bradycardia_given_apnea(duration_s):
                    start = event.start + float(self.rng.uniform(*priors.APNEA_BRADY_DELAY_MIN))
                    severity = float(np.clip(0.10 + 0.85 * event.severity, 0.1, 1.0))
                    kind = "severe_bradycardia" if severity > 0.72 else "bradycardia"
                    event.meta["associated_brady"] = True
                    if not self._has_nearby(schedule, brady_types, start):
                        linked = self._linked(
                            start,
                            float(self.rng.uniform(0.1, 0.9)),
                            kind,
                            severity,
                            "Apnoea-associated bradycardia",
                            "apnea",
                        )
                        linked.meta["associated_desat"] = self._has_nearby(
                            schedule, desat_types, start
                        )
                        schedule.add(linked)
            elif event.type in brady_types:
                if self.rng.random() < priors.BRADY_DESAT_ASSOC:
                    event.meta["associated_desat"] = True
                    start = event.start + float(self.rng.uniform(0.0, 0.2))
                    severity = float(np.clip(0.15 + 0.75 * event.severity, 0.2, 1.0))
                    kind = "severe_desaturation" if severity > 0.70 else "desaturation"
                    if not self._has_nearby(schedule, desat_types, start):
                        schedule.add(
                            self._linked(
                                start,
                                float(self.rng.uniform(0.5, 3.0)),
                                kind,
                                severity,
                                "Bradycardia-associated desaturation",
                                "bradycardia",
                            )
                        )
                else:
                    event.meta["associated_desat"] = False
            elif event.type in desat_types:
                if self.rng.random() < priors.desaturation_bradycardia_probability(event.severity):
                    start = event.start + float(self.rng.uniform(0.0, 0.2))
                    severity = float(np.clip(0.10 + 0.70 * event.severity, 0.1, 1.0))
                    kind = "severe_bradycardia" if severity > 0.72 else "bradycardia"
                    if not self._has_nearby(schedule, brady_types, start):
                        linked = self._linked(
                            start,
                            float(self.rng.uniform(0.1, 0.9)),
                            kind,
                            severity,
                            "Desaturation-associated bradycardia",
                            "desaturation",
                        )
                        linked.meta["associated_desat"] = True
                        schedule.add(linked)

    def _couple_care(self, schedule: EventSchedule) -> None:
        """Model events triggered by feeding, handling and suction.

        The risks are literature-derived where evidence exists.  Feeding
        desaturation is the best-quantified link (Thoyre 2003; Poets 1997);
        handling usually raises heart rate rather than causing bradycardia
        (Brandon 2022); suction effects are strongly modified by preoxygenation
        (Shah 1992).  Each care-associated bradycardia is treated as an
        isolated bradycardia unless a separate desaturation event is generated,
        so the event labels and the enforced vitals stay consistent.
        """
        cfg = self.config
        immaturity = float(np.clip((34.0 - cfg.gestational_age_weeks) / 10.0, 0.0, 1.0))
        brady_types = {"bradycardia", "severe_bradycardia"}
        desat_types = {"desaturation", "severe_desaturation"}
        for event in list(schedule.events):
            if event.type == "feeding":
                route_scale = 1.0 if event.meta.get("route", "bottle") == "bottle" else 0.35
                feeding_desat_risk = (
                    priors.feeding_desaturation_probability(immaturity, self.severity)
                    * route_scale
                )
                if self.rng.random() < feeding_desat_risk:
                    if self.rng.random() < priors.FEEDING_DESAT_DURING_PROB:
                        offset = self.rng.uniform(*priors.FEEDING_DESAT_DURING_WINDOW)
                    else:
                        offset = self.rng.uniform(*priors.FEEDING_DESAT_LATE_WINDOW)
                    start = event.start + float(offset)
                    severity = priors.feeding_desaturation_severity(self.rng)
                    kind = "severe_desaturation" if severity >= 0.52 else "desaturation"
                    if not self._has_nearby(schedule, desat_types, start):
                        schedule.add(
                            self._linked(
                                start,
                                float(self.rng.uniform(0.5, 3.0)),
                                kind,
                                severity,
                                "Feeding-associated desaturation",
                                "feeding",
                            )
                        )
                if self.rng.random() < priors.feeding_bradycardia_probability(
                    immaturity, self.severity
                ):
                    start = event.start + float(self.rng.uniform(*priors.FEEDING_DESAT_DURING_WINDOW))
                    if not self._has_nearby(schedule, brady_types, start):
                        linked = self._linked(
                            start,
                            float(self.rng.uniform(0.1, 0.9)),
                            "bradycardia",
                            float(self.rng.uniform(0.2, 0.5)),
                            "Feeding-associated bradycardia",
                            "feeding",
                        )
                        linked.meta["associated_desat"] = False
                        schedule.add(linked)
            elif event.type == "handling":
                if self.rng.random() < priors.handling_bradycardia_probability(
                    immaturity, self.severity
                ):
                    start = event.start + float(self.rng.uniform(0.0, 2.0))
                    if not self._has_nearby(schedule, brady_types, start):
                        linked = self._linked(
                            start,
                            float(self.rng.uniform(0.1, 0.9)),
                            "bradycardia",
                            float(self.rng.uniform(0.15, 0.45)),
                            "Handling-associated bradycardia",
                            "handling",
                        )
                        linked.meta["associated_desat"] = False
                        schedule.add(linked)
                if self.rng.random() < priors.handling_desaturation_probability(
                    immaturity, self.severity
                ):
                    start = event.start + float(self.rng.uniform(0.0, 2.0))
                    if not self._has_nearby(schedule, desat_types, start):
                        schedule.add(
                            self._linked(
                                start,
                                float(self.rng.uniform(0.5, 3.0)),
                                "desaturation",
                                float(self.rng.uniform(0.2, 0.5)),
                                "Handling-associated desaturation",
                                "handling",
                            )
                        )
            elif event.type == "suction":
                preoxygenated = event.meta.get("preoxygenated")
                if preoxygenated is None:
                    preoxygenated = self.rng.random() < priors.PREOXYGENATION_PROB
                    event.meta["preoxygenated"] = preoxygenated
                risk_scale = priors.suction_risk_scale(preoxygenated)
                if self.rng.random() < (
                    priors.suction_bradycardia_probability(immaturity, self.severity) * risk_scale
                ):
                    start = event.start + float(self.rng.uniform(0.0, 0.5))
                    severity = float(
                        self.rng.uniform(0.2, 0.35)
                        if preoxygenated
                        else self.rng.uniform(0.35, 0.7)
                    )
                    if not self._has_nearby(schedule, brady_types, start):
                        linked = self._linked(
                            start,
                            float(self.rng.uniform(0.1, 0.9)),
                            "bradycardia",
                            severity,
                            "Suction-associated bradycardia",
                            "suction",
                        )
                        linked.meta["associated_desat"] = False
                        schedule.add(linked)
                if self.rng.random() < (
                    priors.suction_desaturation_probability(immaturity, self.severity) * risk_scale
                ):
                    start = event.start + float(self.rng.uniform(0.0, 0.5))
                    severity = float(
                        self.rng.uniform(0.15, 0.35)
                        if preoxygenated
                        else self.rng.uniform(0.35, 0.75)
                    )
                    if not self._has_nearby(schedule, desat_types, start):
                        schedule.add(
                            self._linked(
                                start,
                                float(self.rng.uniform(0.5, 3.0)),
                                "desaturation",
                                severity,
                                "Suction-associated desaturation",
                                "suction",
                            )
                        )

    def _medication_interactions(self, schedule: EventSchedule) -> None:
        """Let prescribed drugs alter later respiratory events."""
        brady_types = {"bradycardia", "severe_bradycardia"}
        desat_types = {"desaturation", "severe_desaturation"}
        for medication in [event for event in schedule.events if event.type == "medication"]:
            drug = medication.meta.get("medication")
            if not drug:
                continue
            for event in schedule.events:
                if event.start <= medication.start or "modified_by" in event.meta:
                    continue
                if drug == "caffeine" and event.type == "apnea":
                    event.severity *= priors.CAFFEINE_APNEA_SEVERITY_MULT
                    event.duration *= priors.CAFFEINE_APNEA_DURATION_MULT
                    event.meta["modified_by"] = "caffeine"
                elif drug == "caffeine" and event.type in brady_types | desat_types:
                    event.severity *= priors.CAFFEINE_RESP_EVENT_SEVERITY_MULT
                    event.meta["modified_by"] = "caffeine"
                elif drug == "surfactant" and event.type in desat_types:
                    event.severity *= priors.SURFACTANT_DESAT_SEVERITY_MULT
                    event.duration *= priors.SURFACTANT_DESAT_DURATION_MULT
                    event.meta["modified_by"] = "surfactant"
                elif drug == "morphine" and event.type in {"apnea"} | brady_types | desat_types:
                    event.severity = min(
                        1.0, event.severity * priors.MORPHINE_EVENT_SEVERITY_MULT
                    )
                    event.meta["modified_by"] = "morphine"

    def _selected(self, name: str) -> bool:
        return not self.config.interventions or name in self.config.interventions

    def _care(self, schedule: EventSchedule) -> None:
        cfg = self.config
        if self._selected("feeding"):
            # Feeding every three hours; slightly jittered as in real care.
            # Bottle feeds produce about three times more desaturations than
            # bolus gavage feeds (Poets 1997), so the route is recorded and the
            # coupling risk is scaled accordingly.
            t = float(self.rng.uniform(0, 60))
            while t < self.shift:
                route = "bottle" if self.rng.random() < 0.5 else "gavage"
                label = "Bottle feed" if route == "bottle" else "Gavage feed"
                schedule.add(
                    ClinicalEvent(
                        t,
                        25.0,
                        "feeding",
                        0.5,
                        label,
                        {"volume_ml": int(self.rng.integers(5, 35)), "route": route},
                    )
                )
                t += 180.0 + float(self.rng.normal(0, 10))
        if self._selected("diaper change"):
            # Cluster nappy changes with feeds.  Bundled care events in preterm
            # infants last about 28 min and contain several activities
            # (Brandon 2022), so care is not scheduled as isolated tasks.
            feedings = [event for event in schedule.events if event.type == "feeding"]
            if feedings:
                for feeding in feedings:
                    t = feeding.start + float(self.rng.uniform(5.0, 20.0))
                    schedule.add(ClinicalEvent(t, 6.0, "handling", 0.35, "Nappy change"))
            else:
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
                schedule.add(
                    ClinicalEvent(
                        t,
                        3.0,
                        "suction",
                        0.6,
                        "Endotracheal / nasal suction",
                        {"preoxygenated": bool(self.rng.random() < priors.PREOXYGENATION_PROB)},
                    )
                )
                t += 240.0 + float(self.rng.normal(0, 25))
        if self._selected("family visitation"):
            for _ in range(int(self.rng.integers(0, 3))):
                t = float(self.rng.uniform(0, self.shift))
                # Kangaroo care has a measurable bradycardia/desaturation
                # benefit, whereas a brief family visit does not necessarily
                # include holding; the severity captures that variation.
                severity = float(self.rng.uniform(0.7, 1.0))
                schedule.add(
                    ClinicalEvent(t, 45.0, "family_visit", severity, "Kangaroo care / family visit")
                )
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
