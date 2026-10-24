"""Physiology layer: intervention response kernels, event enforcement and
thermoregulation.  The neural model supplies realistic baseline dynamics; this
layer makes the simulator controllable and clinically consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neosim.config import SimulationConfig
from neosim.simulator.events import ClinicalEvent, EventSchedule


@dataclass(slots=True)
class VitalArrays:
    hr: np.ndarray
    rr: np.ndarray
    spo2: np.ndarray
    temp: np.ndarray

    def copy(self) -> VitalArrays:
        return VitalArrays(self.hr.copy(), self.rr.copy(), self.spo2.copy(), self.temp.copy())


def _kernel(n: int, start: float, duration: float, tau_rise: float, tau_decay: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)
    u = t - start
    shape = np.zeros(n, dtype=np.float32)
    mask = (u >= 0) & (u <= duration + 3 * tau_decay)
    if not mask.any():
        return shape
    rise = 1.0 - np.exp(-np.clip(u[mask], 0, None) / max(tau_rise, 1e-3))
    decay = np.exp(-np.clip(u[mask] - duration, 0, None) / max(tau_decay, 1e-3))
    shape[mask] = rise * decay
    peak = float(shape.max())
    if peak > 0:
        shape /= peak
    return shape


def _bump(n: int, start: float, duration: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)
    shape = np.zeros(n, dtype=np.float32)
    lo = max(0, int(math.floor(start)))
    hi = min(n, int(math.ceil(start + max(duration, 0.5))))
    if hi <= lo:
        return shape
    u = (t[lo:hi] - start) / max(duration, 0.5)
    shape[lo:hi] = np.sin(np.pi * np.clip(u, 0, 1)) ** 2
    return shape


def _scaled_effect(
    arrays: VitalArrays,
    event: ClinicalEvent,
    hr: float,
    rr: float,
    spo2: float,
    temp: float,
    tau_rise: float,
    tau_decay: float,
) -> None:
    n = len(arrays.hr)
    shape = _kernel(n, event.start, event.duration, tau_rise, tau_decay) * event.severity
    arrays.hr += hr * shape
    arrays.rr += rr * shape
    arrays.spo2 += spo2 * shape
    arrays.temp += temp * shape


def apply_care_effects(
    arrays: VitalArrays,
    schedule: EventSchedule,
    config: SimulationConfig,
    events: list[ClinicalEvent] | None = None,
) -> None:
    """Apply explicit physiological responses to care events.

    The neural generator already captures spontaneous physiology and is
    conditioned on the event flags.  These kernels add the effects of routine
    care that is not represented in the public training data.  When ``events``
    is supplied, only those events are applied; this is used by the closed-loop
    controller after the first pass of generated vitals.
    """
    source = schedule.care_events() if events is None else events
    for event in source:
        etype = event.type
        med = event.meta.get("medication")
        if etype == "feeding":
            _scaled_effect(arrays, event, 5.0, 2.5, -0.8, 0.08, 5.0, 25.0)
        elif etype == "handling":
            if event.meta.get("closed_loop"):
                _scaled_effect(arrays, event, 9.0, 4.0, -0.8, 0.0, 0.3, 3.0)
            else:
                pain = 12.0 + 12.0 * event.severity
                _scaled_effect(arrays, event, pain, pain * 0.45, -2.5, -0.05, 0.6, 6.0)
        elif etype == "position_change":
            _scaled_effect(arrays, event, 4.0, 1.5, 0.5, 0.0, 0.8, 4.0)
        elif etype == "oxygen":
            if event.meta.get("closed_loop"):
                _scaled_effect(arrays, event, -2.0, -1.0, 6.0, 0.0, 1.2, 10.0)
            else:
                _scaled_effect(arrays, event, -1.5, -1.0, 3.0, 0.0, 1.5, 9.0)
        elif etype == "suction":
            _scaled_effect(arrays, event, -12.0, -4.0, -5.0, 0.0, 0.3, 1.5)
        elif etype == "family_visit":
            _scaled_effect(arrays, event, -3.0, -1.5, 0.8, 0.1, 4.0, 20.0)
        elif etype == "pain":
            _scaled_effect(arrays, event, -4.0, -2.0, 0.5, 0.0, 2.0, 12.0)
        elif etype == "medication":
            if med == "caffeine":
                _scaled_effect(arrays, event, 4.0, 5.0, 0.5, 0.05, 3.0, 120.0)
            elif med == "morphine":
                _scaled_effect(arrays, event, -5.0, -6.0, -1.5, -0.08, 4.0, 45.0)
            elif med == "antibiotic":
                _scaled_effect(arrays, event, 0.0, 0.0, 0.0, -0.0, 10.0, 240.0)
            elif med == "surfactant":
                _scaled_effect(arrays, event, -3.0, -6.0, 4.0, 0.0, 2.0, 90.0)
        elif etype == "lighting":
            _scaled_effect(arrays, event, -1.0, -0.5, 0.2, 0.0, 2.0, 20.0)
    # A small, consistent effect of incubator humidity/feeding on temperature.
    arrays.temp += 0.02 * np.sin(np.linspace(0, 2 * math.pi, len(arrays.temp)))


def enforce_events(arrays: VitalArrays, schedule: EventSchedule, config: SimulationConfig) -> None:
    """Guarantee that scheduled bradycardia/desaturation/apnoea events are visible.

    The generative model is conditioned on these events, but an unconditional
    sample can occasionally be too mild.  We only add a correction when the
    generated trace does not already reach the event target, preserving the
    model's morphology instead of overwriting it.
    """
    baseline_hr = float(np.median(arrays.hr))
    baseline_spo2 = float(np.median(arrays.spo2))
    for event in schedule.events:
        if event.type not in {
            "bradycardia",
            "severe_bradycardia",
            "desaturation",
            "severe_desaturation",
            "apnea",
            "tachycardia",
        }:
            continue
        shape = _bump(len(arrays.hr), event.start, event.duration)
        if not shape.any():
            continue
        sev = float(np.clip(event.severity, 0.1, 1.0))
        if event.type in {"bradycardia", "severe_bradycardia"}:
            target_hr = 92.0 - 35.0 * sev
            target_spo2 = baseline_spo2 - 8.0 * sev
            arrays.hr -= np.maximum(0.0, arrays.hr - target_hr) * shape
            arrays.spo2 -= np.maximum(0.0, arrays.spo2 - target_spo2) * shape
        elif event.type in {"desaturation", "severe_desaturation"}:
            target_spo2 = 93.0 - 25.0 * sev
            arrays.spo2 -= np.maximum(0.0, arrays.spo2 - target_spo2) * shape
            # Desaturations are often preceded by a brief sympathetic rise.
            arrays.hr += 6.0 * sev * _kernel(len(arrays.hr), event.start, 0.5, 0.5, 3.0)
        elif event.type == "apnea":
            target_rr = max(0.0, 3.0 - 3.0 * sev)
            arrays.rr -= np.maximum(0.0, arrays.rr - target_rr) * shape
            arrays.spo2 -= np.maximum(0.0, arrays.spo2 - (baseline_spo2 - 12.0 * sev)) * shape
            arrays.hr -= np.maximum(0.0, arrays.hr - (baseline_hr - 15.0 * sev)) * shape
        elif event.type == "tachycardia":
            target_hr = baseline_hr + 25.0 * sev
            arrays.hr += np.maximum(0.0, target_hr - arrays.hr) * shape
            arrays.spo2 -= 1.5 * sev * shape

    arrays.hr = np.clip(arrays.hr, 40, 250)
    arrays.rr = np.clip(arrays.rr, 0, 120)
    arrays.spo2 = np.clip(arrays.spo2, 30, 100)


def generate_temperature(
    n: int,
    config: SimulationConfig,
    schedule: EventSchedule,
    rng: np.random.Generator,
) -> np.ndarray:
    baseline = float(config.temperature_setpoint)
    if config.condition in {"sepsis", "nec"}:
        baseline += 0.35
    if config.birth_weight_g < 1000:
        baseline -= 0.15
    temp = np.empty(n, dtype=np.float32)
    value = baseline
    for t in range(n):
        drift = 0.06 * (baseline - value)
        value += drift + rng.normal(0, 0.035)
        temp[t] = value
    # Slow circadian drift (thermoneutral care still shows a small rhythm).
    temp += 0.12 * np.sin(2 * np.pi * (np.arange(n) / 1440.0 - 0.15))
    for event in schedule.events:
        if event.type == "feeding":
            temp += 0.08 * _kernel(n, event.start, 20.0, 5.0, 30.0) * event.severity
        elif event.type in {"handling", "suction"}:
            temp -= 0.06 * _kernel(n, event.start, 2.0, 0.5, 10.0) * event.severity
        elif event.type == "family_visit":
            temp += 0.05 * _kernel(n, event.start, 30.0, 5.0, 30.0) * event.severity
        elif event.type == "medication" and event.meta.get("medication") == "antibiotic":
            temp -= 0.15 * _kernel(n, event.start, 60.0, 15.0, 240.0) * event.severity
    if config.condition in {"sepsis", "nec"}:
        onset = 0.0
        for event in schedule.events:
            if event.type in {"sepsis", "nec"}:
                onset = event.start
                break
        ramp = np.clip((np.arange(n) - onset) / 360.0, 0, 1)
        temp += 0.75 * ramp * config.severity
    return np.clip(temp, 34.0, 40.0)


def add_monitor_noise(arrays: VitalArrays, rng: np.random.Generator) -> None:
    n = len(arrays.hr)
    arrays.hr += rng.normal(0, 0.7, n)
    arrays.rr += rng.normal(0, 0.5, n)
    arrays.spo2 += rng.normal(0, 0.35, n)
    arrays.temp += rng.normal(0, 0.02, n)
