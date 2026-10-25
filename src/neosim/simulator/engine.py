"""NeoSim 2.0 simulation engine.

The engine combines three layers:

1. **NeoFlow** - one conditional generative model for HR, RR and SpO2.
2. **Clinical event scheduler** - spontaneous physiology and routine care.
3. **Physiology layer** - intervention response kernels, event enforcement,
   thermoregulation and monitor noise.

The output is both a rich JSON/CSV/FHIR dataset and a real-time stream suitable
for the dashboard.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from neosim.config import (
    BASELINE_CALIBRATION_FILE,
    DEFAULT_MODEL,
    EVENT_TYPES,
    NORM_FILE,
    SIGNALS,
    SimulationConfig,
)
from neosim.models.flow import load_checkpoint, sample
from neosim.simulator.events import ClinicalEvent, EventSchedule, EventScheduler
from neosim.simulator.physiology import (
    VitalArrays,
    add_monitor_noise,
    apply_care_effects,
    enforce_events,
    generate_temperature,
)

INTERVENTION_MAP = {
    "feeding": "feeding",
    "handling": "diaper change",
    "position_change": "position change",
    "oxygen": "oxygen administration",
    "medication": "medication",
    "family_visit": "family visitation",
    "suction": "suction",
    "pain": "pain management",
    "closed_loop": "closed-loop care",
}


def _profile_static(config: SimulationConfig) -> np.ndarray:
    condition = config.condition
    brady = condition in {"bradycardia", "apnea_of_prematurity"}
    tachy = condition in {"tachycardia", "sepsis"}
    apnea = condition in {"apnea_of_prematurity", "rds"}
    rds = condition == "rds"
    sepsis = condition in {"sepsis", "nec"}
    nec = condition == "nec"
    intubation = condition == "rds" or (sepsis and config.severity > 0.65)
    male = 1.0 if config.sex.lower().startswith("m") else 0.0
    return np.array(
        [
            config.gestational_age_weeks,
            config.birth_weight_g / 1000.0,
            math.log1p(max(config.postnatal_age_days, 0.0)),
            male,
            1.0 if config.cohort == "pics" else 0.0,
            float(brady),
            float(tachy),
            float(apnea),
            float(rds),
            float(sepsis),
            float(nec),
            float(intubation),
            float(np.clip(config.severity, 0, 1)),
        ],
        dtype=np.float32,
    )


def _infer_cohort(config: SimulationConfig) -> str:
    if config.cohort in {"premo", "pics"}:
        return config.cohort
    if config.birth_weight_g < 1500 or config.gestational_age_weeks < 32:
        return "premo"
    return "pics"


@dataclass
class SimulationResult:
    config: SimulationConfig
    times: list[datetime]
    interval_minutes: float
    arrays: dict[str, np.ndarray]
    events: list[dict]
    interventions: list[dict]
    pain: np.ndarray
    pain_labels: list[str]
    stability: np.ndarray
    sepsis_risk: np.ndarray
    event_matrix: np.ndarray
    model_meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def n_steps(self) -> int:
        return len(self.times)

    def vitals_at(self, index: int) -> dict[str, float]:
        return {name: float(self.arrays[name][index]) for name in ("hr", "rr", "spo2", "temp")}

    def summary(self) -> dict:
        hr = self.arrays["hr"]
        spo2 = self.arrays["spo2"]
        return {
            "patient_id": self.config.patient_id,
            "name": self.config.name,
            "condition": self.config.condition,
            "shift_hours": self.config.shift_hours,
            "model": self.model_meta.get("model", "NeoFlow"),
            "hr_mean": float(np.mean(hr)),
            "hr_min": float(np.min(hr)),
            "hr_max": float(np.max(hr)),
            "spo2_mean": float(np.mean(spo2)),
            "spo2_min": float(np.min(spo2)),
            "temp_mean": float(np.mean(self.arrays["temp"])),
            "rr_mean": float(np.mean(self.arrays["rr"])),
            "brady_minutes": int((hr < 100).sum()),
            "desat_minutes": int((spo2 < 85).sum()),
            "severe_desat_minutes": int((spo2 < 80).sum()),
            "apnea_minutes": int((self.arrays["rr"] < 5).sum()),
            "n_interventions": len(self.interventions),
            "stability_mean": float(np.mean(self.stability)),
            "sepsis_risk_max": float(np.max(self.sepsis_risk)),
        }


class NeoSimulator:
    """High-level simulator used by the CLI, API and evaluation scripts."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        device: str | None = None,
        norm: dict | None = None,
        allow_fallback: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.norm = norm
        self.model_meta: dict = {"model": "physiology-fallback"}
        model_path = Path(model_path)
        if model_path.exists():
            self.model, payload = load_checkpoint(model_path, map_location=self.device)
            self.model.to(self.device).eval()
            self.norm = payload.get("norm", self.norm)
            self.model_meta = {
                "model": "NeoFlow",
                "checkpoint": str(model_path),
                "params": payload.get("n_params"),
                "val_loss": payload.get("val_loss"),
                "epoch": payload.get("epoch"),
                "gpu": payload.get("gpu"),
            }
            print(f"Loaded NeoFlow checkpoint: {model_path} on {self.device}")
        elif not allow_fallback:
            raise FileNotFoundError(f"model checkpoint not found: {model_path}")
        else:
            print("WARNING: no NeoFlow checkpoint found; using physiology fallback.")
        if self.norm is None and NORM_FILE.exists():
            self.norm = json.loads(NORM_FILE.read_text())
        if self.norm is None:
            raise FileNotFoundError("normalisation statistics are required (run preprocessing)")

        self.channel_mean = np.array([self.norm["channels"][s]["mean"] for s in SIGNALS], dtype=np.float32)
        self.channel_std = np.array([self.norm["channels"][s]["std"] for s in SIGNALS], dtype=np.float32)
        self.static_mean = np.asarray(self.norm["static"]["mean"], dtype=np.float32)
        self.static_std = np.asarray(self.norm["static"]["std"], dtype=np.float32)
        self.baseline_calibration: dict = {}
        if BASELINE_CALIBRATION_FILE.exists():
            self.baseline_calibration = json.loads(BASELINE_CALIBRATION_FILE.read_text())

    # ------------------------------------------------------------------
    def simulate(self, config: SimulationConfig) -> SimulationResult:
        config.cohort = _infer_cohort(config)
        rng = np.random.default_rng(config.seed)
        scheduler = EventScheduler(config)
        schedule = scheduler.build()
        total_minutes = int(round(config.shift_hours * 60))
        event_matrix = schedule.matrix(total_minutes)
        if self.model is not None:
            vitals = self._generate_neural(config, schedule, total_minutes, rng)
            if config.baseline_calibration and self.baseline_calibration:
                vitals = self._apply_baseline_calibration(vitals, config)
        else:
            vitals = self._generate_fallback(config, event_matrix, total_minutes, rng)

        arrays = VitalArrays(
            hr=vitals[:, 0].astype(np.float32),
            rr=vitals[:, 1].astype(np.float32),
            spo2=vitals[:, 2].astype(np.float32),
            temp=generate_temperature(total_minutes, config, schedule, rng),
        )
        apply_care_effects(arrays, schedule, config)
        enforce_events(arrays, schedule, config)
        if config.closed_loop:
            responses = self._closed_loop_responses(arrays, schedule, rng)
            if responses:
                apply_care_effects(arrays, schedule, config, events=responses)
                event_matrix = schedule.matrix(total_minutes)
        add_monitor_noise(arrays, rng)
        arrays.hr = np.clip(np.round(arrays.hr), 40, 250)
        arrays.rr = np.clip(np.round(arrays.rr), 0, 120)
        arrays.spo2 = np.clip(np.round(arrays.spo2), 30, 100)
        arrays.temp = np.clip(np.round(arrays.temp, 1), 34.0, 40.0)

        pain, pain_labels, stability, sepsis_risk = self._scores(arrays, schedule, config)
        interval = float(config.interval_minutes)
        start = self._start_time(config)
        steps = int(math.ceil(total_minutes / interval))
        times = [start + timedelta(minutes=interval * i) for i in range(steps)]
        agg = self._aggregate(arrays, schedule, interval, steps, pain, stability, sepsis_risk, event_matrix)
        return SimulationResult(
            config=config,
            times=times,
            interval_minutes=interval,
            arrays=agg["vitals"],
            events=[self._event_dict(e) for e in schedule.sorted()],
            interventions=agg["interventions"],
            pain=agg["pain"],
            pain_labels=agg["pain_labels"],
            stability=agg["stability"],
            sepsis_risk=agg["sepsis_risk"],
            event_matrix=agg["event_matrix"],
            model_meta=self.model_meta,
        )

    def _apply_baseline_calibration(self, vitals: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply validation-fitted per-cohort affine calibration to the baseline.

        The calibration is fitted on held-out validation windows and applied
        before care/event enforcement, so it corrects baseline mean/spread
        without shrinking the enforced clinical events.
        """
        params = self.baseline_calibration.get(config.cohort)
        if not params:
            return vitals
        out = vitals.copy()
        for index, name in enumerate(SIGNALS):
            channel = params.get(name)
            if not channel:
                continue
            out[:, index] = out[:, index] * float(channel["scale"]) + float(channel["shift"])
        out[:, 0] = np.clip(out[:, 0], 40, 250)
        out[:, 1] = np.clip(out[:, 1], 0, 120)
        out[:, 2] = np.clip(out[:, 2], 30, 100)
        return out

    # ------------------------------------------------------------------
    def _generate_neural(
        self,
        config: SimulationConfig,
        schedule: EventSchedule,
        total_minutes: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        assert self.model is not None
        static_raw = _profile_static(config)
        static = torch.from_numpy(((static_raw - self.static_mean) / self.static_std)[None]).to(self.device)
        event_full = schedule.matrix(total_minutes)

        chunk_len = 120
        context_len = 30
        generated = np.zeros((total_minutes, len(SIGNALS)), dtype=np.float32)
        pos = 0
        chunk_index = 0
        while pos < total_minutes:
            if chunk_index == 0:
                ctx_n = 0
                target_start = 0
                take_start = 0
            else:
                ctx_n = min(context_len, pos)
                target_start = pos - ctx_n
                take_start = ctx_n
            target_len = min(chunk_len, total_minutes - target_start)
            events = torch.from_numpy(event_full[target_start : target_start + target_len][None]).to(
                self.device
            )
            time_feats = torch.from_numpy(
                (np.arange(target_len, dtype=np.float32) / 60.0 / 24.0)[None, :, None]
            ).to(self.device)
            mask = torch.ones((1, target_len, len(SIGNALS)), device=self.device)
            context = None
            context_mask = None
            if ctx_n > 0:
                context_np = np.zeros((1, target_len, len(SIGNALS)), dtype=np.float32)
                recent = (generated[pos - ctx_n : pos] - self.channel_mean) / self.channel_std
                context_np[0, :ctx_n] = recent
                context = torch.from_numpy(context_np).to(self.device)
                context_mask = torch.zeros((1, target_len, len(SIGNALS)), device=self.device)
                context_mask[:, :ctx_n] = 1.0
            with torch.no_grad():
                raw = (
                    sample(
                        self.model,
                        static,
                        events,
                        time_feats,
                        mask,
                        context,
                        context_mask,
                        steps=config.sampling_steps,
                        cfg_scale=config.cfg_scale,
                        seed=int(rng.integers(0, 2**31 - 1)),
                        noise_phi=config.noise_phi,
                        noise_scale=config.noise_scale,
                        noise_phi_channels=(
                            config.noise_phi_hr if config.noise_phi_hr is not None else config.noise_phi,
                            config.noise_phi_rr if config.noise_phi_rr is not None else config.noise_phi,
                            config.noise_phi_spo2
                            if config.noise_phi_spo2 is not None
                            else config.noise_phi,
                        ),
                        noise_scale_channels=(
                            config.noise_scale_hr
                            if config.noise_scale_hr is not None
                            else config.noise_scale,
                            config.noise_scale_rr
                            if config.noise_scale_rr is not None
                            else config.noise_scale,
                            config.noise_scale_spo2
                            if config.noise_scale_spo2 is not None
                            else config.noise_scale,
                        ),
                    )[0]
                    .cpu()
                    .numpy()
                )
            raw = raw * self.channel_std + self.channel_mean
            raw = np.nan_to_num(raw, nan=0.0)
            raw[:, 0] = np.clip(raw[:, 0], 40, 250)
            raw[:, 1] = np.clip(raw[:, 1], 0, 120)
            raw[:, 2] = np.clip(raw[:, 2], 30, 100)
            if chunk_index == 0:
                n_take = min(target_len, total_minutes)
                generated[:n_take] = raw[:n_take]
                pos = n_take
            else:
                n_take = min(target_len - take_start, total_minutes - pos)
                if n_take <= 0:
                    break
                generated[pos : pos + n_take] = raw[take_start : take_start + n_take]
                pos += n_take
            chunk_index += 1
            if target_start + target_len >= total_minutes:
                break
        return generated

    def _generate_fallback(
        self, config: SimulationConfig, event_matrix: np.ndarray, total_minutes: int, rng: np.random.Generator
    ) -> np.ndarray:
        ga = config.gestational_age_weeks
        base_hr = float(np.clip(150 - 2.2 * (ga - 30), 115, 175))
        base_rr = float(np.clip(48 - 0.8 * (ga - 30), 30, 70))
        base_spo2 = 96.0
        out = np.zeros((total_minutes, len(SIGNALS)), dtype=np.float32)
        hr, rr, spo2 = base_hr, base_rr, base_spo2
        for t in range(total_minutes):
            hr += 0.08 * (base_hr - hr) + rng.normal(0, 0.8)
            rr += 0.08 * (base_rr - rr) + rng.normal(0, 0.6)
            spo2 += 0.12 * (base_spo2 - spo2) + rng.normal(0, 0.4)
            out[t] = (hr, rr, spo2)
        # Event enforcement in the physiology layer handles the actual shape;
        # here we give it a plausible baseline.
        return out

    @staticmethod
    def _episodes(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        mask = values < threshold
        if not mask.any():
            return []
        padded = np.concatenate(([False], mask, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        return [(int(start), int(end)) for start, end in zip(starts, ends, strict=False)]

    def _closed_loop_responses(
        self,
        arrays: VitalArrays,
        schedule: EventSchedule,
        rng: np.random.Generator,
    ) -> list[ClinicalEvent]:
        """Respond to the generated traces rather than to scheduled labels.

        The controller sees the simulated monitor values, detects episodes, and
        applies oxygen or tactile stimulation after a short clinical delay.  This
        is deliberately a feedback loop: the response is based on what actually
        happened in the generated vitals, not on the event list that conditioned
        the model.
        """
        responses: list[ClinicalEvent] = []
        last_response = {"desaturation": -1e9, "bradycardia": -1e9, "apnea": -1e9}

        for start, end in self._episodes(arrays.spo2, 85.0):
            if start - last_response["desaturation"] < 8.0:
                continue
            minimum = float(np.min(arrays.spo2[start:end]))
            depth = float(np.clip((85.0 - minimum) / 20.0, 0.0, 1.0))
            event = ClinicalEvent(
                start + float(rng.uniform(0.8, 2.0)),
                8.0,
                "oxygen",
                0.45 + 0.45 * depth,
                "Closed-loop oxygen for desaturation",
                {"closed_loop": True, "trigger": "desaturation", "episode_minute": start},
            )
            schedule.add(event)
            if event.start < schedule.shift_minutes:
                responses.append(event)
                last_response["desaturation"] = start

        for start, end in self._episodes(arrays.hr, 100.0):
            if start - last_response["bradycardia"] < 8.0:
                continue
            minimum = float(np.min(arrays.hr[start:end]))
            depth = float(np.clip((100.0 - minimum) / 40.0, 0.0, 1.0))
            event = ClinicalEvent(
                start + float(rng.uniform(0.5, 1.4)),
                5.0,
                "handling",
                0.45 + 0.45 * depth,
                "Closed-loop tactile stimulation for bradycardia",
                {"closed_loop": True, "trigger": "bradycardia", "episode_minute": start},
            )
            schedule.add(event)
            if event.start < schedule.shift_minutes:
                responses.append(event)
                last_response["bradycardia"] = start

        for start, _end in self._episodes(arrays.rr, 5.0):
            if start - last_response["apnea"] < 8.0:
                continue
            event = ClinicalEvent(
                start + float(rng.uniform(0.5, 1.4)),
                5.0,
                "handling",
                0.60,
                "Closed-loop tactile stimulation for apnoea",
                {"closed_loop": True, "trigger": "apnea", "episode_minute": start},
            )
            schedule.add(event)
            if event.start < schedule.shift_minutes:
                responses.append(event)
                last_response["apnea"] = start

        return responses

    # ------------------------------------------------------------------
    def _scores(
        self,
        arrays: VitalArrays,
        schedule: EventSchedule,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
        n = len(arrays.hr)
        baseline_hr = float(np.median(arrays.hr))
        baseline_rr = float(np.median(arrays.rr))
        baseline_temp = float(np.median(arrays.temp))
        pain = np.zeros(n, dtype=np.float32)
        for event in schedule.events:
            if event.type in {"handling", "suction", "pain"}:
                lo = max(0, int(event.start))
                hi = min(n, int(event.start + event.duration + 8))
                if hi > lo:
                    pain[lo:hi] += event.severity * 3.0
        pain += np.clip((arrays.hr - baseline_hr - 12) / 8.0, 0, 3)
        pain += np.clip((arrays.rr - baseline_rr - 5) / 4.0, 0, 2)
        pain += np.clip((93 - arrays.spo2) / 5.0, 0, 2)
        pain = np.clip(pain, 0, 10)
        pain_labels = ["High" if v >= 6 else "Moderate" if v >= 3 else "Low" for v in pain]

        desat = (arrays.spo2 < 85).astype(np.float32)
        brady = (arrays.hr < 100).astype(np.float32)
        stability = (
            100
            - 24 * np.convolve(desat, np.ones(15) / 15, mode="same")
            - 16 * np.convolve(brady, np.ones(15) / 15, mode="same")
            - 8 * np.abs(arrays.temp - baseline_temp)
            - 0.15 * np.abs(arrays.hr - baseline_hr)
        )
        stability = np.clip(stability, 0, 100)
        sepsis = 1 / (
            1
            + np.exp(
                -(
                    -3.0
                    + 2.6 * (config.condition in {"sepsis", "nec"})
                    + 0.025 * (arrays.hr - 155)
                    + 0.045 * (arrays.rr - 50)
                    + 0.35 * (37.2 - arrays.temp)
                    + 0.8 * desat
                )
            )
        )
        return pain, pain_labels, stability, sepsis.astype(np.float32)

    def _aggregate(
        self,
        arrays: VitalArrays,
        schedule: EventSchedule,
        interval: float,
        steps: int,
        pain: np.ndarray,
        stability: np.ndarray,
        sepsis_risk: np.ndarray,
        event_matrix: np.ndarray,
    ) -> dict:
        out = {name: np.zeros(steps, dtype=np.float32) for name in ("hr", "rr", "spo2", "temp")}
        pain_bins = np.zeros(steps, dtype=np.float32)
        stability_bins = np.zeros(steps, dtype=np.float32)
        sepsis_bins = np.zeros(steps, dtype=np.float32)
        event_bins = np.zeros((steps, len(EVENT_TYPES)), dtype=np.float32)
        for i in range(steps):
            lo = int(round(i * interval))
            hi = min(len(arrays.hr), int(round((i + 1) * interval)))
            if hi <= lo:
                hi = min(len(arrays.hr), lo + 1)
            out["hr"][i] = np.mean(arrays.hr[lo:hi])
            out["rr"][i] = np.mean(arrays.rr[lo:hi])
            out["spo2"][i] = np.mean(arrays.spo2[lo:hi])
            out["temp"][i] = np.mean(arrays.temp[lo:hi])
            pain_bins[i] = np.max(pain[lo:hi])
            stability_bins[i] = np.mean(stability[lo:hi])
            sepsis_bins[i] = np.max(sepsis_risk[lo:hi])
            event_bins[i] = np.max(event_matrix[lo:hi], axis=0)
        interventions = []
        for event in schedule.care_events():
            bucket = int(event.start // interval)
            if 0 <= bucket < steps:
                detail = event.label or event.type
                if event.meta.get("medication"):
                    detail = f"{detail} ({event.meta['medication']})"
                if event.meta.get("volume_ml"):
                    detail = f"{detail} · {event.meta['volume_ml']} ml"
                intervention_type = "closed_loop" if event.meta.get("closed_loop") else event.type
                interventions.append(
                    {
                        "timestamp": bucket,
                        "minute": float(event.start),
                        "type": INTERVENTION_MAP.get(intervention_type, intervention_type),
                        "detail": detail,
                        "severity": float(event.severity),
                    }
                )
        pain_labels = ["High" if v >= 6 else "Moderate" if v >= 3 else "Low" for v in pain_bins]
        return {
            "vitals": out,
            "pain": pain_bins,
            "pain_labels": pain_labels,
            "stability": stability_bins,
            "sepsis_risk": sepsis_bins,
            "event_matrix": event_bins,
            "interventions": interventions,
        }

    @staticmethod
    def _event_dict(event) -> dict:
        return {
            "type": event.type,
            "label": event.label,
            "start_minute": round(float(event.start), 2),
            "duration_minutes": round(float(event.duration), 2),
            "severity": round(float(event.severity), 3),
            "meta": event.meta,
        }

    @staticmethod
    def _start_time(config: SimulationConfig) -> datetime:
        now = datetime.now()
        if config.start_time:
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    parsed = datetime.strptime(config.start_time, fmt)
                    return now.replace(
                        hour=parsed.hour, minute=parsed.minute, second=parsed.second, microsecond=0
                    )
                except ValueError:
                    continue
        return now.replace(second=0, microsecond=0)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def to_legacy_json(self, result: SimulationResult) -> dict:
        config = result.config
        if config.cohort == "pics":
            weeks = int(config.gestational_age_weeks)
            days = int(round((config.gestational_age_weeks - weeks) * 7))
            age = f"Post-Conceptual Age: {weeks}/{days} Weeks/Days"
        else:
            age = f"Age: {config.postnatal_age_days:.0f} Days"
        data = []
        prev = {name: None for name in ("hr", "rr", "spo2", "temp")}
        for i, ts in enumerate(result.times):
            vitals = result.vitals_at(i)
            deltas = {}
            for name in ("hr", "rr", "temp"):
                deltas[name] = None if prev[name] is None else round(abs(vitals[name] - prev[name]), 1)
                prev[name] = vitals[name]
            interventions = [
                {"type": item["type"], "detail": item["detail"]}
                for item in result.interventions
                if item["timestamp"] == i
            ]
            entry = {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "vital_signs": {
                    "heart_rate": f"{int(round(vitals['hr']))} bpm",
                    "respiratory_rate": f"{int(round(vitals['rr']))} breaths/min",
                    "body_temperature": f"{vitals['temp']:.1f} °C",
                    "spo2": f"{int(round(vitals['spo2']))} %",
                    "HR_Delta": None if deltas["hr"] is None else f"{deltas['hr']} Δ",
                    "RespRate_Delta": None
                    if deltas["rr"] is None
                    else f"{int(round(deltas['rr']))} Δ breaths/min",
                    "Temp_Delta": None if deltas["temp"] is None else f"{deltas['temp']} °C",
                },
                "pain_level": result.pain_labels[i],
                "stability": round(float(result.stability[i]), 1),
                "sepsis_risk": round(float(result.sepsis_risk[i]), 3),
            }
            if interventions:
                entry["intervention"] = interventions
            data.append(entry)
        return {
            "patient_id": config.patient_id,
            "Sex": config.sex,
            "Age": age,
            "Weight": f"{config.birth_weight_g / 1000:.2f} kg",
            "Conditions": [config.condition],
            "metadata": {
                "generator": "NeoSim 2.0 / NeoFlow",
                "gestational_age_weeks": config.gestational_age_weeks,
                "postnatal_age_days": config.postnatal_age_days,
                "severity": config.severity,
                "shift_hours": config.shift_hours,
                "interval_minutes": config.interval_minutes,
                "cohort": config.cohort,
                "model": result.model_meta,
            },
            "summary": result.summary(),
            "data": data,
        }

    def to_fhir(self, result: SimulationResult) -> dict:
        patient_ref = f"Patient/{result.config.patient_id}"
        entries = [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": result.config.patient_id,
                    "name": [{"text": result.config.name}],
                    "gender": result.config.sex.lower(),
                }
            }
        ]
        codes = {
            "hr": ("8867-4", "Heart rate", "beats/min"),
            "rr": ("9279-1", "Respiratory rate", "breaths/min"),
            "spo2": ("2708-6", "Oxygen saturation", "%"),
            "temp": ("8310-5", "Body temperature", "Cel"),
        }
        for i, ts in enumerate(result.times):
            vitals = result.vitals_at(i)
            for name, (code, display, unit) in codes.items():
                entries.append(
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "status": "final",
                            "category": [{"text": "vital-signs"}],
                            "code": {
                                "coding": [{"system": "http://loinc.org", "code": code, "display": display}]
                            },
                            "subject": {"reference": patient_ref},
                            "effectiveDateTime": ts.isoformat(),
                            "valueQuantity": {"value": round(vitals[name], 1), "unit": unit},
                        }
                    }
                )
        return {"resourceType": "Bundle", "type": "collection", "entry": entries}

    def to_csv(self, result: SimulationResult, directory: str | Path) -> dict[str, Path]:
        import csv

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        vitals_path = directory / "vitals.csv"
        with open(vitals_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "timestamp",
                    "heart_rate",
                    "respiratory_rate",
                    "spo2",
                    "temperature",
                    "pain",
                    "stability",
                    "sepsis_risk",
                ]
            )
            for i, ts in enumerate(result.times):
                v = result.vitals_at(i)
                writer.writerow(
                    [
                        ts.isoformat(),
                        round(v["hr"], 1),
                        round(v["rr"], 1),
                        round(v["spo2"], 1),
                        round(v["temp"], 1),
                        result.pain_labels[i],
                        round(float(result.stability[i]), 1),
                        round(float(result.sepsis_risk[i]), 3),
                    ]
                )
        paths["vitals"] = vitals_path

        events_path = directory / "events.csv"
        with open(events_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["type", "label", "start_minute", "duration_minutes", "severity"])
            for event in result.events:
                writer.writerow(
                    [
                        event["type"],
                        event["label"],
                        event["start_minute"],
                        event["duration_minutes"],
                        event["severity"],
                    ]
                )
        paths["events"] = events_path

        interventions_path = directory / "interventions.csv"
        with open(interventions_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "minute", "type", "detail", "severity"])
            for item in result.interventions:
                ts = (
                    result.times[item["timestamp"]].isoformat()
                    if item["timestamp"] < len(result.times)
                    else ""
                )
                writer.writerow([ts, item["minute"], item["type"], item["detail"], item["severity"]])
        paths["interventions"] = interventions_path
        (directory / "summary.json").write_text(json.dumps(result.summary(), indent=2))
        paths["summary"] = directory / "summary.json"
        return paths
