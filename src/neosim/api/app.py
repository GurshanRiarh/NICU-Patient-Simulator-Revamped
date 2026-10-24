"""FastAPI application and JSON API for NeoSim 2.0."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from neosim.config import DEFAULT_MODEL, EVENT_TYPES, WEB_DIR, SimulationConfig
from neosim.simulator.engine import NeoSimulator

app = FastAPI(title="NeoSim 2.0", version="2.0.0", description="Generative NICU patient simulator")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

_simulator: NeoSimulator | None = None
_sim_lock = Lock()


class SimulationRequest(BaseModel):
    patient_id: str = "SIM-0001"
    name: str = "Baby Doe"
    gestational_age_weeks: float = Field(30.0, ge=22, le=42)
    birth_weight_g: float = Field(1250.0, ge=400, le=6000)
    postnatal_age_days: float = Field(14.0, ge=0, le=365)
    sex: Literal["Female", "Male"] = "Female"
    condition: str = "stable_preterm"
    severity: float = Field(0.35, ge=0, le=1)
    shift_hours: float = Field(8.0, ge=1, le=48)
    interval_minutes: float = Field(5.0, ge=1, le=30)
    start_time: str | None = None
    seed: int | None = None
    cohort: Literal["auto", "premo", "pics"] = "auto"
    interventions: list[str] | None = None
    care_events: bool = True
    closed_loop: bool = True

    def to_config(self) -> SimulationConfig:
        return SimulationConfig(
            patient_id=self.patient_id,
            name=self.name,
            gestational_age_weeks=self.gestational_age_weeks,
            birth_weight_g=self.birth_weight_g,
            postnatal_age_days=self.postnatal_age_days,
            sex=self.sex,
            condition=self.condition,
            severity=self.severity,
            shift_hours=self.shift_hours,
            interval_minutes=self.interval_minutes,
            start_time=self.start_time,
            seed=self.seed,
            cohort=self.cohort,
            interventions=self.interventions or [],
            care_events=self.care_events,
            closed_loop=self.closed_loop,
        )


def get_simulator() -> NeoSimulator:
    global _simulator
    if _simulator is None:
        with _sim_lock:
            if _simulator is None:
                model = Path(os.environ.get("NEOSIM_MODEL", str(DEFAULT_MODEL)))
                _simulator = NeoSimulator(model_path=model, allow_fallback=True)
    return _simulator


def _run(request: SimulationRequest):
    try:
        result = get_simulator().simulate(request.to_config())
    except Exception as exc:  # surface a useful message to the dashboard
        raise HTTPException(status_code=500, detail=f"simulation failed: {exc}") from exc
    return result


def _payload(result) -> dict:
    return {
        "summary": result.summary(),
        "model": result.model_meta,
        "config": {
            "patient_id": result.config.patient_id,
            "name": result.config.name,
            "condition": result.config.condition,
            "gestational_age_weeks": result.config.gestational_age_weeks,
            "birth_weight_g": result.config.birth_weight_g,
            "postnatal_age_days": result.config.postnatal_age_days,
            "sex": result.config.sex,
            "severity": result.config.severity,
            "shift_hours": result.config.shift_hours,
            "interval_minutes": result.config.interval_minutes,
            "cohort": result.config.cohort,
        },
        "times": [t.isoformat() for t in result.times],
        "series": {key: [round(float(v), 2) for v in value] for key, value in result.arrays.items()},
        "events": result.events,
        "event_types": list(EVENT_TYPES),
        "event_matrix": result.event_matrix.round(2).tolist(),
        "interventions": result.interventions,
        "pain": result.pain_labels,
        "stability": [round(float(v), 1) for v in result.stability],
        "sepsis_risk": [round(float(v), 3) for v in result.sepsis_risk],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {"title": "NeoSim 2.0"})


@app.get("/health")
def health() -> dict:
    sim = get_simulator()
    return {"status": "ok", "model": sim.model_meta.get("model"), "cuda": sim.device.type == "cuda"}


@app.get("/api/model")
def model_info() -> dict:
    return get_simulator().model_meta


@app.get("/api/presets")
def presets() -> dict:
    return {
        "presets": [
            {
                "id": "stable",
                "label": "Stable 30-week",
                "gestational_age_weeks": 30,
                "birth_weight_g": 1250,
                "postnatal_age_days": 14,
                "condition": "stable_preterm",
                "severity": 0.2,
                "sex": "Female",
            },
            {
                "id": "rds",
                "label": "RDS 27-week",
                "gestational_age_weeks": 27,
                "birth_weight_g": 900,
                "postnatal_age_days": 2,
                "condition": "rds",
                "severity": 0.72,
                "sex": "Male",
            },
            {
                "id": "apnea",
                "label": "Apnoea 28-week",
                "gestational_age_weeks": 28,
                "birth_weight_g": 1050,
                "postnatal_age_days": 21,
                "condition": "apnea_of_prematurity",
                "severity": 0.55,
                "sex": "Female",
            },
            {
                "id": "sepsis",
                "label": "Late-onset sepsis",
                "gestational_age_weeks": 34,
                "birth_weight_g": 2100,
                "postnatal_age_days": 18,
                "condition": "sepsis",
                "severity": 0.65,
                "sex": "Male",
            },
            {
                "id": "brady",
                "label": "Recurrent bradycardia",
                "gestational_age_weeks": 26,
                "birth_weight_g": 800,
                "postnatal_age_days": 30,
                "condition": "bradycardia",
                "severity": 0.7,
                "sex": "Female",
            },
        ]
    }


@app.post("/api/simulate")
def simulate(request: SimulationRequest) -> JSONResponse:
    return JSONResponse(_payload(_run(request)))


@app.post("/api/export")
def export(request: SimulationRequest, format: Literal["json", "csv", "fhir"] = "json"):
    result = _run(request)
    sim = get_simulator()
    if format == "json":
        data = json.dumps(sim.to_legacy_json(result), indent=2).encode()
        return Response(
            data,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="neosim.json"'},
        )
    if format == "fhir":
        data = json.dumps(sim.to_fhir(result), indent=2).encode()
        return Response(
            data,
            media_type="application/fhir+json",
            headers={"Content-Disposition": 'attachment; filename="neosim_fhir.json"'},
        )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        paths = sim.to_csv(result, tmp)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in paths.values():
                archive.write(path, arcname=path.name)
        buffer.seek(0)
        return Response(
            buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="neosim_csv.zip"'},
        )
