from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neosim.api import app as app_module
from neosim.simulator.engine import NeoSimulator


@pytest.fixture()
def client(dummy_norm):
    app_module._simulator = NeoSimulator(
        model_path=Path("/nonexistent/model.pt"), norm=dummy_norm, allow_fallback=True
    )
    return TestClient(app_module.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_assets_are_cache_busted(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "app.js?v=" in response.text
    assert "style.css?v=" in response.text
    assert response.headers["cache-control"] == "no-cache"


def test_presets_and_simulate(client):
    assert client.get("/api/presets").status_code == 200
    response = client.post(
        "/api/simulate",
        json={"condition": "stable_preterm", "shift_hours": 1, "interval_minutes": 5, "seed": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["patient_id"] == "SIM-0001"
    assert len(payload["times"]) == 12
    assert len(payload["series"]["hr"]) == 12


def test_export_json(client):
    response = client.post(
        "/api/export?format=json",
        json={"condition": "rds", "shift_hours": 1, "interval_minutes": 5, "seed": 2},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
