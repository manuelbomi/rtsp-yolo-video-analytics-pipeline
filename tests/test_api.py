"""Unit tests for the dashboard FastAPI app.

The MQTT client is stubbed out entirely (no real broker needed) by
monkeypatching ``_build_mqtt_client``. Events are injected directly via
``store.push_event`` to simulate what would otherwise arrive over MQTT.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.dashboard.api as api_module
from src.dashboard.api import app, store


class _DummyMqttClient:
    """Stand-in for paho.mqtt.client.Client that never touches the network."""

    def connect_async(self, *args, **kwargs) -> None:
        pass

    def loop_start(self) -> None:
        pass

    def loop_stop(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _stub_mqtt_and_reset_store(monkeypatch):
    monkeypatch.setattr(api_module, "_build_mqtt_client", lambda: _DummyMqttClient())
    store._buffer.clear()
    store._subscribers.clear()
    yield
    store._buffer.clear()
    store._subscribers.clear()


def sample_event(track_id: int = 1) -> dict:
    return {
        "camera_id": "cam1",
        "event_type": "line_crossing",
        "track_id": track_id,
        "confidence": 0.9,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "timestamp": "2026-01-01T00:00:00Z",
        "metadata": {},
    }


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_events_endpoint_starts_empty() -> None:
    with TestClient(app) as client:
        resp = client.get("/events")
        assert resp.status_code == 200
        assert resp.json() == []


def test_pushed_event_shows_up_in_events_endpoint() -> None:
    with TestClient(app):
        store.push_event(sample_event(track_id=1))
        store.push_event(sample_event(track_id=2))

        client = TestClient(app)
        resp = client.get("/events")
        body = resp.json()
        assert len(body) == 2
        assert [e["track_id"] for e in body] == [1, 2]


def test_ring_buffer_respects_max_size() -> None:
    small_store = api_module.EventStore(max_events=3)
    for i in range(10):
        small_store.push_event(sample_event(track_id=i))
    recent = small_store.recent()
    assert len(recent) == 3
    assert [e["track_id"] for e in recent] == [7, 8, 9]  # only the newest 3 kept


def test_websocket_receives_events_pushed_after_it_connects() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as ws:
            store.push_event(sample_event(track_id=99))
            data = ws.receive_json()
            assert data["track_id"] == 99
            assert data["event_type"] == "line_crossing"
