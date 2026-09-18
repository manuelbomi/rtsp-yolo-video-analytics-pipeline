"""FastAPI dashboard service.

Subscribes to the MQTT topic the analytics pipeline publishes
``VisionEvent``s to, keeps the most recent N of them in an in-memory ring
buffer, and re-broadcasts every event in real time to any browser
connected over ``WS /ws/live``.

Endpoints
---------
``GET /health``
    Liveness probe.
``GET /events``
    Recent events currently held in the ring buffer (newest last).
``WS /ws/live``
    Streams every new event, as JSON, to the client as it arrives.

The MQTT connection is established on startup on a best-effort basis: if
no broker is reachable (e.g. running the dashboard alone, or under
pytest) the app still serves ``/health`` and ``/events`` -- it just never
receives anything, and events can be pushed in directly via
``push_event`` for testing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

MAX_EVENTS = int(os.environ.get("DASHBOARD_MAX_EVENTS", "500"))
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "analytics/events")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class EventStore:
    """Thread-safe-enough ring buffer + WebSocket fan-out.

    MQTT callbacks fire on paho's own network thread while WebSocket
    consumers live on the asyncio event loop, so pushing a new event has
    to hop threads via ``run_coroutine_threadsafe``.
    """

    def __init__(self, max_events: int = MAX_EVENTS) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._subscribers: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def recent(self) -> list[dict[str, Any]]:
        return list(self._buffer)

    async def subscribe(self, ws: WebSocket) -> None:
        self._subscribers.add(ws)

    def unsubscribe(self, ws: WebSocket) -> None:
        self._subscribers.discard(ws)

    def push_event(self, event: dict[str, Any]) -> None:
        """Add an event to the buffer and fan it out to WebSocket clients.

        Safe to call from any thread.
        """
        self._buffer.append(event)
        if self._loop is None:
            return
        for ws in list(self._subscribers):
            asyncio.run_coroutine_threadsafe(self._safe_send(ws, event), self._loop)

    async def _safe_send(self, ws: WebSocket, event: dict[str, Any]) -> None:
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001 - a dead socket just gets dropped
            self._subscribers.discard(ws)


store = EventStore()


def _on_mqtt_message(_client: mqtt.Client, _userdata: Any, msg: mqtt.MQTTMessage) -> None:
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Dropping malformed MQTT payload on topic %s", msg.topic)
        return
    store.push_event(payload)


def _build_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(client_id="vision-dashboard")
    client.on_message = _on_mqtt_message

    def _on_connect(client: mqtt.Client, _userdata: Any, _flags: Any, _rc: int) -> None:
        client.subscribe(MQTT_TOPIC)
        logger.info("Dashboard subscribed to MQTT topic %s", MQTT_TOPIC)

    client.on_connect = _on_connect
    return client


_mqtt_client: mqtt.Client | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mqtt_client
    store.bind_loop(asyncio.get_running_loop())
    try:
        _mqtt_client = _build_mqtt_client()
        _mqtt_client.connect_async(MQTT_HOST, MQTT_PORT)
        _mqtt_client.loop_start()
        logger.info("Connecting to MQTT broker %s:%s", MQTT_HOST, MQTT_PORT)
    except Exception:  # noqa: BLE001 - dashboard must still serve without a broker
        logger.warning("Could not start MQTT client; dashboard will run without live events", exc_info=True)
        _mqtt_client = None
    yield
    if _mqtt_client is not None:
        _mqtt_client.loop_stop()
        try:
            _mqtt_client.disconnect()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="Video Analytics Dashboard", lifespan=lifespan)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/events")
def get_events() -> list[dict[str, Any]]:
    return store.recent()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket) -> None:
    await ws.accept()
    await store.subscribe(ws)
    try:
        while True:
            # We don't expect client -> server messages, but reading keeps
            # the connection alive and lets us detect a disconnect promptly.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe(ws)
