"""Event publishing: MQTT (always) and an optional webhook (HTTP POST).

Keeping this as its own module means the zone engine never touches
networking directly -- it just produces ``VisionEvent`` objects, and
whatever is configured to consume them (MQTT broker, webhook receiver,
both, or in tests, neither) is a deployment detail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import paho.mqtt.client as mqtt
import requests

from src.analytics.schemas import VisionEvent

logger = logging.getLogger(__name__)


@dataclass
class PublisherConfig:
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "analytics/events"
    mqtt_client_id: str = "vision-analytics-publisher"
    mqtt_qos: int = 1
    webhook_url: str | None = None
    webhook_timeout_seconds: float = 3.0


class EventPublisher:
    """Publishes :class:`VisionEvent`s to MQTT and, optionally, a webhook.

    The MQTT client is connected lazily on first publish so constructing a
    publisher (e.g. in tests, with a stubbed client) never touches the
    network.
    """

    def __init__(
        self,
        config: PublisherConfig,
        mqtt_client: mqtt.Client | None = None,
        http_session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self._mqtt_client = mqtt_client
        self._http_session = http_session or requests.Session()
        self._connected = False

    def _ensure_mqtt_connected(self) -> mqtt.Client:
        if self._mqtt_client is None:
            self._mqtt_client = mqtt.Client(client_id=self.config.mqtt_client_id)
        if not self._connected:
            self._mqtt_client.connect(self.config.mqtt_host, self.config.mqtt_port)
            self._connected = True
        return self._mqtt_client

    def publish(self, event: VisionEvent) -> None:
        """Publish one event to MQTT, then (best-effort) to the webhook."""
        payload = json.dumps(event.to_wire_dict())
        self._publish_mqtt(payload)
        if self.config.webhook_url:
            self._publish_webhook(payload)

    def _publish_mqtt(self, payload: str) -> None:
        try:
            client = self._ensure_mqtt_connected()
            client.publish(self.config.mqtt_topic, payload=payload, qos=self.config.mqtt_qos)
        except Exception:  # noqa: BLE001 - never let a broker outage kill the pipeline
            logger.exception(
                "Failed to publish event to MQTT broker %s:%s",
                self.config.mqtt_host,
                self.config.mqtt_port,
            )

    def _publish_webhook(self, payload: str) -> None:
        try:
            self._http_session.post(
                self.config.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.webhook_timeout_seconds,
            )
        except Exception:  # noqa: BLE001 - webhook delivery is best-effort
            logger.exception("Failed to POST event to webhook %s", self.config.webhook_url)

    def close(self) -> None:
        if self._mqtt_client is not None and self._connected:
            try:
                self._mqtt_client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._connected = False
