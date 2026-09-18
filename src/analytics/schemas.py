"""Pydantic data models shared across the pipeline.

Keeping the event schema in one place means the detector, tracker, zone
engine, publisher, and dashboard all agree on exactly what a "vision event"
looks like on the wire (MQTT payload, webhook JSON body, WebSocket message).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(str, Enum):
    """Supported zone-rule violation types."""

    RESTRICTED_ZONE_ENTRY = "restricted_zone_entry"
    LINE_CROSSING = "line_crossing"
    DWELL_TIME_EXCEEDED = "dwell_time_exceeded"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates, ``x1 < x2`` and ``y1 < y2``."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_ordering(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"invalid bounding box: x2/y2 must be greater than x1/y1, got {self!r}"
            )
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class VisionEvent(BaseModel):
    """A single zone-rule violation raised for one tracked object on one frame.

    This is the canonical payload published to MQTT/webhook and streamed to
    the live dashboard over WebSocket.
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1, description="Identifier of the source camera/stream.")
    event_type: EventType
    track_id: int = Field(ge=0, description="Stable multi-object-tracker ID.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detector confidence for this object.")
    bbox: BoundingBox
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("camera_id")
    @classmethod
    def _strip_camera_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("camera_id must not be blank")
        return value

    def to_wire_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict suitable for MQTT/webhook/WebSocket payloads."""
        payload = self.model_dump(mode="json")
        return payload
