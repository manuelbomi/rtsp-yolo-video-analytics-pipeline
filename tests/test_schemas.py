"""Unit tests for the VisionEvent pydantic schema."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.analytics.schemas import BoundingBox, EventType, VisionEvent


def valid_event_kwargs() -> dict:
    return dict(
        camera_id="cam1",
        event_type=EventType.RESTRICTED_ZONE_ENTRY,
        track_id=42,
        confidence=0.87,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=90),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"zone": "restricted_area"},
    )


def test_valid_event_passes_validation() -> None:
    event = VisionEvent(**valid_event_kwargs())
    assert event.camera_id == "cam1"
    assert event.event_type == EventType.RESTRICTED_ZONE_ENTRY
    assert event.bbox.as_tuple() == (10, 10, 50, 90)


def test_event_type_accepts_plain_string_value() -> None:
    kwargs = valid_event_kwargs()
    kwargs["event_type"] = "line_crossing"
    event = VisionEvent(**kwargs)
    assert event.event_type == EventType.LINE_CROSSING


def test_wire_dict_is_json_serialisable_and_round_trips_key_fields() -> None:
    event = VisionEvent(**valid_event_kwargs())
    payload = event.to_wire_dict()
    assert payload["camera_id"] == "cam1"
    assert payload["track_id"] == 42
    assert payload["event_type"] == "restricted_zone_entry"
    assert payload["bbox"]["x2"] == 50


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 1.5},  # out of [0,1] range
        {"confidence": -0.1},
        {"track_id": -1},  # negative track id
        {"camera_id": ""},  # blank camera id
        {"camera_id": "   "},  # whitespace-only camera id
        {"event_type": "not_a_real_event_type"},
        {"metadata": {"unexpected_extra_field_at_top_level": 1}, "unexpected_top_level": True},
    ],
)
def test_invalid_event_data_raises_validation_error(overrides: dict) -> None:
    kwargs = valid_event_kwargs()
    kwargs.update(overrides)
    with pytest.raises(ValidationError):
        VisionEvent(**kwargs)


def test_bounding_box_rejects_degenerate_or_inverted_coordinates() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x1=50, y1=10, x2=10, y2=90)  # x2 < x1

    with pytest.raises(ValidationError):
        BoundingBox(x1=10, y1=10, x2=10, y2=90)  # zero width


def test_bounding_box_centroid() -> None:
    box = BoundingBox(x1=0, y1=0, x2=10, y2=20)
    assert box.centroid == (5.0, 10.0)
