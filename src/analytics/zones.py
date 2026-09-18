"""Zone rule engine.

Loads polygon zones and crossing lines (plus the rules attached to them)
from ``config/pipeline.yaml`` and evaluates every tracked object, every
frame, against those rules. A rule that fires produces a
:class:`~src.analytics.schemas.VisionEvent`.

Supported rule types
---------------------
``restricted_zone_entry``
    Fires the moment a tracked object's centroid transitions from
    *outside* a polygon to *inside* it (edge-triggered -- it does not fire
    again every subsequent frame the object stays inside).

``line_crossing``
    Fires when the segment between a tracked object's centroid on the
    previous frame and the current frame crosses a configured line
    segment (e.g. a virtual tripwire across a doorway).

``dwell_time_exceeded``
    Fires once when a tracked object has remained continuously inside a
    polygon for longer than ``max_dwell_seconds``.

Geometry (point-in-polygon, segment intersection) is implemented with
plain arithmetic -- no shapely/GEOS dependency -- so the whole engine is
easy to unit test with synthetic point trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.analytics.schemas import BoundingBox, EventType, VisionEvent
from src.analytics.tracker import TrackedObject

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray-casting point-in-polygon test. ``polygon`` is a list of ``(x, y)`` vertices."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1-p2 intersects segment p3-p4 (standard orientation test)."""
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


# --------------------------------------------------------------------------- #
# Config models
# --------------------------------------------------------------------------- #


@dataclass
class Rule:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Zone:
    name: str
    polygon: list[Point]
    camera_id: str | None = None  # None => applies to every camera
    rules: list[Rule] = field(default_factory=list)


@dataclass
class Line:
    name: str
    points: tuple[Point, Point]
    camera_id: str | None = None
    rules: list[Rule] = field(default_factory=list)


def _parse_rules(raw_rules: list[dict[str, Any]]) -> list[Rule]:
    rules: list[Rule] = []
    for raw in raw_rules or []:
        raw = dict(raw)
        rule_type = raw.pop("type")
        rules.append(Rule(type=rule_type, params=raw))
    return rules


def load_zone_config(path: str | Path) -> tuple[list[Zone], list[Line]]:
    """Parse the ``zones``/``lines`` sections of a pipeline config YAML file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    zones = [
        Zone(
            name=z["name"],
            polygon=[tuple(pt) for pt in z["polygon"]],
            camera_id=z.get("camera_id"),
            rules=_parse_rules(z.get("rules", [])),
        )
        for z in raw.get("zones", [])
    ]
    lines = [
        Line(
            name=ln["name"],
            points=(tuple(ln["points"][0]), tuple(ln["points"][1])),
            camera_id=ln.get("camera_id"),
            rules=_parse_rules(ln.get("rules", [])),
        )
        for ln in raw.get("lines", [])
    ]
    return zones, lines


# --------------------------------------------------------------------------- #
# Per-track bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class _TrackState:
    zones_inside: set[str] = field(default_factory=set)
    zone_entry_time: dict[str, float] = field(default_factory=dict)
    dwell_fired: set[str] = field(default_factory=set)
    last_point: Point | None = None


class ZoneRuleEngine:
    """Evaluates configured zone/line rules against tracked objects, frame by frame."""

    def __init__(self, zones: list[Zone], lines: list[Line]) -> None:
        self.zones = zones
        self.lines = lines
        self._state: dict[tuple[str, int], _TrackState] = {}

    @classmethod
    def from_config(cls, path: str | Path) -> ZoneRuleEngine:
        zones, lines = load_zone_config(path)
        return cls(zones, lines)

    def _state_for(self, camera_id: str, track_id: int) -> _TrackState:
        key = (camera_id, track_id)
        if key not in self._state:
            self._state[key] = _TrackState()
        return self._state[key]

    def reset_track(self, camera_id: str, track_id: int) -> None:
        """Forget bookkeeping for a track (e.g. once the tracker drops it)."""
        self._state.pop((camera_id, track_id), None)

    def evaluate(
        self,
        camera_id: str,
        tracked_objects: list[TrackedObject],
        timestamp: datetime | None = None,
    ) -> list[VisionEvent]:
        """Evaluate all configured rules for every tracked object on this frame.

        Returns the list of :class:`VisionEvent` produced (possibly empty).
        """
        ts = timestamp or datetime.now(UTC)
        events: list[VisionEvent] = []

        for obj in tracked_objects:
            state = self._state_for(camera_id, obj.track_id)
            current_point = obj.centroid

            for zone in self.zones:
                if zone.camera_id is not None and zone.camera_id != camera_id:
                    continue
                is_inside = point_in_polygon(current_point, zone.polygon)
                was_inside = zone.name in state.zones_inside

                for rule in zone.rules:
                    if rule.type == EventType.RESTRICTED_ZONE_ENTRY.value:
                        if is_inside and not was_inside:
                            events.append(
                                self._make_event(
                                    camera_id, obj, EventType.RESTRICTED_ZONE_ENTRY, ts, {"zone": zone.name}
                                )
                            )
                    elif rule.type == EventType.DWELL_TIME_EXCEEDED.value:
                        max_dwell = float(rule.params.get("max_dwell_seconds", 5.0))
                        if is_inside:
                            entry_time = state.zone_entry_time.setdefault(zone.name, ts.timestamp())
                            dwell = ts.timestamp() - entry_time
                            if dwell >= max_dwell and zone.name not in state.dwell_fired:
                                state.dwell_fired.add(zone.name)
                                events.append(
                                    self._make_event(
                                        camera_id,
                                        obj,
                                        EventType.DWELL_TIME_EXCEEDED,
                                        ts,
                                        {"zone": zone.name, "dwell_seconds": round(dwell, 2)},
                                    )
                                )

                if is_inside:
                    state.zones_inside.add(zone.name)
                else:
                    state.zones_inside.discard(zone.name)
                    state.zone_entry_time.pop(zone.name, None)
                    state.dwell_fired.discard(zone.name)

            if state.last_point is not None:
                for line in self.lines:
                    if line.camera_id is not None and line.camera_id != camera_id:
                        continue
                    crossed = segments_intersect(
                        state.last_point, current_point, line.points[0], line.points[1]
                    )
                    if not crossed:
                        continue
                    for rule in line.rules:
                        if rule.type == EventType.LINE_CROSSING.value:
                            events.append(
                                self._make_event(
                                    camera_id, obj, EventType.LINE_CROSSING, ts, {"line": line.name}
                                )
                            )

            state.last_point = current_point

        return events

    @staticmethod
    def _make_event(
        camera_id: str,
        obj: TrackedObject,
        event_type: EventType,
        ts: datetime,
        metadata: dict[str, Any],
    ) -> VisionEvent:
        x1, y1, x2, y2 = obj.bbox
        return VisionEvent(
            camera_id=camera_id,
            event_type=event_type,
            track_id=obj.track_id,
            confidence=obj.confidence,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            timestamp=ts,
            metadata={**metadata, "class_name": obj.class_name},
        )
