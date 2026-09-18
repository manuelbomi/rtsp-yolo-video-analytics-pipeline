"""Unit tests for the zone rule engine: point-in-polygon zone entry, line
crossing, and dwell-time rules -- all driven by synthetic point
trajectories, no video/model involved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.analytics.schemas import EventType
from src.analytics.tracker import TrackedObject
from src.analytics.zones import Line, Rule, Zone, ZoneRuleEngine, point_in_polygon, segments_intersect

CAMERA_ID = "cam-test"
SQUARE_ZONE = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def obj_at(point: tuple[float, float], track_id: int = 1) -> TrackedObject:
    x, y = point
    half = 5.0
    return TrackedObject(
        track_id=track_id,
        bbox=(x - half, y - half, x + half, y + half),
        confidence=0.9,
        class_name="person",
        centroid=point,
        age=1,
        hits=1,
    )


def make_zone_engine(rule_type: str, **rule_params) -> ZoneRuleEngine:
    zone = Zone(
        name="restricted",
        polygon=SQUARE_ZONE,
        camera_id=CAMERA_ID,
        rules=[Rule(type=rule_type, params=rule_params)],
    )
    return ZoneRuleEngine(zones=[zone], lines=[])


def make_line_engine() -> ZoneRuleEngine:
    line = Line(
        name="tripwire",
        points=((0.0, 150.0), (300.0, 150.0)),
        camera_id=CAMERA_ID,
        rules=[Rule(type=EventType.LINE_CROSSING.value)],
    )
    return ZoneRuleEngine(zones=[], lines=[line])


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #


def test_point_in_polygon_inside_and_outside() -> None:
    assert point_in_polygon((150, 150), SQUARE_ZONE) is True
    assert point_in_polygon((10, 10), SQUARE_ZONE) is False


def test_segments_intersect_true_and_false() -> None:
    # x=400 is outside the line's x-range [0, 300], so no intersection even
    # though both segments span y=150.
    assert segments_intersect((400, 140), (400, 160), (0, 150), (300, 150)) is False
    assert segments_intersect((150, 140), (150, 160), (0, 150), (300, 150)) is True


# --------------------------------------------------------------------------- #
# restricted_zone_entry
# --------------------------------------------------------------------------- #


def test_restricted_zone_entry_fires_exactly_on_transition_not_every_frame() -> None:
    engine = make_zone_engine(EventType.RESTRICTED_ZONE_ENTRY.value)

    trajectory = [
        (50, 50),  # outside
        (80, 80),  # outside, approaching
        (150, 150),  # ENTER -> should fire here
        (160, 160),  # still inside -> must NOT fire again
        (170, 170),  # still inside -> must NOT fire again
    ]

    fire_flags = []
    for point in trajectory:
        events = engine.evaluate(CAMERA_ID, [obj_at(point)], BASE_TIME)
        fired = any(e.event_type == EventType.RESTRICTED_ZONE_ENTRY for e in events)
        fire_flags.append(fired)

    assert fire_flags == [False, False, True, False, False]


def test_restricted_zone_entry_fires_again_after_leaving_and_reentering() -> None:
    engine = make_zone_engine(EventType.RESTRICTED_ZONE_ENTRY.value)

    trajectory = [
        (50, 50),  # outside
        (150, 150),  # enter -> fire
        (250, 250),  # leave
        (150, 150),  # re-enter -> fire again
    ]
    fire_flags = []
    for p in trajectory:
        events = engine.evaluate(CAMERA_ID, [obj_at(p)], BASE_TIME)
        fire_flags.append(any(e.event_type == EventType.RESTRICTED_ZONE_ENTRY for e in events))
    assert fire_flags == [False, True, False, True]


def test_restricted_zone_entry_never_fires_when_object_stays_outside() -> None:
    engine = make_zone_engine(EventType.RESTRICTED_ZONE_ENTRY.value)
    trajectory = [(0, 0), (10, 400), (400, 10), (5, 5)]
    for point in trajectory:
        events = engine.evaluate(CAMERA_ID, [obj_at(point)], BASE_TIME)
        assert events == []


# --------------------------------------------------------------------------- #
# line_crossing
# --------------------------------------------------------------------------- #


def test_line_crossing_fires_exactly_when_segment_crosses_the_line() -> None:
    engine = make_line_engine()

    # Track walks straight down, from well above the line (y=150) to well
    # below it. Crossing happens between the (150,140)->(150,160) pair.
    trajectory = [
        (150, 100),
        (150, 130),
        (150, 140),
        (150, 160),  # crossing happens on this step
        (150, 200),
    ]

    fire_flags = []
    for point in trajectory:
        events = engine.evaluate(CAMERA_ID, [obj_at(point)], BASE_TIME)
        fire_flags.append(any(e.event_type == EventType.LINE_CROSSING for e in events))

    assert fire_flags == [False, False, False, True, False]


def test_line_crossing_does_not_fire_when_object_stays_on_one_side() -> None:
    engine = make_line_engine()
    trajectory = [(150, 10), (150, 20), (150, 30), (150, 40)]
    for point in trajectory:
        events = engine.evaluate(CAMERA_ID, [obj_at(point)], BASE_TIME)
        assert events == []


# --------------------------------------------------------------------------- #
# dwell_time_exceeded
# --------------------------------------------------------------------------- #


def test_dwell_time_exceeded_fires_once_after_threshold_and_resets_on_exit() -> None:
    engine = make_zone_engine(EventType.DWELL_TIME_EXCEEDED.value, max_dwell_seconds=3.0)

    point_inside = (150, 150)
    point_outside = (0, 0)

    # t=0: enters zone.
    events0 = engine.evaluate(CAMERA_ID, [obj_at(point_inside)], BASE_TIME)
    assert not any(e.event_type == EventType.DWELL_TIME_EXCEEDED for e in events0)

    # t=1s: still under threshold.
    events1 = engine.evaluate(CAMERA_ID, [obj_at(point_inside)], BASE_TIME + timedelta(seconds=1))
    assert not any(e.event_type == EventType.DWELL_TIME_EXCEEDED for e in events1)

    # t=3.5s: exceeds 3s dwell threshold -> fires exactly once.
    events2 = engine.evaluate(CAMERA_ID, [obj_at(point_inside)], BASE_TIME + timedelta(seconds=3.5))
    assert sum(1 for e in events2 if e.event_type == EventType.DWELL_TIME_EXCEEDED) == 1

    # t=4s: already fired for this stay -> must not fire again.
    events3 = engine.evaluate(CAMERA_ID, [obj_at(point_inside)], BASE_TIME + timedelta(seconds=4))
    assert not any(e.event_type == EventType.DWELL_TIME_EXCEEDED for e in events3)

    # Leaves, then re-enters and dwells past threshold again -> fires again.
    engine.evaluate(CAMERA_ID, [obj_at(point_outside)], BASE_TIME + timedelta(seconds=5))
    events_reenter = engine.evaluate(CAMERA_ID, [obj_at(point_inside)], BASE_TIME + timedelta(seconds=6))
    assert not any(e.event_type == EventType.DWELL_TIME_EXCEEDED for e in events_reenter)
    events_reenter_exceeded = engine.evaluate(
        CAMERA_ID, [obj_at(point_inside)], BASE_TIME + timedelta(seconds=9.5)
    )
    assert sum(1 for e in events_reenter_exceeded if e.event_type == EventType.DWELL_TIME_EXCEEDED) == 1
