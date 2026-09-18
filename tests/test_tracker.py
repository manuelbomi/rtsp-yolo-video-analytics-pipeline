"""Unit tests for the dependency-free centroid/IoU tracker.

All of these feed synthetic ``Detection`` lists frame-by-frame -- no video,
no model, no I/O of any kind.
"""

from __future__ import annotations

from src.analytics.tracker import CentroidIoUTracker, Detection, iou


def box_at(x: float, y: float, w: float = 30, h: float = 60) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def test_iou_of_identical_boxes_is_one() -> None:
    box = box_at(10, 10)
    assert iou(box, box) == 1.0


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou(box_at(0, 0, 10, 10), box_at(1000, 1000, 10, 10)) == 0.0


def test_stable_track_id_persists_across_frames_for_smooth_motion() -> None:
    tracker = CentroidIoUTracker(iou_threshold=0.3, max_disappeared=5)

    # A single object drifting slowly to the right across 6 frames.
    track_ids_seen = []
    for frame_idx in range(6):
        det = Detection(bbox=box_at(10 + frame_idx * 5, 100), confidence=0.9, class_name="person")
        tracked = tracker.update([det])
        assert len(tracked) == 1
        track_ids_seen.append(tracked[0].track_id)

    # Same physical object -> same ID on every single frame.
    assert len(set(track_ids_seen)) == 1
    # And it accumulated one "hit" per frame it was matched.
    assert tracked[0].hits == 6


def test_new_object_gets_a_brand_new_distinct_id() -> None:
    tracker = CentroidIoUTracker()

    det_a = Detection(bbox=box_at(10, 10), confidence=0.9, class_name="person")
    tracked = tracker.update([det_a])
    id_a = tracked[0].track_id

    # Second frame: A moves slightly, and a brand-new object B shows up far away.
    det_a2 = Detection(bbox=box_at(13, 10), confidence=0.9, class_name="person")
    det_b = Detection(bbox=box_at(400, 300), confidence=0.85, class_name="car")
    tracked2 = tracker.update([det_a2, det_b])

    by_class = {t.class_name: t.track_id for t in tracked2}
    assert by_class["person"] == id_a  # A kept its original ID
    assert by_class["car"] != id_a  # B got a different, new ID
    assert len(tracked2) == 2


def test_disappearing_track_id_is_not_reused_by_an_unrelated_object() -> None:
    """A track that goes briefly unmatched (but is still within its
    disappearance tolerance) must keep its own identity in reserve --
    a new, unrelated detection appearing elsewhere must NOT be assigned
    that same ID."""
    tracker = CentroidIoUTracker(max_disappeared=5)

    det_a = Detection(bbox=box_at(10, 10), confidence=0.9, class_name="person")
    tracked = tracker.update([det_a])
    id_a = tracked[0].track_id
    assert id_a in tracker.active_track_ids

    # Frame 2: A is gone (occluded/missed detection), nothing else present.
    tracked_empty = tracker.update([])
    assert tracked_empty == []  # nothing new/matched to report
    assert id_a in tracker.active_track_ids  # still held, within tolerance

    # Frame 3: still no sign of A, but a completely different object C
    # appears far away. It must get a fresh ID, never id_a.
    det_c = Detection(bbox=box_at(500, 400), confidence=0.8, class_name="car")
    tracked_c = tracker.update([det_c])
    assert len(tracked_c) == 1
    id_c = tracked_c[0].track_id
    assert id_c != id_a

    # A is still being held in memory (within tolerance window).
    assert id_a in tracker.active_track_ids


def test_track_is_dropped_after_exceeding_disappearance_tolerance() -> None:
    tracker = CentroidIoUTracker(max_disappeared=2)

    det_a = Detection(bbox=box_at(10, 10), confidence=0.9, class_name="person")
    tracked = tracker.update([det_a])
    id_a = tracked[0].track_id

    # 3 consecutive empty frames > max_disappeared(2) -> track must be dropped.
    tracker.update([])
    tracker.update([])
    tracker.update([])

    assert id_a not in tracker.active_track_ids


def test_reappearing_object_after_tolerance_expires_gets_a_new_id() -> None:
    """Once a track is fully dropped, if the same physical object reappears
    later it is indistinguishable from a new one to this lightweight
    tracker (by design -- see README for the ByteTrack/DeepSORT tradeoff)
    and correctly receives a brand-new ID rather than an incorrect old one."""
    tracker = CentroidIoUTracker(max_disappeared=1)

    det_a = Detection(bbox=box_at(10, 10), confidence=0.9, class_name="person")
    tracked = tracker.update([det_a])
    id_a = tracked[0].track_id

    tracker.update([])  # 1 missed frame
    tracker.update([])  # 2nd missed frame -> exceeds tolerance, dropped

    det_a_again = Detection(bbox=box_at(10, 10), confidence=0.9, class_name="person")
    tracked_again = tracker.update([det_a_again])
    id_new = tracked_again[0].track_id

    assert id_new != id_a


def test_iou_matching_preferred_over_centroid_when_boxes_overlap_strongly() -> None:
    tracker = CentroidIoUTracker(iou_threshold=0.3, max_centroid_distance=5.0)

    det = Detection(bbox=box_at(100, 100), confidence=0.9, class_name="person")
    tracked = tracker.update([det])
    original_id = tracked[0].track_id

    # Moves by 10px -- centroid distance (10) exceeds max_centroid_distance
    # (5), but IoU overlap between old/new boxes is still very high, so
    # the IoU pass alone should match it.
    det_moved = Detection(bbox=box_at(110, 100), confidence=0.9, class_name="person")
    tracked2 = tracker.update([det_moved])

    assert len(tracked2) == 1
    assert tracked2[0].track_id == original_id
