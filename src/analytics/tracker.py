"""A small, dependency-free multi-object tracker.

This is intentionally NOT DeepSORT/ByteTrack. It is a greedy IoU-first,
centroid-fallback tracker that assigns a persistent, monotonically
increasing integer ID to each detection across frames. It has zero
third-party dependencies beyond the standard library, which makes it easy
to read, easy to unit test with synthetic detections, and easy to reason
about in an interview setting. See the README "Design decisions" section
for the tradeoffs versus a production-grade tracker.

Matching strategy per frame:
  1. Build an IoU matrix between existing tracks' last-known bbox and the
     new detections.
  2. Greedily assign the highest-IoU pairs first, as long as IoU clears
     ``iou_threshold``.
  3. Any tracks/detections still unmatched are given a second chance using
     normalised centroid distance (handles small/fast-moving boxes where
     IoU can be zero between consecutive frames).
  4. Detections still unmatched become brand-new tracks with a fresh ID.
  5. Tracks still unmatched get a "disappeared" counter bumped; once that
     counter exceeds ``max_disappeared`` frames the track is dropped for
     good. Its ID is never reassigned to a different physical object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    """One detector output for a single frame."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Track:
    """Internal, mutable state the tracker keeps for one tracked object."""

    track_id: int
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    hits: int = 1
    disappeared: int = 0
    age: int = 0
    centroid_history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class TrackedObject:
    """Public, immutable view of a track handed back to callers for one frame."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    centroid: tuple[float, float]
    age: int
    hits: int


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class CentroidIoUTracker:
    """Greedy IoU + centroid-distance multi-object tracker.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU for two boxes to be considered the same object.
    max_centroid_distance:
        Maximum centroid movement (pixels) allowed for the distance-based
        fallback match, used when IoU is too low (e.g. small/fast objects).
    max_disappeared:
        Number of *consecutive* frames a track may go unmatched before it
        is dropped permanently. While a track is within this tolerance
        window it still occupies its ID; that ID is never handed to a
        different, unrelated detection.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_centroid_distance: float = 60.0,
        max_disappeared: int = 10,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_centroid_distance = max_centroid_distance
        self.max_disappeared = max_disappeared
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1  # monotonically increasing, never reused

    @property
    def active_track_ids(self) -> set[int]:
        return set(self._tracks.keys())

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Advance the tracker by one frame and return the current tracks.

        Only tracks matched (or newly created) *this* frame are returned;
        tracks that are within their disappearance tolerance but were not
        matched this frame are kept internally but not emitted, since there
        is nothing new to report about them.
        """
        unmatched_track_ids = set(self._tracks.keys())
        unmatched_detection_idxs = set(range(len(detections)))
        matches: dict[int, int] = {}  # track_id -> detection index

        # Pass 1: greedy IoU matching, highest IoU first.
        iou_candidates: list[tuple[float, int, int]] = []
        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            for det_idx in unmatched_detection_idxs:
                score = iou(track.bbox, detections[det_idx].bbox)
                if score >= self.iou_threshold:
                    iou_candidates.append((score, track_id, det_idx))
        iou_candidates.sort(key=lambda c: c[0], reverse=True)

        claimed_tracks: set[int] = set()
        claimed_dets: set[int] = set()
        for _score, track_id, det_idx in iou_candidates:
            if track_id in claimed_tracks or det_idx in claimed_dets:
                continue
            matches[track_id] = det_idx
            claimed_tracks.add(track_id)
            claimed_dets.add(det_idx)

        unmatched_track_ids -= claimed_tracks
        unmatched_detection_idxs -= claimed_dets

        # Pass 2: centroid-distance fallback for what's left.
        dist_candidates: list[tuple[float, int, int]] = []
        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            for det_idx in unmatched_detection_idxs:
                dist = centroid_distance(track.centroid, detections[det_idx].centroid)
                if dist <= self.max_centroid_distance:
                    dist_candidates.append((dist, track_id, det_idx))
        dist_candidates.sort(key=lambda c: c[0])

        claimed_tracks2: set[int] = set()
        claimed_dets2: set[int] = set()
        for _dist, track_id, det_idx in dist_candidates:
            if track_id in claimed_tracks2 or det_idx in claimed_dets2:
                continue
            matches[track_id] = det_idx
            claimed_tracks2.add(track_id)
            claimed_dets2.add(det_idx)

        unmatched_track_ids -= claimed_tracks2
        unmatched_detection_idxs -= claimed_dets2

        results: list[TrackedObject] = []

        # Update matched tracks.
        for track_id, det_idx in matches.items():
            det = detections[det_idx]
            track = self._tracks[track_id]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.class_name = det.class_name
            track.hits += 1
            track.disappeared = 0
            track.age += 1
            track.centroid_history.append(det.centroid)
            results.append(
                TrackedObject(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    class_name=track.class_name,
                    centroid=track.centroid,
                    age=track.age,
                    hits=track.hits,
                )
            )

        # Age out unmatched tracks; drop those past tolerance.
        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            track.disappeared += 1
            track.age += 1
            if track.disappeared > self.max_disappeared:
                del self._tracks[track_id]

        # Spawn brand-new tracks for leftover detections.
        for det_idx in unmatched_detection_idxs:
            det = detections[det_idx]
            new_id = self._next_id
            self._next_id += 1
            track = Track(
                track_id=new_id,
                bbox=det.bbox,
                class_name=det.class_name,
                confidence=det.confidence,
                centroid_history=[det.centroid],
            )
            self._tracks[new_id] = track
            results.append(
                TrackedObject(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    class_name=track.class_name,
                    centroid=track.centroid,
                    age=track.age,
                    hits=track.hits,
                )
            )

        return results
