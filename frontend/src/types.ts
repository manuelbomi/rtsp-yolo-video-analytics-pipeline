/**
 * TypeScript mirror of the backend Pydantic models in
 * `src/analytics/schemas.py`. Keeping this in one place means the
 * dashboard agrees with the analytics pipeline on exactly what a
 * "vision event" looks like on the wire (the same JSON shape used for
 * MQTT, the webhook, `GET /events`, and `WS /ws/live`).
 */

/** Mirrors `EventType` (str Enum) in schemas.py. */
export type EventType =
  | "restricted_zone_entry"
  | "line_crossing"
  | "dwell_time_exceeded";

export const EVENT_TYPES: EventType[] = [
  "restricted_zone_entry",
  "line_crossing",
  "dwell_time_exceeded",
];

/** Mirrors `BoundingBox` in schemas.py: axis-aligned, pixel coordinates. */
export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Mirrors `VisionEvent` in schemas.py. */
export interface VisionEvent {
  camera_id: string;
  event_type: EventType;
  track_id: number;
  confidence: number;
  bbox: BoundingBox;
  /** ISO-8601 timestamp string (pydantic `datetime` serialised to JSON). */
  timestamp: string;
  metadata: Record<string, unknown>;
}

/** Human-readable labels for the event-type badges/chart. */
export const EVENT_TYPE_LABELS: Record<EventType, string> = {
  restricted_zone_entry: "Restricted zone entry",
  line_crossing: "Line crossing",
  dwell_time_exceeded: "Dwell time exceeded",
};
