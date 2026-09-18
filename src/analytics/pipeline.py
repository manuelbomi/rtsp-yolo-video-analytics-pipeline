"""Wires ingest -> detect -> track -> zone-rules -> publish into one loop.

This is the module the analytics container actually runs. Every piece it
touches (:mod:`~src.analytics.ingest`, :mod:`~src.analytics.detector`,
:mod:`~src.analytics.tracker`, :mod:`~src.analytics.zones`,
:mod:`~src.analytics.publisher`) is independently unit-tested; this module
is the thin, mostly-untested glue that runs them in order and is exercised
in practice via the demo video / a real camera, not via pytest.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.analytics.detector import Detector, YoloV8Detector
from src.analytics.ingest import VideoSource
from src.analytics.publisher import EventPublisher, PublisherConfig
from src.analytics.tracker import CentroidIoUTracker
from src.analytics.zones import ZoneRuleEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_detector(config: dict[str, Any]) -> Detector:
    det_cfg = config.get("detector", {})
    classes = det_cfg.get("classes_of_interest")
    return YoloV8Detector(
        model_path=det_cfg.get("model_path", "yolov8n.pt"),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.4),
        classes_of_interest=set(classes) if classes else None,
        device=det_cfg.get("device", "cpu"),
    )


def build_tracker(config: dict[str, Any]) -> CentroidIoUTracker:
    trk_cfg = config.get("tracker", {})
    return CentroidIoUTracker(
        iou_threshold=trk_cfg.get("iou_threshold", 0.3),
        max_centroid_distance=trk_cfg.get("max_centroid_distance", 60.0),
        max_disappeared=trk_cfg.get("max_disappeared", 10),
    )


def build_publisher(config: dict[str, Any]) -> EventPublisher:
    """Build the publisher from config, letting MQTT_HOST/MQTT_PORT env vars
    override the YAML values -- this is how docker-compose points the
    pipeline at the `mosquitto` service without editing the config file."""
    pub_cfg = config.get("publisher", {})
    return EventPublisher(
        PublisherConfig(
            mqtt_host=os.environ.get("MQTT_HOST", pub_cfg.get("mqtt_host", "localhost")),
            mqtt_port=int(os.environ.get("MQTT_PORT", pub_cfg.get("mqtt_port", 1883))),
            mqtt_topic=os.environ.get("MQTT_TOPIC", pub_cfg.get("mqtt_topic", "analytics/events")),
            mqtt_qos=pub_cfg.get("mqtt_qos", 1),
            webhook_url=os.environ.get("WEBHOOK_URL") or pub_cfg.get("webhook_url") or None,
        )
    )


def run_pipeline(config_path: str | Path, max_frames: int | None = None) -> None:
    config = load_config(config_path)
    camera_id = config["camera"]["id"]
    source_uri = config["camera"]["source"]
    loop_video = bool(config["camera"].get("loop", False))

    detector = build_detector(config)
    tracker = build_tracker(config)
    zone_engine = ZoneRuleEngine.from_config(config_path)
    publisher = build_publisher(config)

    logger.info("Starting pipeline for camera_id=%s source=%s", camera_id, source_uri)
    frame_count = 0
    try:
        with VideoSource(source_uri, loop=loop_video) as source:
            for bundle in source.frames():
                detections = detector.detect(bundle.frame)
                tracked_objects = tracker.update(detections)
                timestamp = datetime.fromtimestamp(bundle.timestamp, tz=UTC)
                events = zone_engine.evaluate(camera_id, tracked_objects, timestamp)
                for event in events:
                    logger.info("Event: %s track_id=%s", event.event_type.value, event.track_id)
                    publisher.publish(event)

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
    finally:
        publisher.close()
    logger.info("Pipeline stopped after %d frames", frame_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RTSP/YOLO video analytics pipeline.")
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to the pipeline YAML config (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to process (mainly for demos/smoke tests).",
    )
    args = parser.parse_args()
    run_pipeline(args.config, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
