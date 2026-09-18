"""Video source abstraction.

The whole point of this module is that the rest of the pipeline should
never know -- or care -- whether frames are coming from a live RTSP
camera stream or a local demo ``.mp4`` file. Both are just a URI/path
handed to OpenCV's ``VideoCapture``. Config decides which one you get;
the code path is identical.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameBundle:
    """One frame plus a bit of bookkeeping the rest of the pipeline needs."""

    frame: np.ndarray
    frame_index: int
    timestamp: float  # time.time() when the frame was read


class VideoSource:
    """Wraps ``cv2.VideoCapture`` for either an RTSP URL or a local file path.

    Usage::

        with VideoSource("rtsp://user:pass@192.168.1.50:554/stream1") as src:
            for bundle in src.frames():
                ...

        with VideoSource("demo-media/demo.mp4", loop=True) as src:
            for bundle in src.frames():
                ...

    Parameters
    ----------
    uri:
        An RTSP URL (``rtsp://...``) or a filesystem path to a video file.
    loop:
        If True and the source is a finite file, restart from frame 0 when
        it ends instead of stopping iteration. Live RTSP streams ignore
        this (there is no "end" to loop from) but will attempt reconnects.
    reconnect_attempts:
        For RTSP sources, how many times to retry opening the stream after
        a failed read before giving up.
    reconnect_delay_seconds:
        Delay between reconnect attempts.
    """

    def __init__(
        self,
        uri: str,
        loop: bool = False,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 1.0,
    ) -> None:
        self.uri = uri
        self.loop = loop
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._is_stream = uri.strip().lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> VideoSource:
        self._cap = cv2.VideoCapture(self.uri)
        if not self._cap.isOpened():
            raise ConnectionError(f"Unable to open video source: {self.uri!r}")
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0

    def _reconnect(self) -> bool:
        self.close()
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.warning("Reconnecting to %s (attempt %d/%d)", self.uri, attempt, self.reconnect_attempts)
            time.sleep(self.reconnect_delay_seconds)
            self._cap = cv2.VideoCapture(self.uri)
            if self._cap.isOpened():
                return True
        return False

    def frames(self) -> Iterator[FrameBundle]:
        """Yield frames one at a time until the source ends (or forever, for a live/looped source)."""
        if self._cap is None:
            self.open()
        assert self._cap is not None

        frame_index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                if self._is_stream:
                    if self._reconnect():
                        continue
                    logger.error("Giving up on RTSP source %s after repeated failures", self.uri)
                    return
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                    if not ok:
                        return
                else:
                    return

            yield FrameBundle(frame=frame, frame_index=frame_index, timestamp=time.time())
            frame_index += 1
