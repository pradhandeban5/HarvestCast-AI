"""Synchronous OpenCV sources for multispectral camera frames."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

FrameSource: TypeAlias = int | str | Path


class FrameIngestor:
    """Acquire aligned red, green, and near-infrared frames from OpenCV sources."""

    def __init__(self, red_source: FrameSource, green_source: FrameSource, nir_source: FrameSource) -> None:
        self.sources = (red_source, green_source, nir_source)

    @staticmethod
    def _open_source(source: FrameSource) -> cv2.VideoCapture:
        """Open a camera index or video/image file as an OpenCV capture."""
        capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Could not open frame source: {source}")
        return capture

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Capture one frame per band and ensure all channels are aligned."""
        with ExitStack() as stack:
            captures = [stack.enter_context(_CaptureContext(self._open_source(source))) for source in self.sources]
            frames: list[np.ndarray] = []
            for capture in captures:
                success, frame = capture.read()
                if not success or frame is None:
                    raise OSError("A multispectral frame source returned no frame")
                frames.append(frame)
        if len({frame.shape[:2] for frame in frames}) != 1:
            raise ValueError("Red, green, and NIR frames must have matching dimensions")
        return frames[0], frames[1], frames[2]


class _CaptureContext:
    """Release a VideoCapture reliably when used with ExitStack."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture

    def __enter__(self) -> cv2.VideoCapture:
        return self.capture

    def __exit__(self, *_: object) -> None:
        self.capture.release()
