"""Multispectral feature extraction and Weibull shelf-life inference."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch
from torch import nn

from edge.pipelines.weibull import dispatch_action, expected_rsl_days

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpticalFeatures:
    """Per-frame mean multispectral features."""

    red_mean: float
    green_mean: float
    nir_mean: float
    vigor: float
    water_ratio: float


class WeibullSurvivalHead(nn.Module):
    """Small deterministic baseline head; replace weights with a validated model."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[30.0, 20.0], [0.5, -0.3]]))
            self.projection.bias.copy_(torch.tensor([-6.0, 1.2]))
        self.eval()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map vigor and water ratio to positive scale and shape parameters."""
        raw = self.projection(features)
        positive = torch.nn.functional.softplus(raw) + 0.05
        return positive[..., 0], positive[..., 1]


def _as_gray_frame(frame: np.ndarray, name: str) -> np.ndarray:
    """Validate a band frame and convert color images to a single intensity plane."""
    image = np.asarray(frame)
    if image.size == 0:
        raise ValueError(f"{name} frame is empty")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(image.dtype, np.complexfloating):
        raise ValueError(f"{name} frame must contain real numeric pixel values")
    if not np.isfinite(image).all():
        raise ValueError(f"{name} frame must contain only finite pixel values")
    if image.ndim == 3:
        image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"{name} frame must be a finite 2D array or color image")
    return image.astype(np.float32, copy=False)


def extract_features(red: np.ndarray, green: np.ndarray, nir: np.ndarray) -> OpticalFeatures:
    """Calculate frame means, normalized vegetation vigor, and water ratio."""
    bands = [_as_gray_frame(frame, name) for name, frame in (
        ("red", red), ("green", green), ("nir", nir)
    )]
    if len({band.shape for band in bands}) != 1:
        raise ValueError("Red, green, and NIR frames must have matching dimensions")
    red_mean, green_mean, nir_mean = (float(np.mean(band, dtype=np.float64)) for band in bands)
    vigor_denominator = nir_mean + red_mean
    water_denominator = red_mean + green_mean
    vigor = (nir_mean - red_mean) / vigor_denominator if vigor_denominator > 0 else 0.0
    water_ratio = nir_mean / water_denominator if water_denominator > 0 else 0.0
    return OpticalFeatures(red_mean, green_mean, nir_mean, vigor, water_ratio)


def rgb_frame_to_proxy_bands(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a webcam BGR frame and estimate a demo-only NIR proxy from visible bands."""
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("Webcam frame must be a non-empty BGR color image")
    blue, green, red = cv2.split(frame)
    nir_proxy = cv2.addWeighted(green, 0.65, blue, 0.35, 0.0)
    return red, green, nir_proxy


class InferenceWorker:
    """Run edge inference and optionally publish results to an MQTT broker."""

    def __init__(self, mqtt_host: str | None = None, mqtt_port: int = 1883, topic: str = "harvestcast/scans") -> None:
        self.model = WeibullSurvivalHead()
        self.mqtt_host = mqtt_host or os.getenv("MQTT_HOST") or None
        self.mqtt_port = mqtt_port
        self.topic = topic

    def infer(
        self,
        red: np.ndarray,
        green: np.ndarray,
        nir: np.ndarray,
        batch_id: str,
        produce: str,
        chamber_id: str,
    ) -> dict[str, Any]:
        """Return a complete, API-compatible prediction payload."""
        features = extract_features(red, green, nir)
        preview_bands = [
            cv2.normalize(
                cv2.resize(_as_gray_frame(band, name), (32, 24), interpolation=cv2.INTER_AREA),
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)
            for band, name in ((red, "red"), (green, "green"), (nir, "nir"))
        ]
        preview = np.stack((preview_bands[2], preview_bands[1], preview_bands[0]), axis=-1).tolist()
        model_input = torch.tensor([[features.vigor, features.water_ratio]], dtype=torch.float32)
        with torch.inference_mode():
            scale, shape = self.model(model_input)
        lambda_days = float(scale.item())
        k_shape = float(shape.item())
        rsl = expected_rsl_days(lambda_days, k_shape)
        return {
            "batch_id": batch_id,
            "produce": produce,
            "chamber_id": chamber_id,
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "features": asdict(features),
            "scan_preview": preview,
            "vigor": features.vigor,
            "water_ratio": features.water_ratio,
            "lambda_days": lambda_days,
            "k_shape": k_shape,
            "expected_rsl_days": rsl,
            "dispatch_action": dispatch_action(rsl),
        }

    def publish(self, payload: dict[str, Any]) -> None:
        """Publish an inference result when MQTT is configured; otherwise no-op."""
        if not self.mqtt_host:
            logger.debug("MQTT is not configured; inference result was not published")
            return
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        try:
            client.connect(self.mqtt_host, self.mqtt_port, keepalive=15)
            client.publish(self.topic, json.dumps(payload), qos=1).wait_for_publish(timeout=5)
        except (OSError, ValueError) as exc:
            logger.exception("MQTT publish failed")
            raise RuntimeError("Could not publish inference result to MQTT") from exc
        finally:
            client.disconnect()
