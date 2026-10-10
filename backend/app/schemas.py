"""Validated API request and response models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BatchCreate(BaseModel):
    """Input scan result delivered by the edge worker or simulator."""

    batch_id: str = Field(min_length=1, max_length=64)
    produce: str = Field(min_length=1, max_length=80)
    chamber_id: str = Field(min_length=1, max_length=64)
    measured_at: datetime | None = None
    temperature_c: float | None = Field(default=None, ge=-20.0, le=40.0)
    humidity_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    fan_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    vigor: float = Field(ge=-1.0, le=1.0)
    water_ratio: float = Field(ge=0.0, le=100.0)
    lambda_days: float = Field(gt=0.0)
    k_shape: float = Field(gt=0.0)
    expected_rsl_days: float = Field(ge=0.0)
    dispatch_action: str = Field(min_length=1, max_length=40)
    source: Literal["simulator", "webcam_rgb_proxy", "webcam_rgb_trained"] = "simulator"
    model_status: Literal["demo_untrained", "trained_interval_weibull"] = "demo_untrained"
    predicted_range: str | None = Field(default=None, max_length=64)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    scan_preview: list[list[list[int]]] | None = None


class BatchRead(BatchCreate):
    """Stored scan response."""

    id: str
    measured_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ChamberRead(BaseModel):
    """Current chamber sensor values and HVAC targets."""

    chamber_id: str
    temperature_c: float
    humidity_pct: float
    fan_pct: float
    setpoint_c: float
    target_humidity_pct: float
    target_fan_pct: float
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class WebcamScan(BaseModel):
    """A browser-captured JPEG and the batch context for proxy-band inference."""

    image_base64: str = Field(min_length=64, max_length=4_000_000)
    produce: str = Field(min_length=1, max_length=80)
    chamber_id: str = Field(min_length=1, max_length=64)
