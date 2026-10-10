"""Database entities for scans and controlled storage chambers."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for application models."""


class ProduceBatch(Base):
    """A timestamped optical scan and its survival-model prediction."""

    __tablename__ = "produce_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    batch_id: Mapped[str] = mapped_column(String(64), index=True)
    produce: Mapped[str] = mapped_column(String(80), index=True)
    chamber_id: Mapped[str] = mapped_column(String(64), index=True)
    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), primary_key=True
    )
    vigor: Mapped[float] = mapped_column(Float)
    water_ratio: Mapped[float] = mapped_column(Float)
    lambda_days: Mapped[float] = mapped_column(Float)
    k_shape: Mapped[float] = mapped_column(Float)
    expected_rsl_days: Mapped[float] = mapped_column(Float)
    dispatch_action: Mapped[str] = mapped_column(String(40))
    source: Mapped[str] = mapped_column(String(32), default="simulator", server_default="simulator")
    model_status: Mapped[str] = mapped_column(
        String(40), default="demo_untrained", server_default="demo_untrained"
    )
    predicted_range: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    scan_preview: Mapped[list[list[list[int]]] | None] = mapped_column(JSON, nullable=True)


class Chamber(Base):
    """Latest measured and commanded conditions for one cold room."""

    __tablename__ = "chambers"

    chamber_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    temperature_c: Mapped[float] = mapped_column(Float, default=6.0)
    humidity_pct: Mapped[float] = mapped_column(Float, default=90.0)
    fan_pct: Mapped[float] = mapped_column(Float, default=60.0)
    setpoint_c: Mapped[float] = mapped_column(Float, default=6.0)
    target_humidity_pct: Mapped[float] = mapped_column(Float, default=90.0)
    target_fan_pct: Mapped[float] = mapped_column(Float, default=60.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
