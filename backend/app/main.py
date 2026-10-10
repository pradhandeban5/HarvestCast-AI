"""FastAPI service for produce RSL scans and chamber control."""

import asyncio
import base64
import binascii
import logging
import os
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, inspect as sqlalchemy_inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.controller import optimize_microclimate
from backend.app.database import engine, get_session
from backend.app.models import Base, Chamber, ProduceBatch
from backend.app.schemas import BatchCreate, BatchRead, ChamberRead, WebcamScan
from edge.pipelines.inference_worker import InferenceWorker, rgb_frame_to_proxy_bands
from edge.pipelines.shelf_life_model import TrainedShelfLifePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harvestcast.api")


class TelemetryHub:
    """Fan out scan and control events to connected dashboard clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for connection in tuple(self._connections):
            try:
                await connection.send_json(payload)
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)


hub = TelemetryHub()
webcam_worker = InferenceWorker()
model_path = Path(
    os.getenv(
        "HARVESTCAST_MODEL_PATH",
        str(Path(__file__).resolve().parents[2] / "artifacts" / "shelf_life_mobilenet_v3.pt"),
    )
)
trained_webcam_model = TrainedShelfLifePredictor.load_if_available(model_path)


def _ensure_scan_source_column(connection: Any) -> None:
    """Add model provenance fields when upgrading an existing database in place."""
    columns = {column["name"] for column in sqlalchemy_inspect(connection).get_columns("produce_batches")}
    migrations = {
        "source": "ALTER TABLE produce_batches ADD COLUMN source VARCHAR(32) NOT NULL DEFAULT 'simulator'",
        "model_status": "ALTER TABLE produce_batches ADD COLUMN model_status VARCHAR(40) NOT NULL DEFAULT 'demo_untrained'",
        "predicted_range": "ALTER TABLE produce_batches ADD COLUMN predicted_range VARCHAR(64)",
        "confidence": "ALTER TABLE produce_batches ADD COLUMN confidence FLOAT",
    }
    for column_name, statement in migrations.items():
        if column_name not in columns:
            connection.execute(text(statement))


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create application tables on startup and dispose the engine on shutdown."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_ensure_scan_source_column)
        with suppress(Exception):
            await connection.exec_driver_sql(
                "SELECT create_hypertable('produce_batches', 'measured_at', "
                "if_not_exists => TRUE, migrate_data => TRUE)"
            )
    yield
    await engine.dispose()


app = FastAPI(title="HarvestCast AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness check for containers and orchestration."""
    return {"status": "ok", "service": "harvestcast-api"}


@app.get("/api/v1/model/status")
async def get_model_status() -> dict[str, Any]:
    """Expose trained-model availability and its held-out training metrics."""
    if trained_webcam_model is None:
        return {
            "status": "demo_untrained",
            "checkpoint_available": False,
            "message": "Run edge.train_shelf_life to enable trained webcam predictions.",
        }
    return {
        "status": "trained_interval_weibull",
        "checkpoint_available": True,
        "classes": [item["name"] for item in trained_webcam_model.classes],
        "validation_metrics": trained_webcam_model.validation_metrics,
    }


@app.post("/api/v1/batches", response_model=BatchRead, status_code=201)
async def create_batch(
    batch: BatchCreate, session: AsyncSession = Depends(get_session)
) -> ProduceBatch:
    """Persist a scan, apply its HVAC profile, and broadcast the new state."""
    expected_targets = optimize_microclimate(batch.expected_rsl_days)
    batch_data = batch.model_dump(exclude={"measured_at", "temperature_c", "humidity_pct", "fan_pct"})
    batch_data["dispatch_action"] = expected_targets.mode
    record = ProduceBatch(**batch_data, measured_at=batch.measured_at or datetime.now(timezone.utc))
    chamber_result = await session.execute(
        select(Chamber).where(Chamber.chamber_id == batch.chamber_id)
    )
    chamber = chamber_result.scalar_one_or_none()
    if chamber is None:
        chamber = Chamber(chamber_id=batch.chamber_id)
        session.add(chamber)
    if batch.temperature_c is not None:
        chamber.temperature_c = batch.temperature_c
    if batch.humidity_pct is not None:
        chamber.humidity_pct = batch.humidity_pct
    if batch.fan_pct is not None:
        chamber.fan_pct = batch.fan_pct
    chamber.setpoint_c = expected_targets.temperature_c
    chamber.target_humidity_pct = expected_targets.humidity_pct
    chamber.target_fan_pct = expected_targets.fan_pct
    chamber.updated_at = datetime.now(timezone.utc)
    session.add(record)
    try:
        await session.commit()
        await session.refresh(record)
        await session.refresh(chamber)
    except Exception as exc:
        await session.rollback()
        logger.exception("Could not persist batch scan")
        raise HTTPException(status_code=503, detail="Unable to persist batch scan") from exc

    payload = {
        "type": "scan",
        "batch": BatchRead.model_validate(record).model_dump(mode="json"),
        "chamber": ChamberRead.model_validate(chamber).model_dump(mode="json"),
    }
    await hub.broadcast(payload)
    return record


@app.post("/api/v1/scans/webcam", response_model=BatchRead, status_code=201)
async def scan_webcam(
    request: WebcamScan, session: AsyncSession = Depends(get_session)
) -> ProduceBatch:
    """Infer a demo scan from a real RGB webcam frame using a labeled NIR proxy."""
    try:
        encoded_frame = base64.b64decode(request.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Image must be valid base64 JPEG data") from exc
    image = cv2.imdecode(np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Image could not be decoded")
    height, width = image.shape[:2]
    if width < 32 or height < 32 or width > 1920 or height > 1080:
        raise HTTPException(status_code=422, detail="Image dimensions must be between 32x32 and 1920x1080")

    red, green, nir_proxy = rgb_frame_to_proxy_bands(image)
    prediction = webcam_worker.infer(
        red,
        green,
        nir_proxy,
        batch_id=f"CAM-{uuid4().hex[:8].upper()}",
        produce=request.produce,
        chamber_id=request.chamber_id,
    )
    if trained_webcam_model is not None:
        prediction.update(trained_webcam_model.predict_bgr(image, request.produce))
        prediction["dispatch_action"] = optimize_microclimate(
            prediction["expected_rsl_days"]
        ).mode
        prediction["source"] = "webcam_rgb_trained"
    else:
        prediction["source"] = "webcam_rgb_proxy"
        prediction["model_status"] = "demo_untrained"
    return await create_batch(BatchCreate.model_validate(prediction), session)


@app.get("/api/v1/batches", response_model=list[BatchRead])
async def list_batches(
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[ProduceBatch]:
    """Return the latest scans for the operations dashboard."""
    result = await session.execute(
        select(ProduceBatch).order_by(desc(ProduceBatch.measured_at)).limit(limit)
    )
    return list(result.scalars().all())


@app.get("/api/v1/chambers", response_model=list[ChamberRead])
async def list_chambers(session: AsyncSession = Depends(get_session)) -> list[Chamber]:
    """Return current readings and setpoints for all known chambers."""
    result = await session.execute(select(Chamber).order_by(Chamber.chamber_id))
    return list(result.scalars().all())


@app.post("/api/v1/chambers/{chamber_id}/optimize", response_model=ChamberRead)
async def optimize_chamber(
    chamber_id: str,
    expected_rsl_days: float = Query(ge=0),
    session: AsyncSession = Depends(get_session),
) -> Chamber:
    """Manually recompute target values using a supplied RSL estimate."""
    result = await session.execute(select(Chamber).where(Chamber.chamber_id == chamber_id))
    chamber = result.scalar_one_or_none()
    if chamber is None:
        raise HTTPException(status_code=404, detail="Chamber not found")
    targets = optimize_microclimate(expected_rsl_days)
    chamber.setpoint_c = targets.temperature_c
    chamber.target_humidity_pct = targets.humidity_pct
    chamber.target_fan_pct = targets.fan_pct
    chamber.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(chamber)
    payload = {"type": "chamber", "chamber": ChamberRead.model_validate(chamber).model_dump(mode="json")}
    await hub.broadcast(payload)
    return chamber


@app.websocket("/api/v1/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    """Stream new scans and chamber setpoint updates to connected clients."""
    await hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(websocket)
    except asyncio.CancelledError:
        hub.disconnect(websocket)
        raise
