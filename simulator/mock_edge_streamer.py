"""Generate multispectral scans and virtual PLC telemetry for local demos."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import uuid

import httpx
import numpy as np

from edge.pipelines.inference_worker import InferenceWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harvestcast.simulator")

API_URL = os.getenv("SIMULATOR_API_URL", "http://localhost:8000").rstrip("/")
INTERVAL_SECONDS = max(1.0, float(os.getenv("SIMULATOR_INTERVAL_SECONDS", "4")))
PRODUCE_TYPES = ("Tomatoes", "Avocados", "Spinach", "Strawberries", "Bell peppers")


def synthesize_frames(freshness: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create aligned RGB/NIR intensity frames with controlled degradation."""
    shape = (96, 128)
    red_level = 105.0 + (1.0 - freshness) * 30.0
    green_level = 120.0 - (1.0 - freshness) * 12.0
    nir_level = 145.0 - (1.0 - freshness) * 35.0
    noise = lambda level: np.clip(rng.normal(level, 9.0, shape), 0, 255).astype(np.uint8)
    return noise(red_level), noise(green_level), noise(nir_level)


async def stream_forever() -> None:
    """Run scans through the edge worker and post results to the API."""
    worker = InferenceWorker()
    rng = np.random.default_rng()
    timeout = httpx.Timeout(10.0, connect=4.0)
    async with httpx.AsyncClient(base_url=API_URL, timeout=timeout) as client:
        while True:
            freshness = random.uniform(0.2, 1.0)
            red, green, nir = synthesize_frames(freshness, rng)
            chamber_id = f"COLD-{random.randint(1, 3):02d}"
            prediction = worker.infer(
                red,
                green,
                nir,
                batch_id=f"HC-{uuid.uuid4().hex[:8].upper()}",
                produce=random.choice(PRODUCE_TYPES),
                chamber_id=chamber_id,
            )
            prediction.update(
                temperature_c=round(random.uniform(4.2, 8.8), 1),
                humidity_pct=round(random.uniform(83.0, 96.0), 1),
                fan_pct=round(random.uniform(35.0, 75.0), 1),
            )
            try:
                response = await client.post("/api/v1/batches", json=prediction)
                response.raise_for_status()
                logger.info(
                    "%s | %s | %.1f days | %s",
                    prediction["batch_id"], prediction["produce"],
                    prediction["expected_rsl_days"], prediction["dispatch_action"],
                )
            except (httpx.HTTPError, ValueError):
                logger.exception("Could not deliver simulated scan to %s", API_URL)
            await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(stream_forever())
    except KeyboardInterrupt:
        logger.info("Simulator stopped")
