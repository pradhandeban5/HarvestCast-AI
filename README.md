# HarvestCast AI

Edge-to-cloud produce shelf-life intelligence and cold-storage operations.

## Run with Docker

```powershell
docker compose up --build
```

Open the dashboard at <http://localhost:5173> and API documentation at <http://localhost:8000/docs>. The simulator continuously generates aligned multispectral frames, runs edge inference, simulates PLC readings, and posts scans to the API. PostgreSQL/TimescaleDB persists telemetry.

## Run services locally

1. Copy `.env.example` to `.env` and set `DATABASE_URL` to a running TimescaleDB instance. For a local API demo without PostgreSQL, set `DATABASE_URL=sqlite+aiosqlite:///./harvestcast-local.db`.
2. Create and install the Python 3.11 environment: `py -3.11 -m venv .venv`, then `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
3. Start the API with `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload`.
4. In another terminal, run `.venv\Scripts\python.exe -m simulator.mock_edge_streamer`.
5. In `frontend`, run `npm install` then `npm run dev`.

The API accepts scan predictions at `POST /api/v1/batches`, lists batches at `GET /api/v1/batches`, lists storage conditions at `GET /api/v1/chambers`, and streams live updates at `WS /api/v1/ws/telemetry`. Each batch ingestion recomputes the chamber target profile from expected RSL; the dashboard also offers an explicit re-optimization action.

## Webcam demo

Open the dashboard on `http://localhost:5173`, select produce and a chamber, choose **Connect**, grant browser camera permission, then choose **Capture**. The browser sends a JPEG frame to `POST /api/v1/scans/webcam`; the scan is stored and broadcast to the dashboard. A standard RGB webcam cannot measure NIR, so this route derives a visibly labeled demo-only NIR proxy. Its RSL output is not scientifically valid until replaced with an NIR camera and a trained, calibrated model.

## Train the shelf-life model

Install the training dependency with `.venv\Scripts\python.exe -m pip install torchvision`, then validate the folder labels and grouped split with `.venv\Scripts\python.exe -m edge.train_shelf_life --data-dir selflife_dataset --dry-run`. Train with `.venv\Scripts\python.exe -m edge.train_shelf_life --data-dir selflife_dataset --output artifacts/shelf_life_mobilenet_v3.pt --epochs 10`.

Folder labels such as `Apple(1-5)` are trained as shelf-life intervals; `Expired` is a separate class. The trainer removes exact duplicate images and rejects conflicting duplicate labels, groups neighboring `frameNNN` images into the same train/validation partition, and writes held-out metrics beside the checkpoint. The API loads that checkpoint at startup and reports its status at `/api/v1/model/status`; restart the API after training. Webcam predictions use RGB and are limited by the source data, lighting, camera setup, and label quality. The reported validation is not a guarantee of field accuracy, especially for low-count or poorly performing classes.

## Edge inputs and outputs

`edge.pipelines.frame_ingestion.FrameIngestor` reads synchronized red, green, and NIR frames from camera indices or file sources. `InferenceWorker.infer` extracts vigor and water ratios, predicts Weibull scale/shape parameters, computes the analytic expected RSL and FEFO action, and can publish to MQTT when `MQTT_HOST` is configured.

The inference head ships with deterministic demonstration weights, not produce-specific trained weights. Replace these with validated model weights and calibrate the output against labeled shelf-life trials before using its predictions for commercial handling. The threshold HVAC policy is a supervisory setpoint recommendation; it does not replace chamber interlocks or equipment-level safety controls.

## Checks

```powershell
python -m unittest discover -s tests
python -m compileall -q backend edge simulator
```