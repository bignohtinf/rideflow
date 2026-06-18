"""STEP 5a — SERVING: FastAPI real-time prediction (bản DEMO local).

Khác bản production deployment/serve.py ở chỗ ĐÚNG feature parity:
serving lấy đúng danh sách feature trong feature contract mà model đã học
(không liệt kê tay, không bỏ sót feature on-demand như bản gốc).

Chạy:
    uvicorn demo.serve:app --port 8000
"""
import json
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from feast import FeatureStore
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ── Prometheus metrics (Grafana sẽ vẽ từ các metric này) ──────────────
PRED_TOTAL = Counter(
    "rideflow_predictions_total", "Tổng số dự đoán", ["predicted_label"]
)
PRED_ERRORS = Counter("rideflow_prediction_errors_total", "Số request lỗi")
PRED_LATENCY = Histogram(
    "rideflow_prediction_latency_seconds", "Độ trễ /predict",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PRED_PROB = Histogram(
    "rideflow_completion_prob", "Phân phối xác suất hoàn thành",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

from demo.config import (
    MLFLOW_TRACKING_URI, MODEL_NAME, FEATURE_NAMES_FILE, FEATURE_REPO, ENTITY_KEY,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("Đang load model Production + feature store...")
    app.state.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    app.state.store = FeatureStore(repo_path=str(FEATURE_REPO))
    # FEATURE CONTRACT: serving dùng đúng list này -> không skew
    app.state.feature_names = json.loads(FEATURE_NAMES_FILE.read_text())
    app.state.feast_refs = [f"order_features:{f}" for f in app.state.feature_names]
    logger.info(f"Sẵn sàng. {len(app.state.feature_names)} feature theo contract.")
    yield
    logger.info("Tắt serve API")


app = FastAPI(title="RideFlow Demo API", lifespan=lifespan)


class PredictRequest(BaseModel):
    order_id: str


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()
    try:
        feats = app.state.store.get_online_features(
            features=app.state.feast_refs,
            entity_rows=[{ENTITY_KEY: req.order_id}],
        ).to_df()
    except Exception as e:
        PRED_ERRORS.inc()
        logger.error(f"Lỗi đọc online features cho {req.order_id}: {e}")
        raise HTTPException(status_code=503, detail="Feature store unavailable")

    feats = feats.drop(columns=[ENTITY_KEY], errors="ignore")
    if feats.isnull().all(axis=None):
        PRED_ERRORS.inc()
        raise HTTPException(status_code=404, detail=f"Không có feature cho {req.order_id}")

    # Reindex đúng thứ tự contract -> đảm bảo parity với lúc train
    X = feats.reindex(columns=app.state.feature_names)
    prob = float(app.state.model.predict_proba(X)[:, 1][0])
    label = int(prob > 0.5)

    latency = time.time() - start
    PRED_LATENCY.observe(latency)
    PRED_PROB.observe(prob)
    PRED_TOTAL.labels(predicted_label=str(label)).inc()

    return {
        "order_id": req.order_id,
        "completion_prob": round(prob, 4),
        "predicted_label": label,
        "latency_ms": round(latency * 1000, 1),
    }


@app.get("/metrics")
def metrics():
    """Endpoint cho Prometheus scrape (pull model)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    ok = hasattr(app.state, "model") and hasattr(app.state, "store")
    return {"status": "ok" if ok else "degraded", "model_loaded": ok}
