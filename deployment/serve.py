import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
from loguru import logger

from data.feature.transformations import add_interaction_features, add_time_features

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model and feature store...")
    try:
        app.state.model = mlflow.sklearn.load_model("models:/ride_completion/Production")
        app.state.store = FeatureStore(repo_path="data/feature/")
        logger.info("Model and feature store loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down serve API")


app = FastAPI(
    title="RideFlow Prediction API",
    lifespan=lifespan,
)

class PredictRequest(BaseModel):
    order_id: str


@app.post("/predict")
def predict(request: PredictRequest):
    start = time.time()

    try:
        features = app.state.store.get_online_features(
            features=[
                "order_raw_features:num_drivers",
                "order_raw_features:num_orders",
                "order_raw_features:eta_avg",
                "order_raw_features:eta_std",
                "order_raw_features:eta_min",
                "order_raw_features:eda_avg",
                "order_raw_features:eda_std",
                "order_raw_features:eda_min",
                "order_raw_features:distance",
                "order_raw_features:total_fee",
                "order_raw_features:hour_of_day",
                "order_raw_features:minute_of_hour",
                "order_raw_features:rush_hour",
                "order_raw_features:user_waiting_time_seconds",
                "order_derived_features:supply_demand_ratio",
                "order_derived_features:demand_supply_ratio",
                "order_derived_features:eta_confidence",
                "order_derived_features:eda_confidence",
                "order_derived_features:fee_per_km",
                "order_derived_features:eta_per_km",
                "order_derived_features:eta_eda_ratio",
                "order_derived_features:pickup_to_trip_ratio",
                "order_derived_features:is_short_trip",
                "order_derived_features:is_long_eta",
                "order_derived_features:is_high_wait",
                "order_derived_features:is_negative_wait",
                "order_derived_features:is_single_driver",
            ],
            entity_rows=[{"order_id": request.order_id}],
        ).to_df()
    except Exception as e:
        logger.error(f"Feature retrieval failed for {request.order_id}: {e}")
        raise HTTPException(status_code=503, detail="Feature store unavailable")

    features = features.drop(columns=["order_id"], errors="ignore")
    if features.isnull().all(axis=None):
        raise HTTPException(
            status_code=404,
            detail=f"No features found for order_id={request.order_id}",
        )

    # Tính on-demand interactions — dùng chung transformations.py với training
    features = add_time_features(features)
    features = add_interaction_features(features)

    try:
        prob = app.state.model.predict_proba(features)[:, 1][0]
    except Exception as e:
        logger.error(f"Prediction failed for {request.order_id}: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

    latency_ms = (time.time() - start) * 1000

    return {
        "order_id": request.order_id,
        "completion_prob": round(float(prob), 4),
        "latency_ms": round(latency_ms, 1),
    }


@app.get("/health")
def health():
    has_model = hasattr(app.state, "model") and app.state.model is not None
    has_store = hasattr(app.state, "store") and app.state.store is not None
    status = "ok" if (has_model and has_store) else "degraded"
    return {"status": status, "model_loaded": has_model, "store_loaded": has_store}
