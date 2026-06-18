"""STEP 3 — ML PLATFORM: train + track + register + promote.

Sửa đúng 4 điểm đã nêu trong review so với bản production:
  (1) feature contract: lưu thứ tự cột model học -> serving đọc lại (chống skew).
  (2) calibration THẬT: bọc CalibratedClassifierCV (review: bản gốc để dead code).
  (3) register URI đúng: runs:/{run_id}/model (bản gốc thiếu dấu '/').
  (4) PROMOTE Production: bản gốc chỉ dừng ở Staging nên serve không bao giờ
      load được models:/.../Production. Ở đây promote thẳng Production.
"""
import argparse
import json
import os

# Giới hạn OpenMP threads TRƯỚC khi import lightgbm — tránh LightGBM
# busy-spin/oversubscribe trên WSL khi kết hợp cross-validation.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from lightgbm import LGBMClassifier

from models.training.evaluate import evaluate_oof
from demo.config import (
    FEATURES_PARQUET, FEATURE_NAMES_FILE, MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME, MODEL_NAME, TARGET, ENTITY_KEY, TIMESTAMP_COL, SEED,
)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def get_X_y(df: pd.DataFrame):
    drop = [c for c in (TARGET, ENTITY_KEY, TIMESTAMP_COL) if c in df.columns]
    X = df.drop(columns=drop)
    return X, df[TARGET]


def train(features_path: str) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(features_path)
    X, y = get_X_y(df)
    logger.info(f"[ML] Train trên {len(X):,} dòng, {X.shape[1]} feature")

    # (1) FEATURE CONTRACT — chốt danh sách & thứ tự cột cho serving
    FEATURE_NAMES_FILE.write_text(json.dumps(list(X.columns), indent=2))
    logger.info(f"[ML] Lưu feature contract -> {FEATURE_NAMES_FILE}")

    base = LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=2,
    )

    with mlflow.start_run() as run:
        mlflow.log_params({"model": "lgbm", "n_samples": len(X), "n_features": X.shape[1]})

        # OOF đánh giá honest (không leak)
        oof = cross_val_predict(base, X, y, cv=CV, method="predict_proba")[:, 1]
        metrics = evaluate_oof(y, oof)
        mlflow.log_metrics({
            "auc_roc": metrics["AUC-ROC"], "pr_auc": metrics["PR-AUC"],
            "log_loss": metrics["LogLoss"], "f1": metrics["F1"], "ece": metrics["ECE"],
        })
        logger.info(f"[ML] OOF AUC-ROC={metrics['AUC-ROC']:.4f} | ECE={metrics['ECE']:.4f}")

        # (2) CALIBRATION thật -> model serve ra xác suất đã hiệu chỉnh
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(X, y)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(FEATURE_NAMES_FILE))
        run_id = run.info.run_id

    # (3) + (4) register ĐÚNG URI rồi promote Production
    client = mlflow.MlflowClient()
    mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Production",
        archive_existing_versions=True,
    )
    logger.info(f"[ML] Đăng ký {MODEL_NAME} v{mv.version} -> PRODUCTION")
    return run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEATURES_PARQUET))
    ap.parse_args()
    train(ap.parse_args().features)


if __name__ == "__main__":
    main()
