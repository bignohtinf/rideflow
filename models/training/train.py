import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from models.training.models import build_model
from models.training.evaluate import evaluate_oof

BUCKET = "rideflow"
TARGET = "is_completed"
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

CONFIG_PATH = Path("models/configs/model_params.yaml")

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_features(target_date: str) -> pd.DataFrame:
    return pd.read_parquet(f"s3://{BUCKET}/features/{target_date}/features.parquet")


def get_X_y(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]


def train(target_date: str, model_name: str = "lgbm") -> tuple[str, dict]:
    config = load_config()
    model_config = config.get("model", {})
    eval_config = config.get("evaluation", {})

    mlflow.set_experiment("ride_completion")

    df = load_features(target_date)
    X, y = get_X_y(df)

    model = build_model(model_name, model_config)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "target_date": target_date,
            "model": model_name,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "config_file": str(CONFIG_PATH),
        })

        model_params = model_config.get(model_name, {})
        for k, v in model_params.items():
            mlflow.log_param(f"hp_{k}", v)

        logger.info(f"OOF eval [{model_name}]...")
        oof = cross_val_predict(model, X, y, cv=CV, method="predict_proba")[:, 1]
        metrics = evaluate_oof(y, oof)
        mlflow.log_metrics({
            "auc_roc": metrics["AUC-ROC"],
            "pr_auc": metrics["PR-AUC"],
            "log_loss": metrics["LogLoss"],
            "f1": metrics["F1"],
            "ece": metrics["ECE"],
        })

        thresholds = eval_config.get("thresholds", {})
        min_auc = thresholds.get("auc_roc_min", 0.80)
        if metrics["AUC-ROC"] < min_auc:
            logger.warning(
                f"AUC-ROC {metrics['AUC-ROC']:.4f} below threshold {min_auc}"
            )

        logger.info(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")

        model.fit(X, y)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save reference set for drift detection
        reference = df.groupby(TARGET, group_keys=False).apply(
            lambda g: g.sample(frac=0.2, random_state=SEED)
        )
        ref_path = f"s3://{BUCKET}/features/reference/features.parquet"
        reference.to_parquet(ref_path)

        return run.info.run_id, metrics


if __name__ == "__main__":
    run_id, metrics = train(sys.argv[1], sys.argv[2])
    print(f"run_id: {run_id}")
