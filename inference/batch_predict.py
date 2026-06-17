import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from loguru import logger

BUCKET = "rideflow"
TARGET = "is_completed"


def load_model():
    client = mlflow.MlflowClient()
    try:
        prod = client.get_latest_versions("ride_completion", stages=["Production"])[0]
        logger.info(f"Loading production model v{prod.version} (run_id={prod.run_id})")
    except IndexError:
        raise RuntimeError(
            "No production model found in MLflow registry. "
            "Register a model to 'Production' stage first."
        )
    return mlflow.sklearn.load_model(f"models:/ride_completion/Production")

def predict(target_date: str):
    model = load_model()

    feature_path = f"s3://{BUCKET}/features/{target_date}/features.parquet"
    df = pd.read_parquet(feature_path)
    logger.info(f"Loaded {len(df):,} rows from {feature_path}")

    order_ids = df["order_id"] if "order_id" in df.columns else df.index
    X = df.drop(columns=[TARGET, "order_id"], errors="ignore")

    preds = model.predict_proba(X)[:, 1]

    output = pd.DataFrame({
        "order_id": order_ids,
        "completion_prob": preds,
        "predicted_label": (preds > 0.5).astype(int),
        "predicted_at": pd.Timestamp.now(),
    })

    out_path = f"s3://{BUCKET}/predictions/{target_date}/predictions.parquet"
    output.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(output):,} predictions → {out_path}")


if __name__ == "__main__":
    predict(sys.argv[1])
