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

    # Only keep features the model was trained on
    expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
    if expected_features is not None:
        extra = set(X.columns) - set(expected_features)
        if extra:
            logger.warning(f"Dropping {len(extra)} unexpected columns: {extra}")
            X = X[[c for c in expected_features if c in X.columns]]
        missing = set(expected_features) - set(X.columns)
        if missing:
            logger.warning(f"Missing {len(missing)} expected columns: {missing}")

    # Convert HH:MM string columns sang minutes
    for col in X.columns:
        if X[col].dtype == object:
            hhmm_mask = X[col].dropna().str.match(r"^\d{1,2}:\d{2}$", na=False)
            if hhmm_mask.mean() > 0.5:
                def _to_minutes(x):
                    if pd.isna(x):
                        return None
                    try:
                        h, m = str(x).split(":")
                        return int(h) * 60 + int(m)
                    except (ValueError, AttributeError):
                        return None
                X[col] = X[col].apply(_to_minutes).astype(float)
                logger.info(f"Converted column '{col}' from HH:MM to minutes")

    # Drop any remaining non-numeric columns CatBoost can't handle
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        logger.warning(f"Dropping non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

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
