import pandas as pd
import mlflow
from sklearn.metrics import roc_auc_score, log_loss, f1_score
from monitoring.grafana.metrics_pusher import push_model_metrics
from loguru import logger
import sys

BUCKET = "rideflow"


def track_performance(target_date: str):
    preds = pd.read_parquet(
        f"s3://{BUCKET}/predictions/{target_date}/predictions.parquet"
    )
    actuals = pd.read_parquet(
        f"s3://{BUCKET}/features/{target_date}/features.parquet",
        columns=["order_id", "is_completed"],
    )

    df = preds.merge(actuals, on="order_id", how="inner")
    if len(df) == 0:
        logger.warning("No ground truth available yet")
        return None

    if "predicted_label" not in df.columns:
        logger.warning("predicted_label not found — computing from completion_prob > 0.5")
        df["predicted_label"] = (df["completion_prob"] > 0.5).astype(int)

    metrics = {
        "actual_auc_roc": roc_auc_score(df["is_completed"], df["completion_prob"]),
        "actual_log_loss": log_loss(df["is_completed"], df["completion_prob"]),
        "actual_f1": f1_score(df["is_completed"], df["predicted_label"]),
        "actual_completion_rate": df["is_completed"].mean(),
        "pred_completion_rate": df["predicted_label"].mean(),
        "coverage": len(df),
    }

    with mlflow.start_run(run_name=f"performance_{target_date}"):
        mlflow.log_metrics(metrics)
        mlflow.log_param("date", target_date)

    logger.info(f"Performance logged: AUC={metrics['actual_auc_roc']:.4f}")
    push_model_metrics(metrics, target_date)
    return metrics


if __name__ == "__main__":
    track_performance(sys.argv[1])
