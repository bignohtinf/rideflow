import subprocess
import pandas as pd
import mlflow
from dagster import job, op, In, Failure
from datetime import datetime, timedelta


@op(config_schema={"target_date": str})
def run_batch_predict(context) -> str:
    target_date = context.op_config["target_date"]
    result = subprocess.run(
        ["python", "inference/batch_predict.py", target_date],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        context.log.error(f"Batch predict failed:\n{result.stderr}")
        raise Failure(description=f"Batch predict failed for {target_date}")
    return target_date


@op(ins={"target_date": In(str)})
def log_prediction_metrics(context, target_date: str):
    preds = pd.read_parquet(
        f"s3://rideflow/predictions/{target_date}/predictions.parquet"
    )
    with mlflow.start_run(run_name=f"batch_predict_{target_date}"):
        mlflow.log_metrics({
            "mean_completion_prob": preds["completion_prob"].mean(),
            "pct_high_prob": (preds["completion_prob"] > 0.7).mean(),
            "n_predictions": len(preds),
            "median_completion_prob": preds["completion_prob"].median(),
        })
        mlflow.log_param("date", target_date)

    # Auto-push metrics to Pushgateway after inference
    context.log.info("Pushing prediction metrics to Pushgateway ...")
    result = subprocess.run(
        ["python", "monitoring/push_metrics.py"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        context.log.warning(f"push_metrics failed (non-fatal):\n{result.stderr}")
    else:
        context.log.info(f"push_metrics output:\n{result.stdout}")

@job
def inference_pipeline():
    date = run_batch_predict()
    log_prediction_metrics(date)
