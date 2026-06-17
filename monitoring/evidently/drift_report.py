import pandas as pd
import mlflow
import boto3
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric
from monitoring.grafana.metrics_pusher import push_drift_metrics

s3 = boto3.client('s3')
BUCKET    = "rideflow"
THRESHOLD = 0.2   

def run_drift_report(target_date: str) -> dict:
    reference = pd.read_parquet(f"s3://{BUCKET}/features/reference/features.parquet")
    current   = pd.read_parquet(f"s3://{BUCKET}/features/{target_date}/features.parquet")

    for df in [reference, current]:
        df.drop(columns=["is_completed"], errors="ignore", inplace=True)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current)

    # Lưu HTML report lên S3
    html_path = f"/tmp/drift_report_{target_date}.html"
    report.save_html(html_path)
    s3.upload_file(html_path, BUCKET, f"reports/drift/{target_date}.html")

    result = report.as_dict()
    drift_result = result["metrics"][0]["result"]

    metrics = {
        "drift_detected":  drift_result["dataset_drift"],
        "drift_share":     drift_result["share_of_drifted_columns"],
        "n_drifted":       drift_result["number_of_drifted_columns"],
    }

    # Log vào MLflow
    with mlflow.start_run(run_name=f"drift_{target_date}"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(html_path, artifact_path="drift_reports")

    push_drift_metrics(metrics, target_date)
    return metrics

if __name__ == "__main__":
    import sys
    print(run_drift_report(sys.argv[1]))