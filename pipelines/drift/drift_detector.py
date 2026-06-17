import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def detect_drift(reference_path: str, current_path: str) -> dict:
    reference = pd.read_parquet(reference_path)
    current   = pd.read_parquet(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    drift_share    = result["metrics"][0]["result"]["share_of_drifted_columns"]

    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "report": result,
    }