from datetime import timedelta
from dagster import schedule, RunRequest

from pipelines.ingestion.ingest_pipeline  import ingest_pipeline
from pipelines.feature.feature_pipeline   import feature_pipeline
from pipelines.drift.drift_alert_job      import drift_alert_sensor
from inference.inference_pipeline         import inference_pipeline

def _date_config(context, days_back=1) -> str:
    return (context.scheduled_execution_time - timedelta(days=days_back)).strftime("%Y-%m-%d")

def _run_config(op_name: str, target_date: str) -> dict:
    return {"ops": {op_name: {"config": {"target_date": target_date}}}}

# 01:00 — ingest raw → processed
@schedule(job=ingest_pipeline, cron_schedule="0 1 * * *")
def daily_ingest_schedule(context):
    date = _date_config(context)
    return RunRequest(run_key=f"ingest_{date}",
                      run_config=_run_config("run_spark_ingest", date))

# 02:00 — processed → features → Feast materialize
@schedule(job=feature_pipeline, cron_schedule="0 2 * * *")
def daily_feature_schedule(context):
    date = _date_config(context)
    return RunRequest(run_key=f"feature_{date}",
                      run_config=_run_config("run_spark_feature_job", date))

# 03:00 — batch predict
@schedule(job=inference_pipeline, cron_schedule="0 3 * * *")
def daily_inference_schedule(context):
    date = _date_config(context)
    return RunRequest(run_key=f"inference_{date}",
                      run_config=_run_config("run_batch_predict", date))