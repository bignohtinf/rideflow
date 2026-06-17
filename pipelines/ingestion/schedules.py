from dagster import schedule, RunRequest
from datetime import datetime, timedelta
from pipelines.ingestion.ingest_pipeline import ingest_pipeline

@schedule(
    job=ingest_pipeline,
    cron_schedule="0 1 * * *",   # 1am mỗi ngày, sau khi data ngày hôm trước đã lên S3
)
def daily_ingest_schedule(context):
    target_date = (context.scheduled_execution_time - timedelta(days=1)).strftime("%Y-%m-%d")
    return RunRequest(
        run_key=target_date,
        run_config={
            "ops": {
                "run_spark_ingest": {
                    "config": {"target_date": target_date}
                }
            }
        },
    )