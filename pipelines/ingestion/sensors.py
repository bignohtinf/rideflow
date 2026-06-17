import boto3
import json
from dagster import sensor, RunRequest, SkipReason
from pipelines.ingestion.ingest_pipeline import ingest_pipeline

s3 = boto3.client("s3")
BUCKET = "rideflow"
PREFIX = "raw/rides/"


@sensor(job=ingest_pipeline, minimum_interval_seconds=3600)
def s3_new_file_sensor(context):
    seen_dates: set = set(json.loads(context.cursor)) if context.cursor else set()

    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
    contents = response.get("Contents", [])

    if not contents:
        yield SkipReason("No files found in S3 prefix")
        return

    files = [o["Key"] for o in contents]
    new_runs = False

    for key in files:
        parts = key.split("/")
        if len(parts) < 3:
            continue
        date = parts[2]  

        if date not in seen_dates:
            new_runs = True
            seen_dates.add(date)
            yield RunRequest(
                run_key=date,
                run_config={
                    "ops": {
                        "run_spark_ingest": {"config": {"target_date": date}}
                    }
                },
            )

    context.update_cursor(json.dumps(sorted(seen_dates)))

    if not new_runs:
        yield SkipReason(f"No new dates found (already processed {len(seen_dates)} dates)")
