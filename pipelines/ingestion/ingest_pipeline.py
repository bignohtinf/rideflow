import subprocess
from dagster import job, op, In, Failure, RetryPolicy
from pipelines.expectations.rides_suite import build_suite
from pipelines.expectations.checkpoint import run_checkpoint


@op(
    config_schema={"target_date": str},
    retry_policy=RetryPolicy(max_retries=2, delay=30),
)
def run_spark_ingest(context) -> str:
    target_date = context.op_config["target_date"]
    result = subprocess.run(
        ["spark-submit", "pipelines/spark/ingest_job.py", target_date],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        context.log.error(f"Spark ingest failed:\n{result.stderr}")
        raise Failure(description=f"Spark ingest failed for {target_date}")

    context.log.info(f"Spark ingest completed for {target_date}")
    return target_date


@op(ins={"target_date": In(str)})
def run_gx_ingest_validation(target_date: str) -> str:
    build_suite("data/raw/schemas/expectations.json")
    run_checkpoint(
        f"s3://rideflow/processed/{target_date}/data.parquet",
        "rides_processed_suite",
    )
    return target_date


@job
def ingest_pipeline():
    date = run_spark_ingest()
    run_gx_ingest_validation(date)
