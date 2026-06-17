import subprocess
from dagster import job, op, In, Failure, RetryPolicy
from pipelines.expectations.rides_suite import build_suite
from pipelines.expectations.checkpoint import run_checkpoint
from datetime import datetime, timedelta
from feast import FeatureStore


@op(
    config_schema={"target_date": str},
    retry_policy=RetryPolicy(max_retries=2, delay=30),
)
def run_spark_feature_job(context) -> str:
    target_date = context.op_config["target_date"]
    result = subprocess.run(
        ["spark-submit", "pipelines/spark/feature_job.py", target_date],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        context.log.error(f"Spark feature job failed:\n{result.stderr}")
        raise Failure(description=f"Spark feature job failed for {target_date}")

    context.log.info(result.stdout)
    return target_date


@op(ins={"target_date": In(str)})
def run_gx_validation(target_date: str) -> str:
    build_suite("data/raw/schemas/expectations.json")
    parquet_path = f"s3://rideflow/features/{target_date}/features.parquet"
    run_checkpoint(parquet_path, "rides_processed_suite")
    return target_date


@op(ins={"target_date": In(str)})
def run_feast_materialize(target_date: str):
    store = FeatureStore(repo_path="data/feature")
    store.materialize(
        start_date=datetime.strptime(target_date, "%Y-%m-%d"),
        end_date=datetime.strptime(target_date, "%Y-%m-%d") + timedelta(hours=24),
    )
    print(f"Materialized {target_date} → Redshift + Redis")
    
@job
def feature_pipeline():
    date = run_spark_feature_job()
    date = run_gx_validation(date)
    run_feast_materialize(date)
