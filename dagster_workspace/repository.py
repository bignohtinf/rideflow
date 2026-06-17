from dagster import Definitions

from pipelines.ingestion.ingest_pipeline  import ingest_pipeline
from pipelines.ingestion.sensors          import s3_new_file_sensor
from pipelines.feature.feature_pipeline   import feature_pipeline
from pipelines.drift.drift_alert_job      import drift_alert_sensor
from inference.inference_pipeline         import inference_pipeline
from pipelines.drift.retrain_pipeline     import retrain_pipeline

from dagster_workspace.schedules          import (
    daily_ingest_schedule,
    daily_feature_schedule,
    daily_inference_schedule,
)

defs = Definitions(
    jobs=[
        ingest_pipeline,
        feature_pipeline,
        inference_pipeline,
        retrain_pipeline,
    ],
    schedules=[
        daily_ingest_schedule,
        daily_feature_schedule,
        daily_inference_schedule,
    ],
    sensors=[
        s3_new_file_sensor,
        drift_alert_sensor,
    ],
)