from dagster import sensor, RunRequest, SkipReason
from pipelines.drift.drift_detector import detect_drift
from pipelines.drift.retrain_pipeline import retrain_pipeline
from datetime import datetime, timedelta
from loguru import logger

REFERENCE_PATH = "s3://rideflow/features/reference/features.parquet"
THRESHOLD = 0.3 


@sensor(job=retrain_pipeline, minimum_interval_seconds=86400)
def drift_alert_sensor(context):
    if not context.cursor:
        default_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        context.update_cursor(default_date)
        yield SkipReason("Initialized cursor, will check drift on next tick")
        return

    current_date = context.cursor
    current_path = f"s3://rideflow/features/{current_date}/features.parquet"

    # Advance cursor for next tick
    new_date = (
        datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    context.update_cursor(new_date)

    # [IMPROVED] Wrap in try/except — missing feature file shouldn't crash the sensor
    try:
        result = detect_drift(REFERENCE_PATH, current_path)
    except FileNotFoundError:
        logger.warning(f"Feature file not found for {current_date}, skipping")
        yield SkipReason(f"No feature data for {current_date}")
        return
    except Exception as e:
        logger.error(f"Drift detection failed for {current_date}: {e}")
        yield SkipReason(f"Drift detection error: {e}")
        return

    logger.info(
        f"Drift check {current_date}: detected={result['drift_detected']}, "
        f"share={result['drift_share']:.2%}"
    )

    if result["drift_detected"] and result["drift_share"] > THRESHOLD:
        logger.warning(f"Drift threshold exceeded ({result['drift_share']:.2%}), triggering retrain")
        yield RunRequest(
            run_key=f"retrain_{current_date}",
            run_config={
                "ops": {
                    "run_train": {
                        "config": {
                            "target_date": current_date,
                            "model_name": "lgbm",
                        }
                    }
                }
            },
            tags={"reason": "drift_detected", "drift_share": str(result["drift_share"])},
        )
    else:
        yield SkipReason(
            f"No significant drift for {current_date} "
            f"(share={result['drift_share']:.2%}, threshold={THRESHOLD:.0%})"
        )
