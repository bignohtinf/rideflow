"""Dagster definitions cho DEMO — portal ORCHESTRATION.

Gói lại pipeline RideFlow thành một Dagster job (`rideflow_demo_pipeline`) với
các op tương ứng từng lớp kiến trúc. Đây là bản thu nhỏ của
`dagster_workspace/` production, nhưng chạy local end-to-end.

Mở Dagster UI (http://localhost:3000) để:
  - xem DAG các op (data -> feature -> ml -> online store)
  - bấm "Materialize all" / "Launch run" để chạy lại pipeline
  - xem lịch chạy hằng ngày (schedule) và log từng op
"""
from dagster import (
    Definitions, job, op, schedule, sensor, ScheduleEvaluationContext,
    SensorEvaluationContext, RunRequest, SkipReason, SensorResult,
    DefaultSensorStatus, get_dagster_logger,
)

from demo import step1_generate_data as s1
from demo import step2_build_features as s2
from demo import step3_train as s3
from demo import step6_drift_check as s6
from demo.drift_push import push_drift
from demo.config import (
    RAW_PARQUET, FEATURES_PARQUET, REFERENCE_PARQUET, CURRENT_PARQUET, SEED,
)


@op
def ingest_raw_op() -> str:
    """DATA PLATFORM: sinh raw rides (mô phỏng S3 + Spark + GE).

    Đồng thời lưu reference baseline + current ban đầu (không drift) để
    drift_sensor có dữ liệu so sánh ngay sau lần chạy đầu.
    """
    df = s1.generate(8_000, SEED)
    df.to_parquet(RAW_PARQUET, index=False)
    # baseline cho drift + current khởi tạo (cùng phân phối -> PSI ~ 0)
    s1.generate(6_000, SEED).to_parquet(REFERENCE_PARQUET, index=False)
    if not CURRENT_PARQUET.exists():
        s1.generate(6_000, SEED).to_parquet(CURRENT_PARQUET, index=False)
    get_dagster_logger().info(f"raw={len(df):,} rows -> {RAW_PARQUET}")
    return str(RAW_PARQUET)


@op
def build_features_op(raw_path: str) -> str:
    """FEATURE PLATFORM (offline): feature engineering -> parquet."""
    feats = s2.build(raw_path)
    feats.to_parquet(FEATURES_PARQUET, index=False)
    get_dagster_logger().info(f"features={feats.shape} -> {FEATURES_PARQUET}")
    return str(FEATURES_PARQUET)


@op
def train_op(features_path: str) -> str:
    """ML PLATFORM: train + MLflow + promote Production."""
    run_id = s3.train(features_path)
    get_dagster_logger().info(f"trained run_id={run_id} -> Production")
    return run_id


@op
def materialize_op(run_id: str):
    """FEATURE PLATFORM (online): materialize -> online store (Feast)."""
    # import trễ: features.py đọc schema parquet lúc import
    from demo import step4_materialize as s4
    s4.main()
    get_dagster_logger().info("materialized -> online store")


@job
def rideflow_demo_pipeline():
    """Data -> Feature -> ML -> Online store (xem trên Dagster UI)."""
    materialize_op(train_op(build_features_op(ingest_raw_op())))


@schedule(cron_schedule="0 2 * * *", job=rideflow_demo_pipeline)
def daily_pipeline_schedule(context: ScheduleEvaluationContext):
    """Lịch chạy 02:00 hằng ngày (giống production)."""
    return RunRequest(run_key=context.scheduled_execution_time.strftime("%Y-%m-%d"))


@sensor(
    job=rideflow_demo_pipeline,
    minimum_interval_seconds=30,
    default_status=DefaultSensorStatus.RUNNING,
)
def drift_sensor(context: SensorEvaluationContext):
    """MONITORING + FEEDBACK LOOP: mỗi ~30s so reference vs current.

    - Luôn push PSI lên Pushgateway -> Grafana cập nhật.
    - Nếu drift (PSI > ngưỡng) và là batch MỚI -> tự trigger retrain pipeline.
      Dùng cursor (chữ ký file current) để không trigger lặp cùng một batch.
    """
    log = context.log
    if not (REFERENCE_PARQUET.exists() and CURRENT_PARQUET.exists()):
        return SkipReason("Chưa có reference/current — chạy pipeline trước.")

    import pandas as pd
    ref = pd.read_parquet(REFERENCE_PARQUET)
    cur = pd.read_parquet(CURRENT_PARQUET)
    result = s6.detect(ref, cur)

    try:
        push_drift(result)
    except Exception as e:
        log.warning(f"Không push được drift metrics: {e}")

    signature = str(CURRENT_PARQUET.stat().st_mtime_ns)
    max_psi = max(result["psi"].values()) if result["psi"] else 0.0
    log.info(f"drift_detected={result['drift_detected']} max_psi={max_psi:.3f}")

    if result["drift_detected"] and context.cursor != signature:
        log.info(f"DRIFT -> trigger retrain (cols: {result['drifted_columns'][:5]})")
        return SensorResult(
            run_requests=[RunRequest(run_key=f"retrain-{signature}")],
            cursor=signature,
        )

    # Không drift hoặc đã trigger cho batch này rồi
    return SensorResult(run_requests=[], cursor=context.cursor)


defs = Definitions(
    jobs=[rideflow_demo_pipeline],
    schedules=[daily_pipeline_schedule],
    sensors=[drift_sensor],
)
