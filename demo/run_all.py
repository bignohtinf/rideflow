"""RUN ALL — chạy toàn bộ pipeline RideFlow end-to-end trên local.

Đóng vai Dagster cho mục đích DEMO: tuần tự các lớp kiến trúc
(Data -> Feature -> ML -> Serving -> Monitoring) trong MỘT lệnh:

    python -m demo.run_all

Mỗi bước in rõ thuộc lớp nào trong architecture/architecture.png.
"""
import subprocess
import sys
import time

import httpx
import pandas as pd
from loguru import logger

from demo import step1_generate_data as s1
from demo import step2_build_features as s2
from demo import step3_train as s3
from demo import step5_predict as s5
from demo import step6_drift_check as s6
# step4 được import TRỄ trong main() vì feature_repo/features.py đọc schema từ
# parquet ngay lúc import — phải để sau khi step2 đã sinh ra parquet.
from demo.config import (
    RAW_PARQUET, REFERENCE_PARQUET, CURRENT_PARQUET, FEATURES_PARQUET, SEED,
    SERVE_PORT, SERVE_URL,
)


def banner(layer: str, msg: str):
    logger.info("=" * 70)
    logger.info(f"  [{layer}] {msg}")
    logger.info("=" * 70)


def wait_health(url: str, timeout: int = 60) -> bool:
    for _ in range(timeout):
        try:
            if httpx.get(f"{url}/health", timeout=2).json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    # ── DATA PLATFORM ────────────────────────────────────────────────
    banner("DATA PLATFORM", "Sinh dữ liệu raw (mô phỏng S3 + Spark + GE)")
    s1.generate(8_000, SEED).to_parquet(RAW_PARQUET, index=False)
    s1.generate(6_000, SEED).to_parquet(REFERENCE_PARQUET, index=False)        # baseline
    s1.generate(6_000, SEED + 1, drift=True).to_parquet(CURRENT_PARQUET, index=False)  # batch lệch

    # ── FEATURE PLATFORM (offline) ───────────────────────────────────
    banner("FEATURE PLATFORM", "Feature engineering -> offline store (parquet)")
    s2.build(str(RAW_PARQUET)).to_parquet(FEATURES_PARQUET, index=False)

    # ── ML PLATFORM ──────────────────────────────────────────────────
    banner("ML PLATFORM", "Train + MLflow tracking + register + promote Production")
    s3.train(str(FEATURES_PARQUET))

    # ── FEATURE PLATFORM (online) ────────────────────────────────────
    banner("FEATURE PLATFORM", "Materialize features -> online store (Feast/SQLite)")
    from demo import step4_materialize as s4   # import trễ: cần parquet đã tồn tại
    s4.main()

    # ── SERVING ──────────────────────────────────────────────────────
    banner("SERVING", "Khởi động FastAPI + real-time prediction")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "demo.serve:app",
         "--port", str(SERVE_PORT), "--log-level", "warning"],
    )
    try:
        if not wait_health(SERVE_URL):
            logger.error("Serve API không sẵn sàng")
            return
        s5.call(s5.sample_order_ids(5), SERVE_URL)
    finally:
        proc.terminate()
        proc.wait()
        logger.info("Đã tắt serve API")

    # ── MONITORING ───────────────────────────────────────────────────
    banner("MONITORING", "Kiểm tra data drift (PSI) -> feedback loop / retrain")
    s6.detect(pd.read_parquet(REFERENCE_PARQUET), pd.read_parquet(CURRENT_PARQUET))
    s6.main()

    banner("DONE", "Pipeline end-to-end hoàn tất ✅  Xem MLflow tại demo/_mlflow")


if __name__ == "__main__":
    main()
