"""STEP 4 — FEATURE PLATFORM: nạp feature lên ONLINE STORE.

Production: Feast materialize Redshift -> Redis.
Demo: Feast materialize parquet -> SQLite online store.

Đăng ký entity/feature view vào registry (apply) rồi materialize cửa sổ thời
gian chứa event_timestamp của dữ liệu. Dùng API lập trình thay cho CLI `feast`
để không phụ thuộc PATH.
"""
from datetime import datetime

from loguru import logger
from feast import FeatureStore

from demo.config import FEATURE_REPO
from demo.feature_repo import features as fr


def main():
    (FEATURE_REPO / "data").mkdir(parents=True, exist_ok=True)
    store = FeatureStore(repo_path=str(FEATURE_REPO))

    logger.info("[FEAST] apply entity + feature view vào registry ...")
    store.apply([fr.order, fr.order_features])

    store.materialize(
        start_date=datetime(2026, 6, 16),
        end_date=datetime(2026, 6, 18),
    )
    logger.info("[FEAST] Materialize -> online store (SQLite) xong")


if __name__ == "__main__":
    main()
