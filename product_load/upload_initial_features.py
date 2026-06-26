"""Xử lý raw data ban đầu và upload features lên S3.

Thay thế Spark ingest_job + feature_job cho lần setup đầu tiên.
Production sẽ dùng Spark + Dagster để ingest data mới hàng ngày.

Chạy:
    python -m product_load.upload_initial_features
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from data.feature.preprocessing import preprocess_for_training
from data.feature.transformations import build_features

load_dotenv()

RAW_PATH = Path("data/processed/train.parquet")
BUCKET = "rideflow"
TARGET_DATE = "initial"
S3_FEATURES_PATH = f"s3://{BUCKET}/features/{TARGET_DATE}/features.parquet"

STORAGE_OPTIONS = {
    "key": os.environ["AWS_ACCESS_KEY_ID"],
    "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    "client_kwargs": {"region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")},
}


def main():
    logger.info(f"Đọc raw data từ {RAW_PATH}")
    df = pd.read_parquet(RAW_PATH)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    logger.info(f"Raw data: {df.shape[0]:,} rows, {df.shape[1]} cols")

    logger.info("Preprocessing...")
    df = preprocess_for_training(df)

    logger.info("Feature engineering...")
    # use_driver_agg=False: driver_completion_rate_smoothed tính từ target
    # trên cùng dataset → data leakage. Sẽ bật lại khi có historical
    # driver stats từ production data.
    df = build_features(df, use_date=True, use_driver_agg=False)
    logger.info(f"Features ready: {df.shape[0]:,} rows, {df.shape[1]} cols")

    logger.info(f"Upload lên {S3_FEATURES_PATH}")
    df.to_parquet(S3_FEATURES_PATH, index=False, storage_options=STORAGE_OPTIONS)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
