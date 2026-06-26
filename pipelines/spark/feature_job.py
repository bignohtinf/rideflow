import io
import logging
import os
import sys

import boto3
import pandas as pd
from dotenv import load_dotenv

from data.feature.transformations import build_features
from data.feature.preprocessing import preprocess_for_training

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BUCKET = "rideflow"


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python feature_job.py <yyyy-mm-dd>")
    target_date = sys.argv[1]

    s3 = boto3.client(
        "s3",
        aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name           = os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    processed_key = f"processed/{target_date}/data.parquet"
    feature_key   = f"features/{target_date}/features.parquet"

    # ── Đọc processed ────────────────────────────────────────────────────────
    logging.info(f"Reading s3://{BUCKET}/{processed_key} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=processed_key)
    df  = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    logging.info(f"Read {len(df):,} rows")

    # ── Feature engineering ───────────────────────────────────────────────────
    df = preprocess_for_training(df)
    df = build_features(df, use_date=True, use_driver_agg=True)
    logging.info(f"Feature engineering complete: {df.shape[1]} features, {len(df):,} rows")

    # ── Ghi features ─────────────────────────────────────────────────────────
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=feature_key, Body=buf.getvalue())
    logging.info(f"Written to: s3://{BUCKET}/{feature_key}")


if __name__ == "__main__":
    main()
