import io
import os
import sys

import boto3
import numpy as np
import pandas as pd

target_date = sys.argv[1]
BUCKET        = "rideflow"
RAW_KEY       = f"raw/rides/{target_date}/data.parquet"
PROCESSED_KEY = f"processed/{target_date}/data.parquet"

s3 = boto3.client(
    "s3",
    aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name           = os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)

# ── Đọc raw ───────────────────────────────────────────────────────────────────
print(f"Reading s3://{BUCKET}/{RAW_KEY} ...")
obj = s3.get_object(Bucket=BUCKET, Key=RAW_KEY)
df  = pd.read_parquet(io.BytesIO(obj["Body"].read()))
print(f"Raw rows: {len(df):,}")

# ── Làm sạch ─────────────────────────────────────────────────────────────────
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates(subset=["order_id"])
df["date"] = pd.to_datetime(df["date"]).dt.date

# Fill waiting time <= 0 bằng median
median_wait = df.loc[df["user_waiting_time_seconds"] > 0, "user_waiting_time_seconds"].median()
df["user_waiting_time_seconds"] = df["user_waiting_time_seconds"].where(
    df["user_waiting_time_seconds"] > 0, median_wait
)

df["eta_std"] = df.get("eta_std", pd.Series(0.0, index=df.index)).fillna(0.0)
df["eda_std"] = df.get("eda_std", pd.Series(0.0, index=df.index)).fillna(0.0)
df["ingested_at"] = pd.Timestamp.utcnow()

print(f"Processed rows: {len(df):,}")

# ── Ghi processed ─────────────────────────────────────────────────────────────
buf = io.BytesIO()
df.to_parquet(buf, index=False)
buf.seek(0)
s3.put_object(Bucket=BUCKET, Key=PROCESSED_KEY, Body=buf.getvalue())
print(f"Written to: s3://{BUCKET}/{PROCESSED_KEY}")
