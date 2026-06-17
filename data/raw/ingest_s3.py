import boto3
from pathlib import Path

LOCAL_PATH = Path("D:/Vin/GSM/rideflow/data/storage/train.parquet")
BUCKET = "rideflow"
S3_KEY = "raw/train.parquet"

s3 = boto3.client("s3")
print(f"Uploading {LOCAL_PATH.name} ({LOCAL_PATH.stat().st_size / 1e6:.1f} MB)")

s3.upload_file(
    Filename=str(LOCAL_PATH),
    Bucket=BUCKET,
    Key=S3_KEY
)

print(f"Done → s3://{BUCKET}/{S3_KEY}")
