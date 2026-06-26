"""
Nạp dữ liệu production theo giờ vào pipeline.
Đọc production.parquet (đã có cột datetime từ add_time.py),
lọc theo giờ hiện tại (hoặc chỉ định), upload lên S3 để trigger Dagster sensor.

Cách dùng:
  # Chạy 1 lần cho giờ hiện tại
  python product_load/trans_data.py

  # Chạy cho ngày/giờ cụ thể
  python product_load/trans_data.py --date 2026-03-11 --hour 14

  # Chạy tất cả các giờ trong 1 ngày (backfill)
  python product_load/trans_data.py --date 2026-03-11 --all-hours

  # Chạy liên tục mỗi giờ (daemon mode)
  python product_load/trans_data.py --daemon
"""

import argparse
import io
import time
from datetime import datetime, timedelta

import boto3
import pandas as pd

BUCKET = "rideflow"
S3_PREFIX = "raw/rides"
PRODUCTION_FILE = "data/processed/raw_production.parquet"


def load_production_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        raise ValueError(
            "Cột 'datetime' không tồn tại. Chạy add_time.py trước:\n"
            "  python product_load/add_time.py --input data/storage/production.parquet "
            "--output data/storage/production.parquet"
        )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def filter_by_hour(df: pd.DataFrame, target_date: str, target_hour: int) -> pd.DataFrame:
    target = pd.to_datetime(target_date)
    mask = (df["datetime"].dt.date == target.date()) & (df["datetime"].dt.hour == target_hour)
    return df[mask].copy()


def upload_to_s3(df: pd.DataFrame, date_str: str) -> str:
    s3_key = f"{S3_PREFIX}/{date_str}/data.parquet"
    s3 = boto3.client("s3")

    # Nếu đã có file cho ngày này → append thêm rows
    try:
        existing = s3.get_object(Bucket=BUCKET, Key=s3_key)
        existing_df = pd.read_parquet(io.BytesIO(existing["Body"].read()))
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
        print(f"  Append vào file có sẵn ({len(existing_df)} → {len(df)} rows)")
    except s3.exceptions.NoSuchKey:
        pass
    except Exception:
        pass  # File chưa tồn tại → tạo mới

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=s3_key, Body=buf.getvalue())
    return f"s3://{BUCKET}/{s3_key}"


def ingest_hour(df: pd.DataFrame, target_date: str, target_hour: int):
    batch = filter_by_hour(df, target_date, target_hour)
    if batch.empty:
        print(f"[{target_date} {target_hour:02d}:00] Không có dữ liệu → bỏ qua")
        return

    s3_path = upload_to_s3(batch, target_date)
    print(f"[{target_date} {target_hour:02d}:00] {len(batch):,} rows → {s3_path}")


def run_all_hours(df: pd.DataFrame, target_date: str):
    print(f"Backfill toàn bộ 24 giờ cho ngày {target_date}")
    for h in range(24):
        ingest_hour(df, target_date, h)


def run_daemon(df: pd.DataFrame, interval_seconds: int = 3600):
    print(f"Daemon mode — chạy mỗi {interval_seconds}s. Ctrl+C để dừng.")
    while True:
        now = datetime.now()
        target_date = now.strftime("%Y-%m-%d")
        target_hour = now.hour
        print(f"\n{'='*50}")
        print(f"[{now.isoformat()}] Bắt đầu nạp dữ liệu...")
        ingest_hour(df, target_date, target_hour)
        print(f"Đợi {interval_seconds}s đến lần tiếp theo...")
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Nạp dữ liệu production theo giờ vào S3")
    parser.add_argument("--input", default=PRODUCTION_FILE, help="File production.parquet")
    parser.add_argument("--date", default=None, help="Ngày cần nạp (YYYY-MM-DD, mặc định: hôm nay)")
    parser.add_argument("--hour", type=int, default=None, help="Giờ cần nạp (0-23, mặc định: giờ hiện tại)")
    parser.add_argument("--all-hours", action="store_true", help="Nạp tất cả 24 giờ trong ngày")
    parser.add_argument("--daemon", action="store_true", help="Chạy liên tục mỗi giờ")
    parser.add_argument("--interval", type=int, default=3600, help="Khoảng cách giữa các lần nạp (giây, cho daemon)")
    args = parser.parse_args()

    now = datetime.now()
    target_date = args.date or now.strftime("%Y-%m-%d")
    target_hour = args.hour if args.hour is not None else now.hour

    print(f"Đọc dữ liệu từ {args.input}...")
    df = load_production_data(args.input)
    print(f"Tổng: {len(df):,} rows, từ {df['datetime'].min()} đến {df['datetime'].max()}")

    if args.daemon:
        run_daemon(df, args.interval)
    elif args.all_hours:
        run_all_hours(df, target_date)
    else:
        ingest_hour(df, target_date, target_hour)

    print("\nHoàn thành.")


if __name__ == "__main__":
    main()
