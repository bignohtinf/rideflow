import argparse
import pandas as pd
from datetime import date

def shift_dates(input_path: str, output_path: str, date: str = "date"):
    df = pd.read_parquet(input_path)

    if date not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{date}'. Các cột có: {list(df.columns)}")

    df[date] = pd.to_datetime(df[date])

    max_date = df[date].max()
    target_date = pd.Timestamp("2026-06-18")
    offset = target_date - max_date

    print(f"Ngày cuối cùng hiện tại : {max_date.date()}")
    print(f"Ngày cuối cùng mục tiêu : {target_date.date()}")
    print(f"Offset                  : {offset.days} ngày")

    df[date] = df[date] + offset

    print(f"Ngày mới: {df[date].min().date()} → {df[date].max().date()}")

    df.to_parquet(output_path, index=False)
    print(f"Đã lưu: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shift cột date trong file parquet")
    parser.add_argument("--input",    required=True, help="Đường dẫn file parquet đầu vào")
    parser.add_argument("--output",   required=True, help="Đường dẫn file parquet đầu ra")
    parser.add_argument("--date", default="date", help="Tên cột date (mặc định: 'date')")
    args = parser.parse_args()

    shift_dates(args.input, args.output, args.date)
