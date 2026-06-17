import argparse
import pandas as pd

def add_time_column(
    input_path: str,
    output_path: str,
    hour_col: str = "hour_of_day",
    minute_col: str = "minute_of_hour",
    date_col: str = "date",
    out_col: str = "datetime",
):
    df = pd.read_parquet(input_path)

    # Kiểm tra cột bắt buộc
    for col in [hour_col, minute_col]:
        if col not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{col}'. Các cột có: {list(df.columns)}")

    hour   = df[hour_col].fillna(0).astype(int)
    minute = df[minute_col].fillna(0).astype(int)

    # Cột chuỗi HH:MM (luôn tạo)
    df["time_str"] = hour.astype(str).str.zfill(2) + ":" + minute.astype(str).str.zfill(2)
    print(f"Đã tạo cột 'time_str'  (VD: {df['time_str'].iloc[0]})")

    # Nếu có cột date → kết hợp thành datetime đầy đủ
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df[out_col] = df[date_col] + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
        print(f"Đã tạo cột '{out_col}' (VD: {df[out_col].iloc[0]})")
    else:
        # Không có date → tạo timedelta
        df["time_delta"] = pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
        print(f"Không tìm thấy cột date '{date_col}' → đã tạo cột 'time_delta' (VD: {df['time_delta'].iloc[0]})")

    df.to_parquet(output_path, index=False)
    print(f"Đã lưu: {output_path}")
    print(f"\nCác cột hiện có: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thêm cột giờ vào file parquet")
    parser.add_argument("--input",      required=True,           help="File parquet đầu vào")
    parser.add_argument("--output",     required=True,           help="File parquet đầu ra")
    parser.add_argument("--hour-col",   default="hour_of_day",   help="Cột giờ (mặc định: hour_of_day)")
    parser.add_argument("--minute-col", default="minute_of_hour",help="Cột phút (mặc định: minute_of_hour)")
    parser.add_argument("--date-col",   default="date",          help="Cột date để kết hợp (mặc định: date)")
    parser.add_argument("--out-col",    default="datetime",      help="Tên cột output (mặc định: datetime)")
    args = parser.parse_args()

    add_time_column(
        input_path=args.input,
        output_path=args.output,
        hour_col=args.hour_col,
        minute_col=args.minute_col,
        date_col=args.date_col,
        out_col=args.out_col,
    )
