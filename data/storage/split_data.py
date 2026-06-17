"""
Chia raw_data.parquet thành 2 phần theo thời gian:
  - train.parquet      → dùng để huấn luyện model offline
  - production.parquet → giả lập data production, sẽ được ingest dần vào pipeline
"""
import pandas as pd

df = pd.read_parquet("data/storage/raw_data.parquet")
df["date"] = pd.to_datetime(df["date"])

# [FIX] BUG: was df.sort_values("date".reset_index(drop=True))
#   — gọi .reset_index() trên string "date" → AttributeError
#   Sửa: sort trước, reset_index sau
df = df.sort_values("date").reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
production_df = df.iloc[split_idx:].copy()

train_df.to_parquet("data/storage/train.parquet", index=False)
production_df.to_parquet("data/storage/production.parquet", index=False)

print(f"Train: {len(train_df):,} rows  |  Production: {len(production_df):,} rows")
print("Split completed")
