import pandas as pd

df = pd.read_parquet("data/storage/raw_data.parquet")
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
production_df = df.iloc[split_idx:].copy()

train_df.to_parquet("data/storage/train.parquet", index=False)
production_df.to_parquet("data/storage/production.parquet", index=False)

print(f"Train: {len(train_df):,} rows  |  Production: {len(production_df):,} rows")
print("Split completed")
