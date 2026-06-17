import pandas as pd

LEAKAGE_COLS = ["est_time_arrival", "est_distance_arrival", "estimate_dropoff_time", "total_pay"]
ID_COLS      = ["order_id", "matching_batch_id", "driver_id"]
TARGET       = "is_completed"

def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in LEAKAGE_COLS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df

def remove_id_columns(df: pd.DataFrame, keep_driver_id: bool = False) -> pd.DataFrame:
    cols_to_drop = [c for c in ID_COLS if c in df.columns]
    if keep_driver_id and "driver_id" in cols_to_drop:
        cols_to_drop.remove("driver_id")
    return df.drop(columns=cols_to_drop)

def fix_negative_waiting_time(df: pd.DataFrame) -> pd.DataFrame:
    col = "user_waiting_time_seconds"
    mask_neg = df[col] < 0
    if mask_neg.sum() > 0:
        median_val = df.loc[~mask_neg, col].median()
        df.loc[mask_neg, col] = median_val
    return df

def preprocess_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_leakage_features(df)
    df = remove_id_columns(df, keep_driver_id=True)
    df = fix_negative_waiting_time(df)
    return df