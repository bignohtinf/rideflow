import numpy as np
import pandas as pd
from loguru import logger


def add_supply_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    df["supply_demand_ratio"] = df["num_drivers"] / (df["num_orders"] + 1)
    df["demand_supply_ratio"] = df["num_orders"] / (df["num_drivers"] + 1)
    return df


def add_confidence_features(df: pd.DataFrame) -> pd.DataFrame:
    df["eta_confidence"] = df["eta_std"] / (df["eta_avg"] + 1)
    df["eda_confidence"] = df["eda_std"] / (df["eda_avg"] + 0.01)
    return df


def add_trip_value_features(df: pd.DataFrame) -> pd.DataFrame:
    df["fee_per_km"] = df["total_fee"] / (df["distance"] + 0.01)
    df["eta_per_km"] = df["eta_avg"] / (df["distance"] + 0.01)
    df["eta_eda_ratio"] = df["eta_avg"] / (df["eda_avg"] + 0.01)
    df["pickup_to_trip_ratio"] = df["eda_avg"] / (df["distance"] + 0.01)
    return df


def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_short_trip"] = (df["distance"] < 2).astype(int)
    df["is_long_eta"] = (df["eta_avg"] > 900).astype(int)
    df["is_high_wait"] = (df["user_waiting_time_seconds"] > 120).astype(int)
    df["is_negative_wait"] = (df["user_waiting_time_seconds"] < 0).astype(int)
    df["is_single_driver"] = (df["num_drivers"] == 1).astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["short_trip_rush"] = df["is_short_trip"] * df["rush_hour"]
    df["low_supply_flag"] = (df["supply_demand_ratio"] < 0.2).astype(int)
    df["low_supply_short_trip"] = df["low_supply_flag"] * df["is_short_trip"]
    df["high_eta_rush"] = df["is_long_eta"] * df["rush_hour"]
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["minutes_since_midnight"] = df["hour_of_day"] * 60 + df["minute_of_hour"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["rush_hour_weekday"] = df["rush_hour"] * (1 - df["is_weekend"])
    df = df.drop(columns=["date"])
    return df

def add_driver_aggregation(
    df: pd.DataFrame,
    driver_stats: pd.DataFrame | None = None,
    global_mean: float = 0.5,
    smoothing: int = 30,
    min_records: int = 5,
) -> pd.DataFrame:
    if "driver_id" not in df.columns:
        return df

    if driver_stats is None:
        if "is_completed" not in df.columns:
            logger.warning("No driver_stats provided and no target column — skipping driver aggregation")
            df = df.drop(columns=["driver_id"])
            return df

        logger.warning(
            "Computing driver_stats from current df. "
            "This is only safe during training — for inference, pass pre-computed stats."
        )
        global_mean = df["is_completed"].mean()
        driver_stats = df.groupby("driver_id")["is_completed"].agg(["mean", "count"])
        driver_stats["smoothed_cr"] = (
            driver_stats["count"] * driver_stats["mean"] + smoothing * global_mean
        ) / (driver_stats["count"] + smoothing)

    df["driver_order_count"] = df["driver_id"].map(driver_stats["count"]).fillna(0)
    df["driver_completion_rate_smoothed"] = (
        df["driver_id"].map(driver_stats["smoothed_cr"]).fillna(global_mean)
    )

    df = df.drop(columns=["driver_id"])

    n_with_enough = (driver_stats["count"] >= min_records).sum()
    logger.info(f"Driver features: {n_with_enough:,} drivers with >={min_records} records "
                f"({n_with_enough / len(driver_stats):.1%})")
    return df

def build_features(
    df: pd.DataFrame,
    use_date: bool = True,
    use_driver_agg: bool = True,
    driver_stats: pd.DataFrame | None = None,
    global_mean: float = 0.5,
) -> pd.DataFrame:
    df = df.copy()
    df = add_supply_demand_features(df)
    df = add_confidence_features(df)
    df = add_trip_value_features(df)
    df = add_binary_flags(df)
    df = add_interaction_features(df)
    df = add_time_features(df)

    if use_date:
        df = add_date_features(df)

    if use_driver_agg and "driver_id" in df.columns:
        df = add_driver_aggregation(df, driver_stats=driver_stats, global_mean=global_mean)
    elif "driver_id" in df.columns:
        df = df.drop(columns=["driver_id"])

    return df
