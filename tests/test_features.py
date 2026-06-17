"""Unit tests for feature engineering pipeline."""
import numpy as np
import pandas as pd
import pytest
from data.feature.transformations import (
    add_supply_demand_features,
    add_confidence_features,
    add_trip_value_features,
    add_binary_flags,
    add_interaction_features,
    add_time_features,
    add_date_features,
    add_driver_aggregation,
    build_features,
)
from data.feature.preprocessing import (
    remove_leakage_features,
    remove_id_columns,
    fix_negative_waiting_time,
    preprocess_for_training,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal dataframe with all required columns for feature engineering."""
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4"],
        "matching_batch_id": [1, 2, 3, 4],
        "driver_id": ["d1", "d1", "d2", "d2"],
        "num_drivers": [10, 1, 20, 5],
        "num_orders": [50, 3, 10, 30],
        "eta_avg": [300.0, 1000.0, 200.0, 500.0],
        "eta_std": [50.0, 200.0, 10.0, 80.0],
        "eta_min": [200.0, 800.0, 150.0, 400.0],
        "eda_avg": [100.0, 300.0, 80.0, 200.0],
        "eda_std": [20.0, 60.0, 5.0, 40.0],
        "eda_min": [60.0, 200.0, 50.0, 150.0],
        "distance": [5.0, 1.0, 15.0, 3.0],
        "total_fee": [50000.0, 15000.0, 120000.0, 30000.0],
        "hour_of_day": [8, 14, 18, 3],
        "minute_of_hour": [30, 0, 45, 15],
        "rush_hour": [1, 0, 1, 0],
        "user_waiting_time_seconds": [60.0, -5.0, 150.0, 30.0],
        "date": ["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-02"],
        "is_completed": [1, 0, 1, 0],
        # Leakage columns for testing removal
        "est_time_arrival": [100, 200, 300, 400],
        "total_pay": [50000, 15000, 120000, 30000],
    })


# ── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_remove_leakage_features(self, sample_df):
        result = remove_leakage_features(sample_df)
        assert "est_time_arrival" not in result.columns
        assert "total_pay" not in result.columns
        # Non-leakage columns should remain
        assert "distance" in result.columns

    def test_remove_id_columns(self, sample_df):
        result = remove_id_columns(sample_df)
        assert "order_id" not in result.columns
        assert "matching_batch_id" not in result.columns
        assert "driver_id" not in result.columns

    def test_remove_id_columns_keep_driver(self, sample_df):
        result = remove_id_columns(sample_df, keep_driver_id=True)
        assert "driver_id" in result.columns
        assert "order_id" not in result.columns

    def test_fix_negative_waiting_time(self, sample_df):
        result = fix_negative_waiting_time(sample_df)
        assert (result["user_waiting_time_seconds"] >= 0).all()
        # Non-negative values should stay the same
        assert result.loc[0, "user_waiting_time_seconds"] == 60.0

    def test_preprocess_for_training(self, sample_df):
        result = preprocess_for_training(sample_df)
        assert "est_time_arrival" not in result.columns
        assert "order_id" not in result.columns
        assert (result["user_waiting_time_seconds"] >= 0).all()
        # driver_id should be kept for aggregation
        assert "driver_id" in result.columns


# ── Transformation Tests ─────────────────────────────────────────────────────

class TestTransformations:
    def test_supply_demand_features(self, sample_df):
        result = add_supply_demand_features(sample_df)
        assert "supply_demand_ratio" in result.columns
        assert "demand_supply_ratio" in result.columns
        # Check formula: num_drivers / (num_orders + 1)
        expected = 10 / (50 + 1)
        assert abs(result.loc[0, "supply_demand_ratio"] - expected) < 1e-6

    def test_confidence_features(self, sample_df):
        result = add_confidence_features(sample_df)
        assert "eta_confidence" in result.columns
        assert "eda_confidence" in result.columns

    def test_binary_flags(self, sample_df):
        result = add_binary_flags(sample_df)
        # distance < 2 → is_short_trip
        assert result.loc[1, "is_short_trip"] == 1  # distance=1.0
        assert result.loc[0, "is_short_trip"] == 0  # distance=5.0
        # eta_avg > 900 → is_long_eta
        assert result.loc[1, "is_long_eta"] == 1    # eta=1000
        assert result.loc[0, "is_long_eta"] == 0    # eta=300

    def test_time_features(self, sample_df):
        result = add_time_features(sample_df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "minutes_since_midnight" in result.columns
        # hour=8, minute=30 → 8*60+30 = 510
        assert result.loc[0, "minutes_since_midnight"] == 510

    def test_date_features(self, sample_df):
        result = add_binary_flags(sample_df)  # need rush_hour
        result = add_date_features(result)
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "date" not in result.columns  # should be dropped

    def test_driver_aggregation_with_stats(self, sample_df):
        # Pre-compute driver stats (simulating training)
        driver_stats = sample_df.groupby("driver_id")["is_completed"].agg(["mean", "count"])
        driver_stats["smoothed_cr"] = driver_stats["mean"]

        result = add_driver_aggregation(sample_df.copy(), driver_stats=driver_stats)
        assert "driver_order_count" in result.columns
        assert "driver_completion_rate_smoothed" in result.columns
        assert "driver_id" not in result.columns

    def test_build_features_output_shape(self, sample_df):
        df = preprocess_for_training(sample_df)
        result = build_features(df, use_date=True, use_driver_agg=False)
        assert len(result) == len(df)
        assert "supply_demand_ratio" in result.columns
        assert "hour_sin" in result.columns

    def test_cyclical_encoding_range(self, sample_df):
        result = add_time_features(sample_df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()
