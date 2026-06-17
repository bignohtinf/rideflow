"""Unit tests for streaming feature computation."""
import math
import pytest
from pipelines.streaming.flink_consumer import add_realtime_features


@pytest.fixture
def sample_record():
    """Minimal Kafka record matching rides-raw schema."""
    return {
        "order_id": "order_123",
        "num_drivers": 10,
        "num_orders": 50,
        "eta_avg": 300.0,
        "eta_std": 50.0,
        "eda_avg": 100.0,
        "eda_std": 20.0,
        "distance": 5.0,
        "total_fee": 50000.0,
        "hour_of_day": 8,
        "minute_of_hour": 30,
        "user_waiting_time_seconds": 60.0,
    }


class TestRealtimeFeatures:
    def test_supply_demand_ratio(self, sample_record):
        result = add_realtime_features(sample_record)
        expected = 10 / (50 + 1)
        assert abs(result["supply_demand_ratio"] - expected) < 1e-6

    def test_demand_supply_ratio(self, sample_record):
        result = add_realtime_features(sample_record)
        expected = 50 / (10 + 1)
        assert abs(result["demand_supply_ratio"] - expected) < 1e-6

    def test_binary_flags(self, sample_record):
        result = add_realtime_features(sample_record)
        assert result["is_short_trip"] == 0  # distance=5 >= 2
        assert result["is_long_eta"] == 0    # eta=300 <= 900
        assert result["is_high_wait"] == 0   # wait=60 <= 120

    def test_short_trip_flag(self, sample_record):
        sample_record["distance"] = 1.5
        result = add_realtime_features(sample_record)
        assert result["is_short_trip"] == 1

    def test_rush_hour_detection(self, sample_record):
        # 8:30 is rush hour
        result = add_realtime_features(sample_record)
        assert result["rush_hour"] == 1

        # 14:00 is not rush hour
        sample_record["hour_of_day"] = 14
        result = add_realtime_features(sample_record)
        assert result["rush_hour"] == 0

    def test_minutes_since_midnight(self, sample_record):
        result = add_realtime_features(sample_record)
        assert result["minutes_since_midnight"] == 8 * 60 + 30

    def test_cyclical_encoding(self, sample_record):
        result = add_realtime_features(sample_record)
        assert -1 <= result["hour_sin"] <= 1
        assert -1 <= result["hour_cos"] <= 1
        # sin^2 + cos^2 should equal 1
        assert abs(result["hour_sin"] ** 2 + result["hour_cos"] ** 2 - 1.0) < 1e-6

    def test_interaction_features(self, sample_record):
        sample_record["distance"] = 1.0   # short trip
        sample_record["hour_of_day"] = 8   # rush hour
        result = add_realtime_features(sample_record)
        assert result["short_trip_rush"] == 1  # short_trip * rush_hour

    def test_feature_parity_with_batch(self, sample_record):
        """Ensure streaming produces the same features as batch pipeline."""
        result = add_realtime_features(sample_record)

        # All critical features from batch pipeline should exist
        expected_features = [
            "supply_demand_ratio", "demand_supply_ratio",
            "eta_confidence", "eda_confidence",
            "fee_per_km", "eta_per_km", "eta_eda_ratio", "pickup_to_trip_ratio",
            "is_short_trip", "is_long_eta", "is_high_wait", "is_negative_wait",
            "is_single_driver", "rush_hour", "minutes_since_midnight",
            "hour_sin", "hour_cos",
            "short_trip_rush", "low_supply_flag", "low_supply_short_trip",
            "high_eta_rush",
        ]
        for feat in expected_features:
            assert feat in result, f"Missing feature: {feat}"

    def test_zero_division_safety(self):
        """Features should handle zero values without crashing."""
        record = {
            "num_drivers": 0, "num_orders": 0,
            "eta_avg": 0, "eta_std": 0, "eda_avg": 0, "eda_std": 0,
            "distance": 0, "total_fee": 0,
            "hour_of_day": 0, "minute_of_hour": 0,
            "user_waiting_time_seconds": 0,
        }
        result = add_realtime_features(record)
        # Should not raise any exceptions
        assert math.isfinite(result["supply_demand_ratio"])
