"""Unit tests for data ingestion and validation."""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch


class TestSplitData:
    """Tests for data/storage/split_data.py."""

    def test_temporal_split_ordering(self, tmp_path):
        """Ensure train set comes before production set chronologically."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=100, freq="D"),
            "is_completed": np.random.randint(0, 2, 100),
            "distance": np.random.uniform(1, 20, 100),
        })
        raw_path = tmp_path / "raw.parquet"
        df.to_parquet(raw_path, index=False)

        from data.storage.split_data import split_data

        with patch("data.storage.split_data.TRAIN_PATH", tmp_path / "train.parquet"), \
             patch("data.storage.split_data.PRODUCTION_PATH", tmp_path / "prod.parquet"):
            train_df, prod_df = split_data(raw_path)

        assert train_df["date"].max() <= prod_df["date"].min()

    def test_split_ratio(self, tmp_path):
        """Verify 80/20 split ratio."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=100, freq="D"),
            "is_completed": np.random.randint(0, 2, 100),
        })
        raw_path = tmp_path / "raw.parquet"
        df.to_parquet(raw_path, index=False)

        from data.storage.split_data import split_data

        with patch("data.storage.split_data.TRAIN_PATH", tmp_path / "train.parquet"), \
             patch("data.storage.split_data.PRODUCTION_PATH", tmp_path / "prod.parquet"):
            train_df, prod_df = split_data(raw_path, train_ratio=0.8)

        assert len(train_df) == 80
        assert len(prod_df) == 20

    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing input."""
        from data.storage.split_data import split_data

        with pytest.raises(FileNotFoundError):
            split_data(tmp_path / "nonexistent.parquet")


class TestGXSuiteConfig:
    """Tests for Great Expectations suite config parsing."""

    def test_valid_config_structure(self, tmp_path):
        """Verify JSON config is parsed correctly."""
        import json

        config = {
            "suite_name": "test_suite",
            "expectations": [
                {"type": "expect_column_to_exist", "column": "order_id"},
                {"type": "expect_column_values_to_not_be_null", "column": "distance"},
            ],
        }
        config_path = tmp_path / "expectations.json"
        config_path.write_text(json.dumps(config))

        loaded = json.loads(config_path.read_text())
        assert loaded["suite_name"] == "test_suite"
        assert len(loaded["expectations"]) == 2
