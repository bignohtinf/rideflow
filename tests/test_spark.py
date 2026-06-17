"""Unit tests for model training and evaluation logic (no Spark dependency)."""
import numpy as np
import pandas as pd
import pytest
from models.training.evaluate import (
    expected_calibration_error,
    evaluate_oof,
    compare_models,
)
from models.training.models import build_model


class TestEvaluation:
    def test_ece_perfect_calibration(self):
        """ECE should be ~0 for perfectly calibrated predictions."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
        ece = expected_calibration_error(y_true, y_prob)
        assert ece < 0.15  # Should be low

    def test_ece_worst_calibration(self):
        """ECE should be high for badly calibrated predictions."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
        ece = expected_calibration_error(y_true, y_prob)
        assert ece > 0.5

    def test_evaluate_oof_returns_all_metrics(self):
        np.random.seed(42)
        y_true = pd.Series(np.random.randint(0, 2, 100))
        oof = np.random.uniform(0, 1, 100)
        metrics = evaluate_oof(y_true, oof)

        expected_keys = {"AUC-ROC", "PR-AUC", "LogLoss", "F1", "ECE"}
        assert set(metrics.keys()) == expected_keys
        # All metrics should be finite numbers
        for v in metrics.values():
            assert np.isfinite(v)

    def test_evaluate_oof_auc_range(self):
        y_true = pd.Series([0, 0, 0, 1, 1, 1])
        oof = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        metrics = evaluate_oof(y_true, oof)
        assert 0.0 <= metrics["AUC-ROC"] <= 1.0

    def test_compare_models_sorting(self):
        results = {
            "model_a": {"AUC-ROC": 0.85, "F1": 0.80},
            "model_b": {"AUC-ROC": 0.90, "F1": 0.75},
            "model_c": {"AUC-ROC": 0.80, "F1": 0.82},
        }
        df = compare_models(results)
        assert df.index[0] == "model_b"  # Highest AUC-ROC first


class TestModelBuilder:
    def test_build_lgbm_default(self):
        model = build_model("lgbm")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_build_lgbm_with_config(self):
        config = {"lgbm": {"n_estimators": 100, "max_depth": 5}}
        model = build_model("lgbm", config)
        assert model.n_estimators == 100
        assert model.max_depth == 5

    def test_build_xgboost(self):
        model = build_model("xgboost")
        assert hasattr(model, "fit")

    def test_build_lightgbm_alias(self):
        """'lightgbm' should work as alias for 'lgbm'."""
        model = build_model("lightgbm")
        assert hasattr(model, "fit")

    def test_build_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent_model")

    def test_yaml_keys_filtered(self):
        """YAML-only keys like 'objective' should not be passed to sklearn."""
        config = {"lgbm": {"objective": "binary", "metric": ["auc"], "n_estimators": 200}}
        model = build_model("lgbm", config)
        # Should not crash — yaml-only keys are filtered out
        assert model.n_estimators == 200
