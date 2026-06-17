from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from loguru import logger
from catboost import CatBoostClassifier

SEED = 42


def calibrate_model(model, X_val: pd.DataFrame, y_val: pd.Series, method: str = "isotonic"):
    raw_probs = model.predict_proba(X_val)[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_val)
    elif method == "platt":
        calibrator = LogisticRegression()
        calibrator.fit(raw_probs.reshape(-1, 1), y_val)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    calibrated = (
        calibrator.predict(raw_probs)
        if method == "isotonic"
        else calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    )
    return calibrator, calibrated

_DEFAULTS = {
    "lgbm": {
        "n_estimators": 500,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "catboost": {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
    },
    "random_forest": {
        "n_estimators": 672,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "min_samples_split": 50,
        "max_features": 0.3,
    },
}


def build_model(model_name: str, model_config: dict | None = None):
    if model_config is None:
        model_config = {}

    canonical_name = "lgbm" if model_name == "lightgbm" else model_name

    defaults = _DEFAULTS.get(canonical_name, {})
    params = {**defaults, **model_config.get(canonical_name, {})}

    yaml_only_keys = {"objective", "metric", "boosting_type", "early_stopping_rounds"}
    clean_params = {k: v for k, v in params.items() if k not in yaml_only_keys}

    logger.info(f"Building {canonical_name} with params: {clean_params}")

    builders = {
        "lgbm": _build_lgbm,
        "xgboost": _build_xgboost,
        "catboost": _build_catboost,
        "random_forest": _build_random_forest,
    }

    if canonical_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(builders)}")

    return builders[canonical_name](clean_params)


def _build_lgbm(params: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **params,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )


def _build_catboost(params: dict) -> CatBoostClassifier:
    return CatBoostClassifier(
        **params,
        random_seed=SEED,
        eval_metric="AUC",
        verbose=100,
        thread_count=-1,
    )


def _build_xgboost(params: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


def _build_random_forest(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        **params,
        random_state=SEED,
        n_jobs=-1,
    )

def train_lgbm(n_estimators: int = 500) -> lgb.LGBMClassifier:
    return build_model("lgbm", {"lgbm": {"n_estimators": n_estimators}})


def train_catboost(n_estimators: int = 1000) -> CatBoostClassifier:
    return build_model("catboost", {"catboost": {"iterations": n_estimators}})


def train_xgboost(n_estimators: int = 500) -> xgb.XGBClassifier:
    return build_model("xgboost", {"xgboost": {"n_estimators": n_estimators}})


def train_random_forest(n_estimators: int = 672) -> RandomForestClassifier:
    return build_model("random_forest", {"random_forest": {"n_estimators": n_estimators}})
