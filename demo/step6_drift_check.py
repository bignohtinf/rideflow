"""STEP 6 — MONITORING: phát hiện data drift bằng PSI.

Production: Evidently + PSI threshold 0.2, drift -> Dagster sensor trigger retrain.
Demo: tự tính PSI (Population Stability Index) để không phụ thuộc Evidently,
so sánh tập reference (lúc train) với batch hiện tại.

PSI < 0.1  : ổn định
0.1–0.2    : dịch chuyển nhẹ
> 0.2      : drift đáng kể -> nên retrain
"""
import argparse

import numpy as np
import pandas as pd
from loguru import logger

from demo.config import REFERENCE_PARQUET, CURRENT_PARQUET

PSI_THRESHOLD = 0.2


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=quantiles)[0] / len(expected)
    a = np.histogram(actual, bins=quantiles)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def detect(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    num_cols = reference.select_dtypes("number").columns
    num_cols = [c for c in num_cols if c in current.columns]
    scores = {c: psi(reference[c].values, current[c].values) for c in num_cols}
    drifted = {c: s for c, s in scores.items() if s > PSI_THRESHOLD}
    return {
        "psi": scores,
        "drifted_columns": sorted(drifted, key=drifted.get, reverse=True),
        "drift_detected": len(drifted) > 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(REFERENCE_PARQUET))
    ap.add_argument("--current", default=str(CURRENT_PARQUET))
    args = ap.parse_args()

    ref, cur = pd.read_parquet(args.reference), pd.read_parquet(args.current)
    result = detect(ref, cur)

    for col in result["drifted_columns"][:8]:
        logger.warning(f"[DRIFT] {col}: PSI={result['psi'][col]:.3f}")
    if result["drift_detected"]:
        logger.warning(
            f"[DRIFT] Phát hiện drift ở {len(result['drifted_columns'])} cột "
            f"-> Dagster sensor sẽ trigger retrain_pipeline"
        )
    else:
        logger.info("[DRIFT] Không có drift đáng kể.")
    return result


if __name__ == "__main__":
    main()
