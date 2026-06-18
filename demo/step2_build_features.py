"""STEP 2 — FEATURE PLATFORM: feature engineering (offline).

Production: Spark tính feature -> ghi Redshift (offline store).
Demo: tái sử dụng CHÍNH code feature của repo
    (data/feature/preprocessing.py + transformations.py)
để không có 2 phiên bản logic feature khác nhau (chống skew + đúng DRY),
rồi ghi parquet local đóng vai offline store cho Feast.

Output có thêm 2 cột bắt buộc cho Feast:
    - order_id        : entity key
    - event_timestamp : thời điểm sự kiện
"""
import argparse
import pandas as pd
from loguru import logger

from data.feature.preprocessing import preprocess_for_training
from data.feature.transformations import build_features
from demo.config import RAW_PARQUET, FEATURES_PARQUET, ENTITY_KEY, TIMESTAMP_COL


def build(raw_path: str) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    logger.info(f"[FEATURE] Đọc {len(raw):,} dòng raw từ {raw_path}")

    # Giữ lại order_id (preprocessing sẽ drop ID) để gắn lại làm entity key
    order_ids = raw[ENTITY_KEY].copy()

    clean = preprocess_for_training(raw)         # loại leakage, fix waiting âm, giữ driver_id
    feats = build_features(clean, use_date=True, use_driver_agg=True)

    # Cast mọi cột feature về float64 để Feast/online-store có dtype đồng nhất
    # (target/key/timestamp giữ nguyên). Tránh lỗi int/float khi materialize.
    from demo.config import TARGET
    feature_cols = [c for c in feats.columns if c != TARGET]
    feats[feature_cols] = feats[feature_cols].astype("float64")

    feats.insert(0, ENTITY_KEY, order_ids.values)
    feats[TIMESTAMP_COL] = pd.Timestamp("2026-06-17 00:00:00")

    logger.info(f"[FEATURE] Sinh {feats.shape[1]} cột (gồm target + key + timestamp)")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_PARQUET))
    ap.add_argument("--out", default=str(FEATURES_PARQUET))
    args = ap.parse_args()

    feats = build(args.raw)
    feats.to_parquet(args.out, index=False)
    logger.info(f"[FEATURE] Ghi offline store -> {args.out}")


if __name__ == "__main__":
    main()
