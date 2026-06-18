"""Công cụ TRÌNH DIỄN vòng feedback: ghi một batch "current" mới.

  python -m demo.inject_drift            # batch BÌNH THƯỜNG (PSI ~ 0, Grafana xanh)
  python -m demo.inject_drift --drift    # batch LỆCH phân phối (PSI cao, drift đỏ)

Dagster `drift_sensor` (chạy mỗi ~30s) sẽ tự đọc batch này, push PSI lên
Pushgateway (Grafana cập nhật) và nếu drift -> tự trigger retrain pipeline.
"""
import argparse

from loguru import logger

from demo import step1_generate_data as s1
from demo.config import REFERENCE_PARQUET, CURRENT_PARQUET, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", action="store_true", help="sinh batch lệch phân phối")
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args()

    # Đảm bảo có reference (baseline) để so sánh
    if not REFERENCE_PARQUET.exists():
        s1.generate(args.n, SEED).to_parquet(REFERENCE_PARQUET, index=False)
        logger.info(f"[INJECT] Tạo reference baseline -> {REFERENCE_PARQUET}")

    # seed khác nhau mỗi lần -> chữ ký file thay đổi -> sensor coi là batch mới
    import time
    seed = SEED + int(time.time()) % 10_000
    df = s1.generate(args.n, seed, drift=args.drift)
    df.to_parquet(CURRENT_PARQUET, index=False)
    kind = "LỆCH (drift)" if args.drift else "bình thường"
    logger.info(f"[INJECT] Ghi current batch {kind} ({len(df):,} dòng) -> {CURRENT_PARQUET}")


if __name__ == "__main__":
    main()
