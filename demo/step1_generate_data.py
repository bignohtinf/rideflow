"""STEP 1 — DATA PLATFORM (mô phỏng).

Production: Spark đọc raw từ S3, validate bằng Great Expectations, ghi parquet.
Demo: sinh dữ liệu chuyến xe tổng hợp đúng theo schema
    data/raw/schemas/rides_raw_schema.json
rồi ghi ra parquet local (đóng vai S3).

Dữ liệu được sinh sao cho is_completed có quan hệ THẬT với feature
(supply/demand, ETA, waiting time...) để model học được tín hiệu.
"""
import argparse
import numpy as np
import pandas as pd
from loguru import logger

from demo.config import RAW_PARQUET, SEED


def generate(n: int = 20_000, seed: int = SEED, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    num_drivers = rng.integers(1, 40, n)
    num_orders = rng.integers(0, 80, n)
    distance = np.round(rng.gamma(2.0, 2.5, n) + 0.3, 2)          # km
    eta_avg = np.round(rng.gamma(3.0, 120, n) + 60, 1)            # giây
    eda_avg = np.round(distance * rng.uniform(60, 140, n), 1)     # giây tới điểm đón
    hour = rng.integers(0, 24, n)
    minute = rng.integers(0, 60, n)
    rush = ((hour >= 7) & (hour <= 9) | (hour >= 17) & (hour <= 19)).astype(int)
    waiting = rng.normal(45, 40, n).round().astype(int)          # có thể âm -> bẩn

    if drift:
        # Mô phỏng phân phối thay đổi (giờ cao điểm nhiều hơn, chờ lâu hơn)
        eta_avg = eta_avg * 1.6
        waiting = waiting + 60
        num_orders = num_orders + 30

    df = pd.DataFrame({
        "order_id": [f"ORD-{seed}-{i:06d}" for i in range(n)],
        "matching_batch_id": [f"B-{i // 50:05d}" for i in range(n)],
        "driver_id": [f"D-{d:05d}" for d in rng.integers(0, 3000, n)],
        "num_drivers": num_drivers,
        "num_orders": num_orders,
        "eta_avg": eta_avg,
        "eta_std": np.round(eta_avg * rng.uniform(0.1, 0.4, n), 1),
        "eta_min": np.round(eta_avg * rng.uniform(0.3, 0.7, n), 1),
        "eda_avg": eda_avg,
        "eda_std": np.round(eda_avg * rng.uniform(0.1, 0.4, n), 1),
        "eda_min": np.round(eda_avg * rng.uniform(0.3, 0.7, n), 1),
        "distance": distance,
        "total_fee": np.round(distance * rng.uniform(8000, 15000, n), 0),
        "hour_of_day": hour,
        "minute_of_hour": minute,
        "rush_hour": rush,
        "user_waiting_time_seconds": waiting,
        # cột leakage (sẽ bị loại ở preprocessing) — để dạy về data leakage
        "total_pay": np.round(distance * rng.uniform(6000, 12000, n), 0),
        "est_time_arrival": eta_avg + rng.normal(0, 30, n),
        "date": "2026-06-17",
    })

    # ── Nhãn is_completed phụ thuộc THẬT vào feature ──────────────────
    supply_demand = num_drivers / (num_orders + 1)
    logit = (
        1.2
        + 0.8 * np.log1p(supply_demand)        # nhiều tài x' -> dễ hoàn thành
        - 0.0015 * eta_avg                      # ETA lớn -> dễ huỷ
        - 0.01 * np.maximum(waiting, 0)         # chờ lâu -> dễ huỷ
        - 0.05 * distance                       # đi xa -> hơi khó hơn
        - 0.3 * rush                            # cao điểm -> khó hơn
    )
    prob = 1 / (1 + np.exp(-logit))
    df["is_completed"] = (rng.uniform(0, 1, n) < prob).astype(int)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--out", type=str, default=str(RAW_PARQUET))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--drift", action="store_true", help="sinh dữ liệu lệch phân phối")
    args = ap.parse_args()

    df = generate(args.n, args.seed, args.drift)
    df.to_parquet(args.out, index=False)
    logger.info(
        f"[DATA] Ghi {len(df):,} dòng -> {args.out} "
        f"| tỉ lệ hoàn thành = {df['is_completed'].mean():.1%}"
    )


if __name__ == "__main__":
    main()
