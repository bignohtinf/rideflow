"""STEP 5b — gọi thử Serving API (real-time inference).

Lấy vài order_id từ dữ liệu raw rồi gọi POST /predict.
Chạy độc lập (cần serve đang chạy):  python -m demo.step5_predict
"""
import argparse

import httpx
import pandas as pd
from loguru import logger

from demo.config import RAW_PARQUET, SERVE_URL


def sample_order_ids(n: int = 5) -> list[str]:
    df = pd.read_parquet(RAW_PARQUET, columns=["order_id"])
    return df["order_id"].head(n).tolist()


def call(order_ids: list[str], base_url: str = SERVE_URL) -> None:
    for oid in order_ids:
        r = httpx.post(f"{base_url}/predict", json={"order_id": oid}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            logger.info(
                f"[PREDICT] {d['order_id']} -> prob={d['completion_prob']:.4f} "
                f"label={d['predicted_label']} ({d['latency_ms']}ms)"
            )
        else:
            logger.error(f"[PREDICT] {oid} -> HTTP {r.status_code}: {r.text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--url", default=SERVE_URL)
    args = ap.parse_args()
    call(sample_order_ids(args.n), args.url)


if __name__ == "__main__":
    main()
