"""
Push metrics từ MLflow + S3 lên Prometheus Pushgateway.
Chạy một lần hoặc dùng --daemon để push định kỳ.

Cách dùng:
  python monitoring/push_metrics.py
  python monitoring/push_metrics.py --daemon --interval 60
"""
import argparse
import io
import os
import time
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
_pgw = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
PUSHGATEWAY_URL = _pgw if _pgw.startswith("http") else f"http://{_pgw}"
MODEL_NAME      = "ride_completion"
BUCKET          = "rideflow"
TARGET          = "is_completed"

S3_OPTS = dict(
    key    = os.getenv("AWS_ACCESS_KEY_ID"),
    secret = os.getenv("AWS_SECRET_ACCESS_KEY"),
    client_kwargs={"region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1")},
)

mlflow.set_tracking_uri(MLFLOW_URI)


# ── Helpers ───────────────────────────────────────────────────────────────────

def push(job: str, metrics: dict[str, float]):
    """Push key=value pairs to Pushgateway using the text format."""
    lines = []
    for name, value in metrics.items():
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")
    body = "\n".join(lines) + "\n"
    url  = f"{PUSHGATEWAY_URL}/metrics/job/{job}"
    try:
        r = requests.post(url, data=body, headers={"Content-Type": "text/plain"}, timeout=5)
        r.raise_for_status()
        print(f"  [push] {job}: {metrics}")
    except Exception as e:
        print(f"  [push] ERROR pushing {job}: {e}")


def compute_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    rp = np.histogram(ref, bins=edges)[0].astype(float) + 1e-6
    cp = np.histogram(cur, bins=edges)[0].astype(float) + 1e-6
    rp /= rp.sum(); cp /= cp.sum()
    return float(np.sum((rp - cp) * np.log(rp / cp)))


# ── Metric collectors ─────────────────────────────────────────────────────────

def push_model_metrics():
    """Đọc metrics của best run từ MLflow và push lên Pushgateway."""
    try:
        client = MlflowClient(MLFLOW_URI)
        exp = client.get_experiment_by_name(MODEL_NAME)
        if not exp:
            print("  [mlflow] experiment not found")
            return
        runs = client.search_runs(exp.experiment_id, order_by=["metrics.auc_roc DESC"], max_results=1)
        if not runs:
            print("  [mlflow] no runs found")
            return
        best = runs[0]
        m = best.data.metrics
        metrics = {}
        if "auc_roc"  in m: metrics["model_auc_roc"]  = m["auc_roc"]
        if "log_loss" in m: metrics["model_log_loss"]  = m["log_loss"]
        if "pr_auc"   in m: metrics["model_pr_auc"]    = m["pr_auc"]
        if "f1"       in m: metrics["model_f1"]        = m["f1"]
        if "ece"      in m: metrics["model_ece"]       = m["ece"]
        if metrics:
            push("mlflow_metrics", metrics)
    except Exception as e:
        print(f"  [mlflow] ERROR: {e}")


def _read_s3_parquet(key: str) -> pd.DataFrame:
    """Read parquet from S3 using boto3 (no s3fs dependency)."""
    import boto3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=S3_OPTS["key"],
        aws_secret_access_key=S3_OPTS["secret"],
        region_name=S3_OPTS["client_kwargs"]["region_name"],
    )
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _list_s3_prefixes(prefix: str) -> list[str]:
    """List immediate sub-prefixes under a prefix using boto3."""
    import boto3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=S3_OPTS["key"],
        aws_secret_access_key=S3_OPTS["secret"],
        region_name=S3_OPTS["client_kwargs"]["region_name"],
    )
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))


def push_drift_metrics():
    """Tính PSI giữa reference và current features, push lên Pushgateway."""
    try:
        ref_df = _read_s3_parquet("features/reference/features.parquet")
        cur_df = _read_s3_parquet("features/initial/features.parquet")

        numeric_cols = [c for c in ref_df.columns
                        if c != TARGET and pd.api.types.is_numeric_dtype(ref_df[c])][:12]

        psi_values = [compute_psi(ref_df[c], cur_df[c]) for c in numeric_cols]
        drift_share    = sum(p > 0.2 for p in psi_values) / len(psi_values) if psi_values else 0.0
        drift_detected = 1.0 if drift_share > 0.2 else 0.0
        max_psi        = max(psi_values) if psi_values else 0.0

        push("drift_metrics", {
            "drift_share":    round(drift_share, 4),
            "drift_detected": drift_detected,
            "drift_psi_max":  round(max_psi, 4),
        })

    except Exception as e:
        print(f"  [drift] ERROR: {e}")


def push_prediction_metrics():
    """Đọc predictions mới nhất từ S3 và push completion rate lên Pushgateway."""
    try:
        pred_dirs = _list_s3_prefixes("predictions/")
        if not pred_dirs:
            print("  [predictions] No prediction folders found")
            return

        latest_key = f"{pred_dirs[-1]}predictions.parquet"
        pred_df = _read_s3_parquet(latest_key)

        completion_rate = float(pred_df["predicted_label"].mean())
        total_preds     = len(pred_df)
        mean_prob       = float(pred_df["completion_prob"].mean())

        push("data_metrics", {
            "prediction_completion_rate": round(completion_rate, 4),
            "prediction_volume":         float(total_preds),
            "prediction_mean_prob":       round(mean_prob, 4),
        })
        print(f"  [predictions] {latest_key}: {total_preds:,} preds, "
              f"completion_rate={completion_rate:.4f}, mean_prob={mean_prob:.4f}")

    except Exception as e:
        print(f"  [predictions] ERROR: {e}")


def backfill_prediction_metrics():
    """Push metrics cho TẤT CẢ predictions đã có trên S3 (lịch sử).
    Mỗi ngày push 1 lần với instance label riêng để Prometheus lưu riêng biệt."""
    try:
        pred_dirs = _list_s3_prefixes("predictions/")
        if not pred_dirs:
            print("  [backfill] No prediction folders found")
            return

        for pred_dir in pred_dirs:
            # pred_dir = "predictions/2026-06-18/"
            date_str = pred_dir.strip("/").split("/")[-1]
            try:
                pred_df = _read_s3_parquet(f"{pred_dir}predictions.parquet")
            except Exception as e:
                print(f"  [backfill] Skip {date_str}: {e}")
                continue

            completion_rate = float(pred_df["predicted_label"].mean())
            total_preds     = len(pred_df)
            mean_prob       = float(pred_df["completion_prob"].mean())

            # Push with instance=date so each date has its own metric series
            metrics_body = (
                f"# TYPE prediction_completion_rate gauge\n"
                f"prediction_completion_rate {completion_rate:.4f}\n"
                f"# TYPE prediction_volume gauge\n"
                f"prediction_volume {total_preds}\n"
                f"# TYPE prediction_mean_prob gauge\n"
                f"prediction_mean_prob {mean_prob:.4f}\n"
            )
            url = f"{PUSHGATEWAY_URL}/metrics/job/data_metrics/instance/{date_str}"
            try:
                r = requests.post(url, data=metrics_body,
                                  headers={"Content-Type": "text/plain"}, timeout=5)
                r.raise_for_status()
                print(f"  [backfill] {date_str}: {total_preds:,} preds, "
                      f"rate={completion_rate:.4f}, prob={mean_prob:.4f}")
            except Exception as e:
                print(f"  [backfill] ERROR pushing {date_str}: {e}")

    except Exception as e:
        print(f"  [backfill] ERROR: {e}")


def push_serve_metrics():
    """Kiểm tra serve API health và push status."""
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        h = r.json()
        push("serve_metrics", {
            "serve_api_up":      1.0 if h.get("status") == "ok" else 0.0,
            "model_loaded":      1.0 if h.get("model_loaded") else 0.0,
            "feature_store_up":  1.0 if h.get("store_loaded") else 0.0,
        })
    except Exception as e:
        push("serve_metrics", {"serve_api_up": 0.0, "model_loaded": 0.0, "feature_store_up": 0.0})
        print(f"  [serve] API offline: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_once():
    print(f"[{time.strftime('%H:%M:%S')}] Pushing metrics to {PUSHGATEWAY_URL} ...")
    push_model_metrics()
    push_drift_metrics()
    push_prediction_metrics()
    push_serve_metrics()
    print("Done.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon",   action="store_true", help="Chạy liên tục")
    parser.add_argument("--interval", type=int, default=60, help="Giây giữa mỗi lần push (mặc định: 60)")
    parser.add_argument("--backfill", action="store_true", help="Push metrics cho tất cả predictions lịch sử")
    args = parser.parse_args()

    if args.backfill:
        print("Backfilling prediction metrics from S3 ...")
        backfill_prediction_metrics()
        print("Backfill done.\n")

    if args.daemon:
        print(f"Daemon mode — push mỗi {args.interval}s. Ctrl+C để dừng.")
        while True:
            run_once()
            time.sleep(args.interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
