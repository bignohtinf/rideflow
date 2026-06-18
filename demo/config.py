"""Cấu hình tập trung cho bản DEMO chạy local.

Bản production dùng AWS (S3, Redshift, Redis). Bản demo này thay thế bằng
local filesystem + SQLite để intern chạy được trên một máy, không cần cloud.
Map tương ứng:

    Production (AWS)         ->   Demo (local)
    ----------------------------------------------------
    S3 (raw/features)        ->   demo/_data/*.parquet
    Redshift (offline store) ->   parquet FileSource (Feast)
    Redis (online store)     ->   SQLite (Feast online_store)
    MLflow trên server       ->   MLflow file store local (demo/_mlflow)
"""
import os
from pathlib import Path

# Thư mục gốc của demo
DEMO_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_DIR / "_data"
MLFLOW_DIR = DEMO_DIR / "_mlflow"
FEATURE_REPO = DEMO_DIR / "feature_repo"
ARTIFACTS_DIR = DEMO_DIR / "_artifacts"

# Tạo sẵn các thư mục output
for _d in (DATA_DIR, MLFLOW_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# File dữ liệu (mô phỏng các "bucket"/"table" trên AWS)
RAW_PARQUET = DATA_DIR / "rides_raw.parquet"            # ~ s3://rideflow/raw
FEATURES_PARQUET = DATA_DIR / "order_features.parquet"  # ~ Redshift offline store
REFERENCE_PARQUET = DATA_DIR / "reference.parquet"      # baseline cho drift
CURRENT_PARQUET = DATA_DIR / "current.parquet"          # batch mới cho drift

# MLflow
MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR}"
EXPERIMENT_NAME = "ride_completion_demo"
MODEL_NAME = "ride_completion"

# Hợp đồng feature (feature contract) — lưu thứ tự cột model được train.
# Đây là chìa khoá CHỐNG training/serving skew: serving đọc đúng list này.
FEATURE_NAMES_FILE = ARTIFACTS_DIR / "feature_names.json"

TARGET = "is_completed"
ENTITY_KEY = "order_id"
TIMESTAMP_COL = "event_timestamp"

SEED = 42

# Cổng serving demo. Mặc định 8000; override khi cổng bận:
#   RIDEFLOW_DEMO_PORT=8001 python -m demo.run_all
SERVE_PORT = int(os.getenv("RIDEFLOW_DEMO_PORT", "8000"))
SERVE_URL = f"http://localhost:{SERVE_PORT}"

# Monitoring: Pushgateway cho metric DẠNG BATCH (drift/PSI) — khác serve dùng
# pull /metrics. Trống = không push (chạy run_all local không cần).
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "")

# Ngưỡng PSI coi là drift đáng kể -> trigger retrain (giống production 0.2).
PSI_THRESHOLD = 0.2
