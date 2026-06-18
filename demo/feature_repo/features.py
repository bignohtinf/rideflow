"""Định nghĩa Feast cho DEMO.

Khác bản production (data/feature/feature_views.py) ở 2 điểm quan trọng:

1. Dùng FileSource (parquet local) thay cho RedshiftSource.
2. GỘP toàn bộ feature vào MỘT FeatureView duy nhất ("order_features"),
   schema đọc TRỰC TIẾP từ parquet. Nhờ vậy tập feature offline (train) và
   online (serve) LUÔN trùng nhau -> chống training/serving skew tận gốc.
   (Bản production tách raw / derived / on-demand nên dễ lệch khi serve.)
"""
from datetime import timedelta
from pathlib import Path

import pyarrow.parquet as pq
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64

# features.py nằm trong feature_repo/, parquet ở demo/_data/
_DEMO_DIR = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = _DEMO_DIR / "_data" / "order_features.parquet"

ENTITY_KEY = "order_id"
TIMESTAMP_COL = "event_timestamp"
TARGET = "is_completed"

order = Entity(name="order_id", join_keys=["order_id"], description="ID chuyến/lệnh ghép")

source = FileSource(
    name="order_features_source",
    path=str(FEATURES_PARQUET),
    timestamp_field=TIMESTAMP_COL,
)


def _schema_from_parquet() -> list[Field]:
    """Đọc tên cột từ parquet -> Field(Float64). Single source of truth."""
    cols = pq.read_schema(FEATURES_PARQUET).names
    skip = {ENTITY_KEY, TIMESTAMP_COL, TARGET}
    return [Field(name=c, dtype=Float64) for c in cols if c not in skip]


order_features = FeatureView(
    name="order_features",
    entities=[order],
    ttl=timedelta(days=90),
    schema=_schema_from_parquet(),
    source=source,
    online=True,
    tags={"team": "crp", "demo": "true"},
)
