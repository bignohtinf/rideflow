from datetime import datetime, timedelta
from feast import FeatureStore
import sys

target_date = sys.argv[1]

store = FeatureStore(repo_path="data/feature")

store.materialize(
    start_date=datetime.strptime(target_date, "%Y-%m-%d"),
    end_date=datetime.strptime(target_date, "%Y-%m-%d") + timedelta(hours=24),
)
