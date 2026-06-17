import pandas as pd
from datetime import datetime

BUCKET = "rideflow"

def log_predictions(order_ids: list, probs: list, target_date: str):
    df = pd.DataFrame({
        "order_id":        order_ids,
        "completion_prob": probs,
        "predicted_label": [1 if p > 0.5 else 0 for p in probs],
        "predicted_at":    datetime.now(),
        "date":            target_date,
    })

    path = f"s3://{BUCKET}/predictions/{target_date}/predictions.parquet"
    df.to_parquet(path, index=False)
    print(f"Logged {len(df):,} predictions → {path}")