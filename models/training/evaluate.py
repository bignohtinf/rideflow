import numpy as np 
import pandas as pd 
import mlflow
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    f1_score,
    classification_report,
    confusion_matrix,
)

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue 
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return ece 

def evaluate_oof(y_true: pd.Series, oof_proba: np.ndarray, name: str = "") -> dict:
    preds = (oof_proba > 0.5).astype(int)
    metrics = {
        "AUC-ROC": roc_auc_score(y_true, oof_proba),
        "PR-AUC": average_precision_score(y_true, oof_proba),
        "LogLoss": log_loss(y_true, oof_proba),
        "F1": f1_score(y_true, preds),
        "ECE": expected_calibration_error(y_true, oof_proba)
    }
    return metrics 

def print_evaluation(y_true: pd.Series, oof_proba: np.ndarray, name: str = "Model"):
    metrics = evaluate_oof(y_true, oof_proba, name)
    preds = (oof_proba > 0.5).astype(int)

    print(f"\n{'_'*50}")
    print(f" {name}")
    print(f"{'_'*50}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print()
    print(classification_report(y_true, preds, target_names=["Not Completed", "Completed"]))
    return metrics

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    return df.sort_values("AUC-ROC", ascending=False)

def is_better_than_production(new_metrics: dict, threshold: float = 0.01) -> bool:
    client = mlflow.MlflowClient()

    try:
        prod_model = client.get_latest_versions("ride_completion", stages=["Production"])[0]
        prod_run = client.get_run(prod_model.run_id)
        prod_auc = prod_run.data.metrics["auc-roc"]
        return new_metrics.get("AUC-ROC") > prod_auc - threshold
    except IndexError:
        return True