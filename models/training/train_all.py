"""Train tất cả model, so sánh AUC-ROC, promote model tốt nhất lên Production.

Chạy:
    python -m models.training.train_all <target_date>

Ví dụ:
    python -m models.training.train_all 2024-01-15
"""
import sys
import mlflow
from loguru import logger

from models.training.train import train
from models.training.register_model import register
from models.training.evaluate import compare_models

MODELS = ["lgbm", "xgboost", "catboost", "random_forest"]


def train_all(target_date: str):
    results = {}  # model_name -> {run_id, metrics}

    # 1. Train từng model
    for model_name in MODELS:
        logger.info(f"\n{'='*50}\nTraining: {model_name}\n{'='*50}")
        try:
            run_id, metrics = train(target_date, model_name)
            results[model_name] = {"run_id": run_id, "metrics": metrics}
            logger.info(f"{model_name} → AUC-ROC: {metrics['AUC-ROC']:.4f}  run_id: {run_id}")
        except Exception as e:
            logger.error(f"{model_name} failed: {e}")

    if not results:
        raise RuntimeError("Tất cả model đều fail, không có gì để promote.")

    # 2. So sánh
    metrics_by_model = {name: r["metrics"] for name, r in results.items()}
    comparison = compare_models(metrics_by_model)
    print("\n" + "="*50)
    print("KẾT QUẢ SO SÁNH (sắp xếp theo AUC-ROC):")
    print("="*50)
    print(comparison.to_string())

    # 3. Chọn model tốt nhất theo AUC-ROC
    best_name = comparison.index[0]
    best_run_id = results[best_name]["run_id"]
    best_auc = comparison.loc[best_name, "AUC-ROC"]
    logger.info(f"\nModel tốt nhất: {best_name} (AUC-ROC={best_auc:.4f})")

    # 4. Register lên Staging
    version = register(best_run_id)
    logger.info(f"Registered {best_name} v{version} → Staging")

    # 5. Promote lên Production
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name="ride_completion",
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(f"Promoted ride_completion v{version} ({best_name}) → Production ✓")

    return best_name, version, comparison


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m models.training.train_all <target_date>")
        print("       target_date format: YYYY-MM-DD")
        sys.exit(1)

    target_date = sys.argv[1]
    train_all(target_date)
