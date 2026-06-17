import os
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from loguru import logger

PUSHGATEWAY = os.getenv("PUSHGATEWAY_URL", "pushgateway:9091")

def _safe_push(registry: CollectorRegistry, job: str, target_date: str):
    try:
        push_to_gateway(
            PUSHGATEWAY, job=job,
            grouping_key={"date": target_date}, registry=registry,
        )
    except Exception as e:
        logger.error(f"Failed to push metrics to {PUSHGATEWAY}: {e}")


def push_model_metrics(metrics: dict, target_date: str):
    registry = CollectorRegistry()

    auc = Gauge("model_auc_roc", "Model AUC-ROC", registry=registry)
    ll = Gauge("model_log_loss", "Model Log Loss", registry=registry)
    f1 = Gauge("model_f1", "Model F1 Score", registry=registry)

    auc.set(metrics.get("actual_auc_roc", 0))
    ll.set(metrics.get("actual_log_loss", 0))
    f1.set(metrics.get("actual_f1", 0))

    _safe_push(registry, "model_performance", target_date)


def push_drift_metrics(metrics: dict, target_date: str):
    registry = CollectorRegistry()

    drift_share = Gauge("drift_share", "Share of drifted columns", registry=registry)
    n_drifted = Gauge("drift_n_cols", "Number of drifted columns", registry=registry)
    detected = Gauge("drift_detected", "Drift detected (0/1)", registry=registry)

    drift_share.set(metrics.get("drift_share", 0))
    n_drifted.set(metrics.get("n_drifted", 0))
    detected.set(int(metrics.get("drift_detected", False)))

    _safe_push(registry, "data_drift", target_date)


def push_prediction_metrics(
    target_date: str, n_preds: int, mean_prob: float, completion_rate: float,
):
    registry = CollectorRegistry()

    vol = Gauge("prediction_volume", "Number of predictions", registry=registry)
    prob = Gauge("prediction_mean_prob", "Mean completion prob", registry=registry)
    cr = Gauge("prediction_completion_rate", "Actual completion rate", registry=registry)

    vol.set(n_preds)
    prob.set(mean_prob)
    cr.set(completion_rate)

    _safe_push(registry, "predictions", target_date)
