"""Đẩy metric drift (PSI) lên Pushgateway để Prometheus scrape -> Grafana vẽ.

Drift là phép tính DẠNG BATCH (so reference vs current), không gắn với request
nào, nên dùng Pushgateway (push model) thay vì /metrics (pull model) của serve.
Đây đúng pattern production trong monitoring/grafana/metrics_pusher.py.
"""
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from loguru import logger

from demo.config import PUSHGATEWAY_URL, PSI_THRESHOLD

JOB = "rideflow_drift"


def push_drift(result: dict, gateway: str | None = None) -> bool:
    """Đẩy PSI từng cột + tổng hợp lên Pushgateway. Trả True nếu đã push."""
    gateway = gateway or PUSHGATEWAY_URL
    if not gateway:
        logger.info("[DRIFT] PUSHGATEWAY_URL trống -> bỏ qua push")
        return False

    registry = CollectorRegistry()
    g_psi = Gauge("rideflow_drift_psi", "PSI theo cột", ["column"], registry=registry)
    g_max = Gauge("rideflow_drift_psi_max", "PSI lớn nhất", registry=registry)
    g_detected = Gauge("rideflow_drift_detected", "1 nếu phát hiện drift", registry=registry)
    g_ncols = Gauge("rideflow_drift_columns", "Số cột vượt ngưỡng", registry=registry)
    g_thr = Gauge("rideflow_drift_threshold", "Ngưỡng PSI", registry=registry)

    psi = result["psi"]
    for col, val in psi.items():
        g_psi.labels(column=col).set(val)
    g_max.set(max(psi.values()) if psi else 0.0)
    g_detected.set(1 if result["drift_detected"] else 0)
    g_ncols.set(len(result["drifted_columns"]))
    g_thr.set(PSI_THRESHOLD)

    push_to_gateway(gateway, job=JOB, registry=registry)
    logger.info(f"[DRIFT] Đã push {len(psi)} cột PSI -> {gateway}")
    return True
