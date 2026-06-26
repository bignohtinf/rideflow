FROM python:3.11-slim AS builder

WORKDIR /rideflow

# [FIX] Two bugs on these lines:
#   1. COPY requirements.txt was missing destination → Docker build error
#   2. "requirementes.txt" was a typo (extra 'e') → file not found
COPY deployment/requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY data/feature/      data/feature/
COPY deployment/serve.py deployment/serve.py

# /app là WORKDIR trong container — PYTHONPATH cần trỏ vào đây để
# import như "data.feature.transformations" resolve đúng.
ENV PYTHONPATH=/app
ENV AWS_DEFAULT_REGION=us-east-1
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

# [IMPROVED] Added --workers for production readiness
CMD ["uvicorn", "deployment.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
