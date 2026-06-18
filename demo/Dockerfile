# Dockerfile cho bản DEMO RideFlow (chạy toàn bộ stack MLOps local trong container).
# Build context = thư mục gốc repo:  docker build -f demo/Dockerfile -t rideflow-demo .
FROM python:3.11-slim

# Thư viện hệ thống tối thiểu cho lightgbm (libgomp) + curl cho healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cài deps trước (tận dụng layer cache)
COPY demo/requirements-demo.txt /app/demo/requirements-demo.txt
RUN pip install --upgrade pip && pip install -r demo/requirements-demo.txt

# Copy code cần cho demo (data/, models/, demo/)
COPY data /app/data
COPY models /app/models
COPY demo /app/demo

# Mặc định: chạy serving. Pipeline được gọi qua service riêng trong compose.
EXPOSE 8000
CMD ["uvicorn", "demo.serve:app", "--host", "0.0.0.0", "--port", "8000"]
