# Hướng dẫn triển khai — RideFlow

Tài liệu này có 2 phần: **(A) triển khai bản demo local** (để thị phạm) và
**(B) đường nâng cấp lên production** trên AWS theo `architecture/architecture.png`.

---

## A. Triển khai bản DEMO (local)

### A.1. Một máy, một lệnh
```bash
source .venv-demo/bin/activate
python -m demo.run_all
```
Pipeline chạy tuần tự 5 lớp. Artifacts sinh ra trong `demo/_data`,
`demo/_mlflow`, `demo/feature_repo/data`.

### A.2. Tách dịch vụ (giống production hơn)
Mở nhiều terminal:

```bash
# T1 — MLflow tracking UI
mlflow ui --backend-store-uri demo/_mlflow --port 5000

# T2 — pipeline data->feature->train->materialize
python -m demo.step1_generate_data
python -m demo.step2_build_features
python -m demo.step3_train
python -m demo.step4_materialize

# T3 — serving
uvicorn demo.serve:app --host 0.0.0.0 --port 8000

# T4 — gọi thử + giám sát
python -m demo.step5_predict
python -m demo.step6_drift_check
```

### A.3. Chạy full stack bằng Docker Compose (khuyến nghị)
Đã có sẵn [demo/Dockerfile](Dockerfile) và [demo/docker-compose.demo.yml](docker-compose.demo.yml).
Compose chạy 3 service: `pipeline` (data→feature→train→materialize, một lần) →
`serve` (FastAPI) + `mlflow` (UI).

```bash
docker compose -f demo/docker-compose.demo.yml up --build
```

| Lớp | Service | Portal | Mô tả |
|---|---|---|---|
| Orchestration | dagster | http://localhost:3000 | Dagster UI: DAG, launch run, schedule |
| ML Platform | mlflow | http://localhost:5000 | Tracking + Model Registry |
| Serving | serve | http://localhost:8010/docs | FastAPI Swagger (`/predict`, `/health`, `/metrics`) |
| Monitoring | prometheus | http://localhost:9091 | Metric store (scrape serve + pushgateway) |
| Monitoring | pushgateway | http://localhost:9092 | Nhận metric batch drift/PSI từ `drift_sensor` |
| Monitoring | grafana | http://localhost:3001 | Dashboard serving + drift (admin/admin) |

Ngoài ra service `pipeline` chạy một lần để **seed model** (data→feature→train→
materialize) rồi thoát; `serve` chờ nó xong mới khởi động.

> API map ra **8010**, Grafana **3001** (vì 8000/3000 hay bận). State chia sẻ giữa
> các container qua named volume (`mlflow_data`, `data_data`, `artifacts`,
> `feast_data`, `dagster_home`). Dọn sạch:
> `docker compose -f demo/docker-compose.demo.yml down -v`.

**Sinh dữ liệu cho Grafana:** bắn vài request để dashboard có số liệu —
```bash
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8010/predict -H "Content-Type: application/json" \
       -d "{\"order_id\":\"ORD-42-$(printf %06d $i)\"}" > /dev/null
done
```

**Demo vòng feedback (drift → tự retrain):** `drift_sensor` (Dagster) mỗi ~30s
so reference vs current, push PSI lên Pushgateway → Grafana; PSI > 0.2 thì tự
trigger `rideflow_demo_pipeline` để retrain.
```bash
# tiêm batch lệch phân phối -> trong ~30s Grafana drift chuyển đỏ + Dagster retrain
docker compose -f demo/docker-compose.demo.yml exec dagster python -m demo.inject_drift --drift
# reset về bình thường (PSI ~ 0)
docker compose -f demo/docker-compose.demo.yml exec dagster python -m demo.inject_drift
```
Kết quả đã kiểm chứng: `psi_max 0.006 → 4.108`, MLflow `v1 Archived → v2 Production`,
cursor đảm bảo chỉ retrain 1 lần / batch.

### A.4. Kiểm chứng
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"order_id":"ORD-42-000000"}'
```

---

## B. Đường nâng cấp lên PRODUCTION (AWS)

Mỗi thành phần local đổi sang dịch vụ thật, **không đổi kiến trúc**:

| Lớp | Local (demo) | Production | File cấu hình |
|---|---|---|---|
| Data | parquet | S3 + Spark + Great Expectations | `data/raw/ingest_s3.py`, `pipelines/spark/` |
| Feature offline | FileSource | Redshift | `data/feature/feature_store.yaml` |
| Feature online | sqlite | Redis/ElastiCache | `data/feature/feature_store.yaml` |
| Registry | file store | MLflow server + S3 | `infra/docker-compose.yml` |
| Serving | uvicorn | FastAPI container / Seldon | `deployment/Dockerfile`, `deployment/serving/` |
| Monitoring | PSI script | Evidently + Prometheus + Grafana | `monitoring/`, `infra/prometheus.yml` |
| Orchestration | `run_all.py` | Dagster (schedule + sensor) | `dagster_workspace/` |

### B.1. Hạ tầng nền (local server / EC2)
```bash
cd infra
cp ../.env.example ../.env       # điền AWS keys
docker-compose up -d             # Kafka, MLflow, Prometheus, Grafana, serve
```

### B.2. Checklist trước khi lên production (rút ra từ review)
- [ ] **Promote Production**: bổ sung bước promote sau Staging (bản gốc còn thiếu — xem `demo/step3_train.py`).
- [ ] **Feature parity**: serving đọc feature theo contract/FeatureService, không liệt kê tay (xem `demo/serve.py`).
- [ ] **Register URI**: sửa `runs:/{run_id}/model` trong `models/training/register_model.py`.
- [ ] **Metric key**: thống nhất `auc_roc` giữa `train.py` và `evaluate.is_better_than_production`.
- [ ] **Pin version** trong `requirements.txt`.
- [ ] Thêm **CI** chạy `pytest` (đã có test trong `tests/`).
- [ ] Cập nhật README cho khớp thực tế (terraform/, streaming Flink vs Spark).

### B.3. Lịch chạy (Dagster)
| Giờ (UTC) | Job | Mô tả |
|---|---|---|
| 01:00 | ingest | S3 -> Spark -> S3 |
| 02:00 | feature | tính feature -> Feast materialize |
| 03:00 | inference | batch predict -> S3 |
| on drift | retrain | sensor PSI > 0.2 -> retrain_pipeline |
