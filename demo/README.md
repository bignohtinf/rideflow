# RideFlow — Bộ thị phạm MLOps (chạy local)

> Bản demo **chạy được end-to-end trên một máy**, bám theo
> `architecture/architecture.png`. Mục tiêu: để thực tập sinh **nhìn thấy** từng
> lớp kiến trúc hoạt động thật, và đối chiếu với các điểm cần sửa của bản gốc.

## 1. Bản demo này khác bản production ở đâu?

Bản production dùng AWS (S3, Redshift, Redis) nên không chạy được trên laptop.
Demo thay bằng tương đương local — **giữ nguyên kiến trúc & công cụ**:

| Lớp kiến trúc | Production | Demo (local) |
|---|---|---|
| Data Platform | Spark + S3 + Great Expectations | sinh parquet local (`step1`) |
| Feature offline | Redshift | parquet + Feast `FileSource` |
| Feature online | Redis / ElastiCache | Feast `online_store: sqlite` |
| ML Platform | MLflow server + S3 artifacts | MLflow file store (`demo/_mlflow`) |
| Serving | FastAPI + Seldon | FastAPI (`demo/serve.py`) |
| Monitoring | Evidently + Grafana | PSI tự tính (`step6`) |
| Orchestration | Dagster | `run_all.py` (tuần tự) |

## 2. Chạy toàn bộ trong 1 lệnh

**Cách A — venv (nhanh nhất để giảng dạy):**
```bash
# từ thư mục gốc repo
python3 -m venv .venv-demo
source .venv-demo/bin/activate          # Windows: .venv-demo\Scripts\activate
pip install -r demo/requirements-demo.txt

python -m demo.run_all
```

**Cách B — Docker full stack (giống production hơn):**
```bash
docker compose -f demo/docker-compose.demo.yml up --build
```

### Portal sau khi chạy (mỗi lớp kiến trúc một cửa)

| Lớp | Stack | Portal | URL |
|---|---|---|---|
| Orchestration | Dagster | DAG, launch run, schedule | http://localhost:3000 |
| ML Platform | MLflow | experiment + model registry | http://localhost:5000 |
| Serving | FastAPI | Swagger thử `/predict` | http://localhost:8010/docs |
| Monitoring | Prometheus | metric thô | http://localhost:9091 |
| Monitoring | Pushgateway | nhận metric drift (PSI) | http://localhost:9092 |
| Monitoring | Grafana | dashboard serving + drift (admin/admin) | http://localhost:3001 |

> Kịch bản trình diễn: **Dagster** chạy pipeline → **MLflow** thấy model lên
> Production → **Swagger** bắn `/predict` → **Grafana** thấy biểu đồ real-time.

### Demo vòng feedback (drift → tự retrain)
`drift_sensor` chạy sẵn trong Dagster. Tiêm batch lệch để kích hoạt:
```bash
docker compose -f demo/docker-compose.demo.yml exec dagster python -m demo.inject_drift --drift
```
Trong ~30s: Grafana panel drift chuyển **đỏ** (PSI ↑), Dagster tự chạy
`rideflow_demo_pipeline` để retrain, MLflow ra **version mới → Production**.
Tiêm batch bình thường để reset: bỏ cờ `--drift`.

Bạn sẽ thấy log theo đúng 4 lớp kiến trúc, kết thúc bằng:
- model được promote **Production** trong MLflow (`demo/_mlflow`)
- 5 dự đoán real-time qua FastAPI
- báo cáo drift (PSI) giữa baseline và batch lệch phân phối

## 3. Chạy từng bước (để giảng dạy)

```bash
python -m demo.step1_generate_data       # DATA: sinh raw rides
python -m demo.step2_build_features       # FEATURE: feature engineering -> offline
python -m demo.step3_train                # ML: train + MLflow + promote Production
python -m demo.step4_materialize          # FEATURE: nạp lên online store (Feast)
uvicorn demo.serve:app --port 8000        # SERVING: bật API (terminal khác)
python -m demo.step5_predict              # gọi thử /predict
python -m demo.step6_drift_check          # MONITORING: PSI drift
```

Xem experiment & model registry:

```bash
mlflow ui --backend-store-uri demo/_mlflow --port 5000   # mở http://localhost:5000
```

## 4. 4 điểm bản demo sửa đúng so với bản gốc (đối chiếu với review)

| # | Bản gốc (lỗi) | Bản demo (đúng) |
|---|---|---|
| 1 | Serving liệt kê feature tay, thiếu on-demand → **skew** | Đọc **feature contract** (`step3` lưu) → parity tuyệt đối |
| 2 | Chỉ register tới **Staging**, serve load `Production` → đứt | `step3` **promote Production** |
| 3 | `runs:{id}/model` thiếu `/` → register fail | `runs:/{id}/model` |
| 4 | `calibrate_model()` là **dead code** | `CalibratedClassifierCV` áp dụng thật |

## Tài liệu & trình chiếu

| File | Mô tả |
|---|---|
| [docs/RideFlow_MLOps_Guide.pdf](../docs/RideFlow_MLOps_Guide.pdf) | Tài liệu thị phạm đầy đủ (A4) |
| [docs/RideFlow_Demo_Presentation.pdf](../docs/RideFlow_Demo_Presentation.pdf) | Slide trình chiếu (9 slide, kèm **output thật**) |
| [docs/build_pdf.py](../docs/build_pdf.py) · [docs/build_presentation.py](../docs/build_presentation.py) | Script build lại PDF từ Markdown |

Build lại PDF (sau khi sửa Markdown):
```bash
python docs/build_pdf.py && python docs/build_presentation.py
```

## 5. Cấu trúc

```
demo/
├── config.py                 # map production <-> local
├── step1_generate_data.py    # DATA PLATFORM
├── step2_build_features.py    # FEATURE PLATFORM (offline) — tái dùng code repo
├── step3_train.py             # ML PLATFORM (4 fix ở trên)
├── step4_materialize.py       # FEATURE PLATFORM (online, Feast)
├── serve.py                   # SERVING (FastAPI, feature parity)
├── step5_predict.py           # client gọi API
├── step6_drift_check.py       # MONITORING (PSI)
├── run_all.py                 # orchestrator tuần tự (chạy nhanh, không cần Docker)
├── feature_repo/              # Feast local (sqlite)
├── dagster_demo/              # ORCHESTRATION: Dagster job + schedule (portal :3300)
├── monitoring/                # Prometheus config + Grafana datasource/dashboard
├── Dockerfile                 # image full stack
├── docker-compose.demo.yml    # 6 service + portal cho từng lớp
└── requirements-demo.txt      # đã pin version
```

Xem thêm: [INSTALL.md](INSTALL.md) · [DEPLOYMENT.md](DEPLOYMENT.md)
