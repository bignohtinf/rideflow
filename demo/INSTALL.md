# Cài đặt — RideFlow Demo

## Yêu cầu
- Python 3.10–3.12
- ~2 GB đĩa trống (mlflow, lightgbm, feast)
- Không cần Docker, không cần AWS

## Các bước

### 1. Tạo virtual environment
Bắt buộc dùng venv (nhiều bản Linux chặn cài system-wide — lỗi
`externally-managed-environment`).

```bash
cd rideflow
python3 -m venv .venv-demo
source .venv-demo/bin/activate        # Linux/macOS
# .venv-demo\Scripts\activate         # Windows PowerShell
```

### 2. Cài dependencies (đã pin version)
```bash
pip install --upgrade pip
pip install -r demo/requirements-demo.txt
```

### 3. Kiểm tra
```bash
python -c "import sklearn, lightgbm, mlflow, feast, fastapi; print('OK')"
```

### 4. Chạy thử nhanh
```bash
python -m demo.run_all
```

## Sự cố thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `externally-managed-environment` | Chưa kích hoạt venv → làm lại bước 1 |
| `ModuleNotFoundError: data` / `demo` | Chạy lệnh từ **thư mục gốc repo**, dùng `python -m demo.xxx` |
| `feast apply` lỗi | Xoá `demo/feature_repo/data/*.db` rồi chạy lại `step4` |
| Port 8000 bận | `uvicorn demo.serve:app --port 8001` rồi `step5 --url http://localhost:8001` |
| Reset toàn bộ | Xoá `demo/_data`, `demo/_mlflow`, `demo/_artifacts`, `demo/feature_repo/data` |

## Gỡ cài đặt
```bash
deactivate
rm -rf .venv-demo demo/_data demo/_mlflow demo/_artifacts demo/feature_repo/data
```
