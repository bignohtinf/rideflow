"""
Trigger ingest_pipeline + feature_pipeline trên Dagster cho các ngày có data trong S3.

Cách dùng:
  python product_load/trigger_pipeline.py
  python product_load/trigger_pipeline.py --date 2026-06-19
  python product_load/trigger_pipeline.py --date 2026-06-19 --watch
"""

import argparse
import json
import time
from datetime import datetime

import boto3
import requests

DAGSTER_URL = "http://localhost:3001/graphql"
BUCKET      = "rideflow"
RAW_PREFIX  = "raw/rides/"


# ── Helpers ───────────────────────────────────────────────────────────────────

def gql(query: str, variables: dict | None = None) -> dict:
    r = requests.post(DAGSTER_URL, json={"query": query, "variables": variables or {}}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


def get_repo_info() -> tuple[str, str]:
    """Lấy repositoryLocationName và repositoryName từ workspace."""
    q = """{
      workspaceOrError {
        ... on Workspace {
          locationEntries {
            name
            locationOrLoadError {
              ... on RepositoryLocation {
                repositories { name }
              }
            }
          }
        }
      }
    }"""
    data = gql(q)
    entries = data["workspaceOrError"]["locationEntries"]
    for entry in entries:
        loc = entry["locationOrLoadError"]
        if "repositories" in loc and loc["repositories"]:
            return entry["name"], loc["repositories"][0]["name"]
    raise RuntimeError("Không tìm thấy repository trong Dagster workspace")


def list_s3_dates(date_filter: str | None = None) -> list[str]:
    """Liệt kê các ngày có data trong s3://rideflow/raw/rides/."""
    s3 = boto3.client("s3")
    res = s3.list_objects_v2(Bucket=BUCKET, Prefix=RAW_PREFIX, Delimiter="/")
    prefixes = res.get("CommonPrefixes", [])
    dates = []
    for p in prefixes:
        # "raw/rides/2026-06-19/" → "2026-06-19"
        date = p["Prefix"].replace(RAW_PREFIX, "").rstrip("/")
        if date_filter is None or date == date_filter:
            dates.append(date)
    return sorted(dates)


def launch_run(loc_name: str, repo_name: str, job_name: str, target_date: str) -> str:
    """Launch một Dagster job, trả về runId."""
    mutation = """
    mutation LaunchRun($params: ExecutionParams!) {
      launchRun(executionParams: $params) {
        __typename
        ... on LaunchRunSuccess { run { runId } }
        ... on PythonError { message }
        ... on InvalidSubsetError { message }
        ... on RunConflict { message }
      }
    }"""

    op_key = {
        "ingest_pipeline":  "run_spark_ingest",
        "feature_pipeline": "run_spark_feature_job",
    }.get(job_name, job_name)

    variables = {
        "params": {
            "selector": {
                "repositoryLocationName": loc_name,
                "repositoryName": repo_name,
                "jobName": job_name,
            },
            "runConfigData": {
                "ops": {
                    op_key: {"config": {"target_date": target_date}}
                }
            },
        }
    }
    data = gql(mutation, variables)
    result = data["launchRun"]
    if result["__typename"] == "LaunchRunSuccess":
        return result["run"]["runId"]
    raise RuntimeError(f"Launch failed: {result.get('message')}")


def wait_for_run(run_id: str, timeout_s: int = 600, poll_s: int = 5) -> str:
    """Poll run status cho đến khi hoàn thành hoặc timeout."""
    q = """query RunStatus($runId: ID!) {
      runOrError(runId: $runId) {
        ... on Run { status }
      }
    }"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = gql(q, {"runId": run_id})
        status = data["runOrError"].get("status", "UNKNOWN")
        if status in ("SUCCESS", "FAILURE", "CANCELED"):
            return status
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {run_id[:8]}… {status}", end="\r")
        time.sleep(poll_s)
    return "TIMEOUT"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Ngày cụ thể (YYYY-MM-DD). Mặc định: tất cả ngày trong S3.")
    parser.add_argument("--watch", action="store_true", help="Đợi mỗi run hoàn thành trước khi tiếp tục.")
    args = parser.parse_args()

    print("Kết nối Dagster...")
    try:
        loc_name, repo_name = get_repo_info()
        print(f"  Workspace: {loc_name} / {repo_name}")
    except Exception as e:
        print(f"Lỗi kết nối Dagster ({DAGSTER_URL}): {e}")
        print("Đảm bảo Dagster đang chạy: docker compose ps dagster-webserver")
        return

    print("\nLiệt kê S3 dates...")
    try:
        dates = list_s3_dates(args.date)
    except Exception as e:
        print(f"Lỗi đọc S3: {e}")
        return

    if not dates:
        print(f"Không tìm thấy ngày nào trong s3://{BUCKET}/{RAW_PREFIX}")
        print("Chạy trans_data.py trước để upload data lên S3.")
        return

    print(f"  Tìm thấy {len(dates)} ngày: {', '.join(dates)}")

    for date in dates:
        print(f"\n{'='*50}")
        print(f"Xử lý ngày: {date}")

        # Step 1: ingest_pipeline
        print(f"\n  [1/2] Launch ingest_pipeline...")
        try:
            run_id = launch_run(loc_name, repo_name, "ingest_pipeline", date)
            print(f"  ✅ Launched: {run_id[:8]}  →  http://localhost:3001/runs/{run_id}")
        except Exception as e:
            print(f"  ❌ ingest_pipeline thất bại: {e}")
            continue

        if args.watch:
            status = wait_for_run(run_id)
            print(f"\n  ingest_pipeline: {status}")
            if status != "SUCCESS":
                print(f"  ⛔ Bỏ qua feature_pipeline cho {date} vì ingest thất bại.")
                continue

        # Step 2: feature_pipeline
        print(f"\n  [2/2] Launch feature_pipeline...")
        try:
            run_id2 = launch_run(loc_name, repo_name, "feature_pipeline", date)
            print(f"  ✅ Launched: {run_id2[:8]}  →  http://localhost:3001/runs/{run_id2}")
        except Exception as e:
            print(f"  ❌ feature_pipeline thất bại: {e}")
            continue

        if args.watch:
            status2 = wait_for_run(run_id2)
            print(f"\n  feature_pipeline: {status2}")
            if status2 == "SUCCESS":
                print(f"  🎉 Hoàn thành! Features cho {date} đã vào Redis + S3.")

    print(f"\n{'='*50}")
    print("Xong. Kiểm tra tiến độ tại: http://localhost:3001")
    if not args.watch:
        print("Thêm --watch để đợi từng run hoàn thành trước khi chạy bước tiếp theo.")


if __name__ == "__main__":
    main()
