"""RideFlow MLOps Dashboard — Real Data Connected"""
import os
import time
import random
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SERVE_URI   = "http://localhost:8000"
DAGSTER_URI = "http://localhost:3001"
BUCKET      = "rideflow"
MODEL_NAME  = "ride_completion"
TARGET      = "is_completed"

S3_OPTS = {
    "key":    os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {"region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1")},
}

mlflow.set_tracking_uri(MLFLOW_URI)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RideFlow MLOps",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #f8fafc; }
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: white; border-radius: 12px; padding: 20px 24px;
        border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-label  { font-size: 12px; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .metric-value  { font-size: 28px; font-weight: 700; color: #0f172a; line-height: 1; }
    .metric-delta  { font-size: 12px; margin-top: 4px; }
    .delta-up   { color: #10b981; }
    .delta-down { color: #ef4444; }
    .delta-neu  { color: #64748b; }
    .section-header { font-size: 15px; font-weight: 600; color: #0f172a; margin: 0 0 14px 0; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
    .badge-green  { background: #dcfce7; color: #16a34a; }
    .badge-yellow { background: #fef9c3; color: #ca8a04; }
    .badge-red    { background: #fee2e2; color: #dc2626; }
    .badge-blue   { background: #dbeafe; color: #2563eb; }
    .step-card { background: white; border-radius: 10px; padding: 14px 18px; border: 1px solid #e2e8f0; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_mlflow_runs() -> pd.DataFrame:
    try:
        client = MlflowClient(MLFLOW_URI)
        exp = client.get_experiment_by_name(MODEL_NAME)
        if not exp:
            return pd.DataFrame()
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
        rows = []
        for r in runs:
            rows.append({
                "run_id":      r.info.run_id[:8],
                "model":       r.data.params.get("model", "?"),
                "auc_roc":     r.data.metrics.get("auc_roc"),
                "pr_auc":      r.data.metrics.get("pr_auc"),
                "log_loss":    r.data.metrics.get("log_loss"),
                "f1":          r.data.metrics.get("f1"),
                "ece":         r.data.metrics.get("ece"),
                "n_samples":   r.data.params.get("n_samples"),
                "target_date": r.data.params.get("target_date"),
                "started_at":  pd.to_datetime(r.info.start_time, unit="ms"),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"MLflow unavailable: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_registry() -> pd.DataFrame:
    try:
        client = MlflowClient(MLFLOW_URI)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        return pd.DataFrame([{
            "Version": v.version,
            "Stage":   v.current_stage,
            "Run ID":  v.run_id[:8],
            "Created": pd.to_datetime(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M"),
        } for v in versions])
    except:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_features() -> pd.DataFrame | None:
    try:
        return pd.read_parquet(
            f"s3://{BUCKET}/features/initial/features.parquet",
            storage_options=S3_OPTS,
        )
    except:
        return None

@st.cache_data(ttl=300)
def get_reference() -> pd.DataFrame | None:
    try:
        return pd.read_parquet(
            f"s3://{BUCKET}/features/reference/features.parquet",
            storage_options=S3_OPTS,
        )
    except:
        return None

@st.cache_data(ttl=3600)
def get_daily_stats() -> pd.DataFrame:
    try:
        df = pd.read_parquet("data/processed/train.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return (
            df.groupby(df["date"].dt.date).agg(
                orders         =("order_id", "count"),
                completion_rate=("is_completed", "mean"),
                avg_distance   =("distance", "mean"),
                avg_fee        =("total_fee", "mean"),
            )
            .reset_index()
            .tail(30)
        )
    except:
        return pd.DataFrame()

def serve_health() -> dict:
    try:
        r = requests.get(f"{SERVE_URI}/health", timeout=3)
        return r.json()
    except:
        return {"status": "offline", "model_loaded": False, "store_loaded": False}

@st.cache_data(ttl=30)
def get_dagster_runs() -> list:
    q = """{
      runsOrError {
        __typename
        ... on Runs {
          results {
            runId pipelineName status
            startTime endTime
          }
        }
      }
    }"""
    try:
        r = requests.post(f"{DAGSTER_URI}/graphql", json={"query": q}, timeout=5)
        return r.json()["data"]["runsOrError"]["results"][:20]
    except:
        return []

def compute_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    rp = np.histogram(ref, bins=edges)[0].astype(float) + 1e-6
    cp = np.histogram(cur, bins=edges)[0].astype(float) + 1e-6
    rp /= rp.sum(); cp /= cp.sum()
    return float(np.sum((rp - cp) * np.log(rp / cp)))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 RideFlow MLOps")
    st.markdown("---")
    page = st.radio(
        "nav", ["📊 Overview", "🔁 Pipeline", "🧠 Model", "📡 Drift", "⚡ Real-time"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    health     = serve_health()
    runs_df    = get_mlflow_runs()
    mlflow_ok  = not runs_df.empty

    def dot(ok): return "🟢" if ok else "🔴"
    st.markdown("**Services**")
    st.caption(f"{dot(mlflow_ok)} MLflow  ·  {dot(health['status']=='ok')} Serve API")
    st.caption(f"{dot(health['model_loaded'])} Model loaded  ·  {dot(health['store_loaded'])} Feast")
    st.markdown("---")
    st.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# Overview")
    st.caption(f"RideFlow Production · {datetime.now().strftime('%A, %d %B %Y')}")
    st.markdown("")

    feat_df  = get_features()
    daily_df = get_daily_stats()
    best_auc = runs_df["auc_roc"].max() if not runs_df.empty else None
    prod_model = runs_df[runs_df["model"].notna()].iloc[0]["model"] if not runs_df.empty else "—"

    total_orders    = len(feat_df) if feat_df is not None else 0
    completion_rate = feat_df[TARGET].mean() if feat_df is not None else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Orders (train set)</div>
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-delta delta-neu">→ S3 features/initial</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Historical Completion Rate</div>
            <div class="metric-value">{completion_rate:.1%}</div>
            <div class="metric-delta delta-neu">→ real label distribution</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        auc_str = f"{best_auc:.4f}" if best_auc else "—"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Best AUC-ROC (MLflow)</div>
            <div class="metric-value">{auc_str}</div>
            <div class="metric-delta delta-neu">→ {prod_model}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        api_status = health["status"].upper()
        cls = "delta-up" if health["status"] == "ok" else "delta-down"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Serve API</div>
            <div class="metric-value" style="font-size:22px">{api_status}</div>
            <div class="metric-delta {cls}">{'🟢 Model + Store loaded' if health['model_loaded'] else '🔴 Model not loaded'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not daily_df.empty:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown('<p class="section-header">Orders & Completion Rate — Last 30 Days</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_bar(x=daily_df["date"], y=daily_df["orders"], name="Orders",
                        marker_color="#e2e8f0", yaxis="y1", opacity=0.8)
            fig.add_scatter(x=daily_df["date"], y=daily_df["completion_rate"],
                            name="Completion Rate", yaxis="y2",
                            line=dict(color="#3b82f6", width=2.5), mode="lines+markers",
                            marker=dict(size=4))
            fig.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Orders", showgrid=True, gridcolor="#f1f5f9"),
                yaxis2=dict(title="Completion Rate", overlaying="y", side="right",
                            tickformat=".0%", range=[0.5, 1.0]),
                legend=dict(orientation="h", y=1.1),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<p class="section-header">Distance Distribution</p>', unsafe_allow_html=True)
            if feat_df is not None:
                fig2 = go.Figure()
                fig2.add_histogram(
                    x=feat_df["distance"].clip(0, 30),
                    nbinsx=40, marker_color="#3b82f6", opacity=0.8,
                )
                fig2.update_layout(
                    height=280, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="white", plot_bgcolor="white",
                    xaxis=dict(title="km", showgrid=False),
                    yaxis=dict(gridcolor="#f1f5f9"), showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("S3 features unavailable")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔁 Pipeline":
    st.markdown("# Pipeline Status")
    st.caption(f"Dagster · {DAGSTER_URI}")
    st.markdown("")

    dagster_runs = get_dagster_runs()

    if dagster_runs:
        status_map = {
            "SUCCESS":  ("✅", "badge-green",  "Success"),
            "FAILURE":  ("❌", "badge-red",    "Failed"),
            "STARTED":  ("🔄", "badge-blue",   "Running"),
            "QUEUED":   ("⏳", "badge-yellow", "Queued"),
            "CANCELED": ("⛔", "badge-yellow", "Canceled"),
        }
        for r in dagster_runs[:8]:
            status = r.get("status", "UNKNOWN")
            icon, badge_cls, label = status_map.get(status, ("❓", "badge-yellow", status))
            start_ts = r.get("startTime")
            end_ts   = r.get("endTime")
            dur = f"{int(end_ts - start_ts)//60}m {int(end_ts - start_ts)%60}s" if start_ts and end_ts else "—"
            started  = datetime.fromtimestamp(start_ts).strftime("%H:%M:%S") if start_ts else "—"
            job_name = r.get("pipelineName") or r.get("jobName") or "unknown"
            st.markdown(f"""
            <div class="step-card" style="display:flex;align-items:center;gap:12px">
                <span style="font-size:20px">{icon}</span>
                <div style="flex:1">
                    <div style="font-weight:600;color:#0f172a;font-size:14px">{job_name}</div>
                    <div style="font-size:12px;color:#64748b">{started} · {dur}</div>
                </div>
                <span class="badge {badge_cls}">{label}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Chưa có pipeline run nào trong Dagster. Trigger thủ công tại http://localhost:3001")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Pipeline Architecture</p>', unsafe_allow_html=True)

    nodes  = ["S3 Raw", "Spark\nIngest", "S3\nProcessed", "Spark\nFeature",
              "Feast\nMaterialize", "Redis", "Redshift", "Train", "MLflow\nRegistry", "Serve API"]
    x      = [0, 1, 2, 3, 4,  5,  5,  3,  4,   5]
    y      = [0, 0, 0, 0, 0,  1, -1, -1, -1,  -2]
    edges  = [(0,1),(1,2),(2,3),(3,4),(4,5),(4,6),(2,7),(7,8),(8,9)]
    colors = ["#3b82f6","#3b82f6","#3b82f6","#3b82f6","#f59e0b",
              "#ef4444","#6366f1","#3b82f6","#f59e0b","#10b981"]

    fig = go.Figure()
    for e in edges:
        fig.add_scatter(x=[x[e[0]], x[e[1]]], y=[y[e[0]], y[e[1]]],
                        mode="lines", line=dict(color="#cbd5e1", width=1.5), showlegend=False)
    fig.add_scatter(x=x, y=y, mode="markers+text", text=nodes,
                    textposition="top center",
                    marker=dict(size=30, color=colors, line=dict(color="white", width=2)),
                    textfont=dict(size=10, color="#0f172a"), showlegend=False)
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=40),
                      paper_bgcolor="white", plot_bgcolor="white",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model":
    st.markdown("# Model Performance")
    st.caption(f"MLflow · {MLFLOW_URI} · experiment: {MODEL_NAME}")
    st.markdown("")

    if runs_df.empty:
        st.error("Không kết nối được MLflow. Đảm bảo service đang chạy tại http://localhost:5000")
    else:
        best = runs_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        for col, (lbl, key, fmt) in zip([c1, c2, c3, c4], [
            ("AUC-ROC",  "auc_roc",  "{:.4f}"),
            ("PR-AUC",   "pr_auc",   "{:.4f}"),
            ("Log Loss", "log_loss", "{:.4f}"),
            ("ECE",      "ece",      "{:.4f}"),
        ]):
            val = best.get(key)
            val_str = fmt.format(val) if val is not None else "—"
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value">{val_str}</div>
                    <div class="metric-delta delta-neu">→ {best['model']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="section-header">AUC-ROC by Model (all runs)</p>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=runs_df["model"],
                y=runs_df["auc_roc"],
                marker_color=["#3b82f6" if i == 0 else "#94a3b8" for i in range(len(runs_df))],
                text=[f"{v:.4f}" if v else "" for v in runs_df["auc_roc"]],
                textposition="outside",
            ))
            y_min = runs_df["auc_roc"].dropna().min() - 0.01 if not runs_df["auc_roc"].dropna().empty else 0.5
            y_max = runs_df["auc_roc"].dropna().max() + 0.01 if not runs_df["auc_roc"].dropna().empty else 1.0
            fig.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="white", plot_bgcolor="white",
                yaxis=dict(range=[y_min, y_max], gridcolor="#f1f5f9"),
                xaxis=dict(showgrid=False), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<p class="section-header">All Runs</p>', unsafe_allow_html=True)
            display_cols = [c for c in ["model", "auc_roc", "pr_auc", "log_loss", "f1", "started_at"] if c in runs_df.columns]
            st.dataframe(runs_df[display_cols], use_container_width=True, hide_index=True)

        st.markdown('<p class="section-header">Model Registry — ride_completion</p>', unsafe_allow_html=True)
        reg_df = get_registry()
        if not reg_df.empty:
            st.dataframe(reg_df, use_container_width=True, hide_index=True)
        else:
            st.info("Không có model nào trong registry. Chạy train_all.py và register model trước.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DRIFT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Drift":
    st.markdown("# Data Drift Monitor")
    st.caption("PSI — Reference vs Current (S3 features/initial)")
    st.markdown("")

    ref_df = get_reference()
    cur_df = get_features()

    if ref_df is None or cur_df is None:
        st.error("Không đọc được S3 features. Kiểm tra AWS credentials và bucket rideflow.")
    else:
        numeric_cols = [c for c in ref_df.columns
                        if c != TARGET and pd.api.types.is_numeric_dtype(ref_df[c])][:12]

        with st.spinner("Tính PSI..."):
            psi_rows = []
            for col in numeric_cols:
                psi = compute_psi(ref_df[col], cur_df[col])
                psi_rows.append({"feature": col, "psi": psi, "drifted": psi > 0.2})
            psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=True)

        n_drifted = int(psi_df["drifted"].sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Drifted Features (PSI > 0.2)</div>
                <div class="metric-value" style="color:{'#ef4444' if n_drifted > 2 else '#10b981'}">{n_drifted} / {len(psi_df)}</div>
                <div class="metric-delta {'delta-down' if n_drifted > 2 else 'delta-up'}">{'⚠️ Cần chú ý' if n_drifted > 2 else '✅ Ổn định'}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            max_psi  = psi_df["psi"].max()
            max_feat = psi_df.loc[psi_df["psi"].idxmax(), "feature"]
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Max PSI</div>
                <div class="metric-value">{max_psi:.3f}</div>
                <div class="metric-delta delta-neu">↑ {max_feat}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Dataset Size</div>
                <div class="metric-value">{len(cur_df):,}</div>
                <div class="metric-delta delta-neu">Ref: {len(ref_df):,} rows</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<p class="section-header">PSI per Feature</p>', unsafe_allow_html=True)
            colors = ["#ef4444" if d else "#3b82f6" for d in psi_df["drifted"]]
            fig = go.Figure(go.Bar(
                x=psi_df["psi"], y=psi_df["feature"], orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in psi_df["psi"]], textposition="outside",
            ))
            fig.add_vline(x=0.2, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Threshold 0.2")
            fig.update_layout(height=340, margin=dict(l=0, r=60, t=10, b=0),
                              paper_bgcolor="white", plot_bgcolor="white",
                              xaxis=dict(showgrid=True, gridcolor="#f1f5f9",
                                         range=[0, max(psi_df["psi"].max() * 1.3, 0.25)]),
                              yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<p class="section-header">So sánh phân phối</p>', unsafe_allow_html=True)
            sel = st.selectbox("Feature", psi_df["feature"].tolist()[::-1], label_visibility="collapsed")
            lo, hi = ref_df[sel].quantile(0.01), ref_df[sel].quantile(0.99)
            fig2 = go.Figure()
            fig2.add_histogram(x=ref_df[sel].clip(lo, hi),
                               name="Reference", opacity=0.65, marker_color="#94a3b8", nbinsx=30)
            fig2.add_histogram(x=cur_df[sel].clip(lo, hi),
                               name="Current",   opacity=0.65, marker_color="#3b82f6", nbinsx=30)
            fig2.update_layout(barmode="overlay", height=340,
                               margin=dict(l=0, r=0, t=10, b=0),
                               paper_bgcolor="white", plot_bgcolor="white",
                               xaxis=dict(title=sel, showgrid=False),
                               yaxis=dict(gridcolor="#f1f5f9"),
                               legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REAL-TIME
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Real-time":
    st.markdown("# Real-time Inference")
    st.caption("CatBoost Production · MLflow model loaded locally")
    st.markdown("")

    model   = load_model()
    feat_df = get_features()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<p class="section-header">Predict Single Row</p>', unsafe_allow_html=True)

        if model is None:
            st.error("Không load được model từ MLflow. Chạy train_all.py và register model trước.")
        elif feat_df is None:
            st.error("Không đọc được features từ S3.")
        else:
            feature_cols = [c for c in feat_df.columns if c != TARGET]

            if "pred_idx" not in st.session_state:
                st.session_state["pred_idx"] = 0

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🎲 Random", use_container_width=True):
                    st.session_state["pred_idx"] = random.randint(0, len(feat_df) - 1)
            with col_b:
                run_pred = st.button("🚀 Predict", use_container_width=True)

            idx = st.number_input("Row index", min_value=0, max_value=len(feat_df)-1,
                                  value=st.session_state["pred_idx"], step=1, key="idx_input")
            st.session_state["pred_idx"] = idx

            row        = feat_df.iloc[[idx]][feature_cols]
            true_label = feat_df.iloc[idx][TARGET]

            st.dataframe(row.T.rename(columns={row.index[0]: "value"}),
                         use_container_width=True, height=250)

            if run_pred:
                t0   = time.time()
                prob = float(model.predict_proba(row)[:, 1][0])
                lat  = (time.time() - t0) * 1000
                color = "#10b981" if prob > 0.65 else "#f59e0b" if prob > 0.45 else "#ef4444"
                label = "Likely Complete" if prob > 0.65 else "Uncertain" if prob > 0.45 else "Likely Cancel"
                st.markdown(f"""
                <div class="metric-card" style="text-align:center;margin-top:12px">
                    <div class="metric-label">Completion Probability</div>
                    <div class="metric-value" style="color:{color};font-size:40px">{prob:.1%}</div>
                    <div style="margin-top:8px">
                        <span class="badge" style="background:{color}22;color:{color};font-size:13px;padding:4px 16px">{label}</span>
                    </div>
                    <div style="font-size:11px;color:#94a3b8;margin-top:8px">
                        Actual: {'✅ Completed' if true_label else '❌ Cancelled'} · {lat:.1f}ms
                    </div>
                </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">Batch Sample — 100 rows</p>', unsafe_allow_html=True)

        if model is not None and feat_df is not None:
            feature_cols = [c for c in feat_df.columns if c != TARGET]
            sample = feat_df.sample(min(100, len(feat_df)), random_state=42).copy()
            probs  = model.predict_proba(sample[feature_cols])[:, 1]
            sample["predicted_prob"] = probs
            sample["correct"] = ((probs > 0.5).astype(int) == sample[TARGET]).astype(int)

            fig = go.Figure()
            comp_mask = sample[TARGET] == 1
            fig.add_histogram(x=sample.loc[comp_mask, "predicted_prob"],
                              name="Actual: Completed", opacity=0.7,
                              marker_color="#10b981", nbinsx=20)
            fig.add_histogram(x=sample.loc[~comp_mask, "predicted_prob"],
                              name="Actual: Cancelled", opacity=0.7,
                              marker_color="#ef4444", nbinsx=20)
            fig.update_layout(barmode="overlay", height=220,
                              margin=dict(l=0, r=0, t=10, b=0),
                              paper_bgcolor="white", plot_bgcolor="white",
                              xaxis=dict(title="Predicted Probability", showgrid=False),
                              yaxis=dict(gridcolor="#f1f5f9"),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

            acc = sample["correct"].mean()
            st.caption(f"Accuracy on sample: **{acc:.1%}**  ·  Mean prob: **{probs.mean():.3f}**")

            st.markdown('<p class="section-header">Sample Predictions</p>', unsafe_allow_html=True)
            disp = sample[["predicted_prob", TARGET, "correct"]].head(10).copy()
            disp.columns = ["Predicted Prob", "Actual", "Correct"]
            disp["Predicted Prob"] = disp["Predicted Prob"].round(4)
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("Model hoặc features chưa sẵn sàng.")
