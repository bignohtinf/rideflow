import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RideFlow MLOps",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: #f8fafc; }
    
    .stApp { background: #f8fafc; }
    
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: none;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 12px; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-label { font-size: 12px; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #0f172a; line-height: 1; }
    .metric-delta { font-size: 12px; margin-top: 4px; }
    .delta-up   { color: #10b981; }
    .delta-down { color: #ef4444; }
    .delta-neu  { color: #64748b; }

    .section-header {
        font-size: 16px; font-weight: 600; color: #0f172a;
        margin: 0 0 16px 0; padding-bottom: 10px;
        border-bottom: 2px solid #e2e8f0;
    }

    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 11px; font-weight: 600;
    }
    .badge-green  { background: #dcfce7; color: #16a34a; }
    .badge-yellow { background: #fef9c3; color: #ca8a04; }
    .badge-red    { background: #fee2e2; color: #dc2626; }
    .badge-blue   { background: #dbeafe; color: #2563eb; }

    .pipeline-step {
        background: white; border-radius: 10px; padding: 14px 18px;
        border: 1px solid #e2e8f0; margin-bottom: 8px;
        display: flex; align-items: center; gap: 12px;
    }

    div[data-testid="stMetric"] {
        background: white; border-radius: 12px; padding: 16px;
        border: 1px solid #e2e8f0;
    }

    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
    
    h1 { color: #0f172a !important; font-weight: 700 !important; }
    h2 { color: #0f172a !important; font-weight: 600 !important; font-size: 18px !important; }
    h3 { color: #334155 !important; font-weight: 600 !important; font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ── Fake data helpers ─────────────────────────────────────────────────────────
np.random.seed(42)

def gen_daily_metrics(days=14):
    dates = [datetime(2026, 3, 11) + timedelta(days=i) for i in range(days)]
    return pd.DataFrame({
        "date": dates,
        "orders": np.random.randint(25000, 48000, days),
        "completion_rate": np.random.uniform(0.72, 0.82, days),
        "auc_roc": np.random.uniform(0.835, 0.862, days),
        "drift_score": np.clip(np.random.normal(0.12, 0.05, days), 0.02, 0.35),
        "avg_fee": np.random.uniform(95000, 115000, days),
        "latency_ms": np.random.randint(18, 45, days),
    })

def gen_feature_drift():
    features = [
        "supply_demand_ratio", "eta_confidence", "distance",
        "fee_per_km", "rush_hour", "user_waiting_time_seconds",
        "num_drivers", "eta_avg", "is_short_trip", "hour_of_day",
    ]
    return pd.DataFrame({
        "feature": features,
        "psi": np.random.uniform(0.01, 0.28, len(features)),
        "drifted": [True, False, False, True, False, False, False, True, False, False],
    }).sort_values("psi", ascending=True)

def gen_prediction_dist():
    completed     = np.random.beta(5, 2, 3000)
    not_completed = np.random.beta(2, 5, 1500)
    return completed, not_completed

def gen_pipeline_runs():
    now = datetime.now()
    return [
        {"name": "ingest_pipeline",   "status": "success", "duration": "2m 14s", "time": now - timedelta(hours=1)},
        {"name": "feature_pipeline",  "status": "success", "duration": "4m 38s", "time": now - timedelta(hours=1, minutes=5)},
        {"name": "inference_pipeline","status": "success", "duration": "1m 52s", "time": now - timedelta(hours=1, minutes=10)},
        {"name": "drift_alert_job",   "status": "warning", "duration": "0m 48s", "time": now - timedelta(hours=2)},
        {"name": "retrain_pipeline",  "status": "running", "duration": "—",      "time": now - timedelta(minutes=12)},
    ]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 RideFlow MLOps")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔁 Pipeline", "🧠 Model", "📡 Drift", "⚡ Real-time"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Environment**")
    env = st.selectbox("", ["Production", "Staging"], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

df = gen_daily_metrics()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# Overview")
    st.caption(f"Production · {datetime.now().strftime('%A, %d %B %Y')}")
    st.markdown("")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Orders Today</div>
            <div class="metric-value">41,234</div>
            <div class="metric-delta delta-up">↑ 8.2% vs yesterday</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Completion Rate</div>
            <div class="metric-value">76.4%</div>
            <div class="metric-delta delta-up">↑ 1.1pp</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Model AUC-ROC</div>
            <div class="metric-value">0.851</div>
            <div class="metric-delta delta-neu">→ stable</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Drift Score</div>
            <div class="metric-value">0.12</div>
            <div class="metric-delta delta-up">↓ below threshold</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">API Latency</div>
            <div class="metric-value">24ms</div>
            <div class="metric-delta delta-up">↓ 6ms vs avg</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<p class="section-header">Orders & Completion Rate — 14 Days</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_bar(x=df["date"], y=df["orders"], name="Orders",
                    marker_color="#e2e8f0", yaxis="y1", opacity=0.8)
        fig.add_scatter(x=df["date"], y=df["completion_rate"], name="Completion Rate",
                        line=dict(color="#3b82f6", width=2.5), yaxis="y2", mode="lines+markers",
                        marker=dict(size=5))
        fig.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Orders", showgrid=True, gridcolor="#f1f5f9"),
            yaxis2=dict(title="Completion Rate", overlaying="y", side="right",
                        tickformat=".0%", range=[0.6, 1.0]),
            legend=dict(orientation="h", y=1.1),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Travel Mode Distribution</p>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["Ô tô (2)", "Xe máy (1)", "Giao hàng (3)"],
            values=[58, 32, 10],
            hole=0.65,
            marker=dict(colors=["#3b82f6", "#10b981", "#f59e0b"]),
        ))
        fig2.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            annotations=[dict(text="500k", x=0.5, y=0.5, font_size=22,
                              font_color="#0f172a", showarrow=False)],
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔁 Pipeline":
    st.markdown("# Pipeline Status")
    st.caption("Dagster orchestration · All jobs")
    st.markdown("")

    runs = gen_pipeline_runs()
    status_map = {
        "success": ("✅", "badge-green",  "Success"),
        "warning": ("⚠️", "badge-yellow", "Warning"),
        "running": ("🔄", "badge-blue",   "Running"),
        "failed":  ("❌", "badge-red",    "Failed"),
    }

    for r in runs:
        icon, badge_cls, label = status_map[r["status"]]
        st.markdown(f"""
        <div class="pipeline-step">
            <span style="font-size:20px">{icon}</span>
            <div style="flex:1">
                <div style="font-weight:600;color:#0f172a;font-size:14px">{r['name']}</div>
                <div style="font-size:12px;color:#64748b">{r['time'].strftime('%H:%M:%S')} · {r['duration']}</div>
            </div>
            <span class="badge {badge_cls}">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Pipeline Architecture</p>', unsafe_allow_html=True)

    fig = go.Figure()
    nodes = ["S3 Raw", "Spark\nIngest", "GX\nValidate", "S3\nProcessed",
             "Spark\nFeature", "Feast\nMaterialize", "Redshift", "Redis",
             "Train", "MLflow\nRegistry", "Batch\nPredict", "Serve API"]
    x =    [0, 1, 2, 3,  3,  4,   5,   5,  6,  7,  8, 8]
    y =    [1, 1, 1, 1,  0,  0,   1,  -1,  0,  0,  1, -1]

    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(5,7),(3,8),(8,9),(9,10),(7,11)]
    for e in edges:
        fig.add_scatter(x=[x[e[0]], x[e[1]]], y=[y[e[0]], y[e[1]]],
                        mode="lines", line=dict(color="#cbd5e1", width=1.5),
                        showlegend=False)

    colors = ["#3b82f6","#3b82f6","#10b981","#3b82f6","#3b82f6","#f59e0b",
              "#6366f1","#ef4444","#3b82f6","#f59e0b","#3b82f6","#ef4444"]
    fig.add_scatter(x=x, y=y, mode="markers+text", text=nodes,
                    textposition="top center",
                    marker=dict(size=28, color=colors, line=dict(color="white", width=2)),
                    textfont=dict(size=10, color="#0f172a"), showlegend=False)
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="white", plot_bgcolor="white",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model":
    st.markdown("# Model Performance")
    st.caption("MLflow Registry · ride_completion")
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [("AUC-ROC", "0.851", "↑ 0.003", True),
               ("PR-AUC",  "0.823", "↑ 0.005", True),
               ("Log Loss","0.218", "↓ 0.004", True),
               ("ECE",     "0.031", "→ stable", None)]
    for col, (label, val, delta, up) in zip([c1,c2,c3,c4], metrics):
        cls = "delta-up" if up is True else "delta-down" if up is False else "delta-neu"
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-delta {cls}">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">AUC-ROC Over Time</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_scatter(x=df["date"], y=df["auc_roc"], mode="lines+markers",
                        line=dict(color="#3b82f6", width=2.5),
                        marker=dict(size=5),
                        fill="tozeroy", fillcolor="rgba(59,130,246,0.06)")
        fig.add_hline(y=0.80, line_dash="dash", line_color="#ef4444",
                      annotation_text="Min threshold 0.80", annotation_position="right")
        fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                          paper_bgcolor="white", plot_bgcolor="white",
                          yaxis=dict(range=[0.78, 0.90], gridcolor="#f1f5f9"),
                          xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Prediction Distribution</p>', unsafe_allow_html=True)
        comp, not_comp = gen_prediction_dist()
        fig2 = go.Figure()
        fig2.add_histogram(x=comp, name="Completed", opacity=0.7,
                           marker_color="#10b981", nbinsx=40)
        fig2.add_histogram(x=not_comp, name="Not Completed", opacity=0.7,
                           marker_color="#ef4444", nbinsx=40)
        fig2.update_layout(barmode="overlay", height=260,
                           margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="white", plot_bgcolor="white",
                           xaxis=dict(title="Completion Probability", showgrid=False),
                           yaxis=dict(gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-header">Model Registry</p>', unsafe_allow_html=True)
    registry_df = pd.DataFrame({
        "Version": ["v4", "v3", "v2", "v1"],
        "Model":   ["LightGBM", "CatBoost", "LightGBM", "XGBoost"],
        "Stage":   ["Production", "Staging", "Archived", "Archived"],
        "AUC-ROC": [0.851, 0.848, 0.839, 0.821],
        "Trained":  ["2026-03-24", "2026-03-17", "2026-03-10", "2026-03-01"],
    })
    st.dataframe(registry_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DRIFT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Drift":
    st.markdown("# Data Drift Monitor")
    st.caption("Evidently · Reference vs Production (2026-03-24)")
    st.markdown("")

    drift_df = gen_feature_drift()
    n_drifted = drift_df["drifted"].sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Drifted Features</div>
            <div class="metric-value" style="color:{'#ef4444' if n_drifted >= 3 else '#10b981'}">{n_drifted} / {len(drift_df)}</div>
            <div class="metric-delta {'delta-down' if n_drifted >= 3 else 'delta-up'}">
                {'⚠️ Above threshold' if n_drifted >= 3 else '✅ Below threshold'}
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Max PSI</div>
            <div class="metric-value">0.24</div>
            <div class="metric-delta delta-down">↑ supply_demand_ratio</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Retrain Trigger</div>
            <div class="metric-value">Pending</div>
            <div class="metric-delta delta-neu">→ Manual review</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<p class="section-header">PSI per Feature</p>', unsafe_allow_html=True)
        colors = ["#ef4444" if d else "#3b82f6" for d in drift_df["drifted"]]
        fig = go.Figure(go.Bar(
            x=drift_df["psi"], y=drift_df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in drift_df["psi"]],
            textposition="outside",
        ))
        fig.add_vline(x=0.2, line_dash="dash", line_color="#f59e0b",
                      annotation_text="Threshold 0.2")
        fig.update_layout(height=320, margin=dict(l=0,r=60,t=10,b=0),
                          paper_bgcolor="white", plot_bgcolor="white",
                          xaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 0.35]),
                          yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Drift Score — 14 Days</p>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_scatter(x=df["date"], y=df["drift_score"], mode="lines+markers",
                         line=dict(color="#f59e0b", width=2.5), marker=dict(size=5),
                         fill="tozeroy", fillcolor="rgba(245,158,11,0.06)")
        fig2.add_hline(y=0.2, line_dash="dash", line_color="#ef4444",
                       annotation_text="Alert threshold")
        fig2.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="white", plot_bgcolor="white",
                           xaxis=dict(showgrid=False),
                           yaxis=dict(range=[0, 0.4], gridcolor="#f1f5f9"))
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REAL-TIME
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Real-time":
    st.markdown("# Real-time Inference")
    st.caption("FastAPI · Feast Online Store · Redis")
    st.markdown("")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<p class="section-header">Predict Single Order</p>', unsafe_allow_html=True)
        with st.form("predict_form"):
            order_id     = st.text_input("Order ID", value="order_12345")
            num_drivers  = st.slider("Num Drivers",  1, 50, 15)
            num_orders   = st.slider("Num Orders",   1, 100, 40)
            eta_avg      = st.slider("ETA Avg (s)",  60, 900, 300)
            distance     = st.slider("Distance (km)", 1.0, 30.0, 7.5)
            travel_mode  = st.selectbox("Travel Mode", [1, 2, 3],
                                        format_func=lambda x: {1:"Xe máy",2:"Ô tô",3:"Giao hàng"}[x])
            rush_hour    = st.checkbox("Rush Hour", value=False)
            submitted    = st.form_submit_button("🚀 Predict", use_container_width=True)

        if submitted:
            sdr   = num_drivers / (num_orders + 1)
            score = min(0.95, max(0.05,
                0.6 + sdr * 0.15 - (eta_avg / 900) * 0.2
                + (0.05 if rush_hour else 0)
                - (0.05 if distance > 15 else 0)
            ))
            color = "#10b981" if score > 0.65 else "#f59e0b" if score > 0.45 else "#ef4444"
            label = "High" if score > 0.65 else "Medium" if score > 0.45 else "Low"
            st.markdown(f"""
            <div class="metric-card" style="margin-top:12px;text-align:center">
                <div class="metric-label">Completion Probability</div>
                <div class="metric-value" style="color:{color};font-size:40px">{score:.1%}</div>
                <div style="margin-top:8px">
                    <span class="badge" style="background:{color}22;color:{color};font-size:13px;padding:4px 16px">{label} Likelihood</span>
                </div>
                <div style="font-size:11px;color:#94a3b8;margin-top:8px">Latency: {random.randint(18,35)}ms</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">API Throughput & Latency</p>', unsafe_allow_html=True)
        times = pd.date_range(datetime.now() - timedelta(hours=1), datetime.now(), freq="1min")
        throughput = np.random.randint(200, 800, len(times))
        latency    = np.random.randint(15, 50, len(times))

        fig = go.Figure()
        fig.add_bar(x=times, y=throughput, name="Requests/min",
                    marker_color="#e0f2fe", yaxis="y1")
        fig.add_scatter(x=times, y=latency, name="Latency (ms)",
                        line=dict(color="#3b82f6", width=2), yaxis="y2")
        fig.update_layout(
            height=240, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis=dict(title="Req/min", showgrid=True, gridcolor="#f1f5f9"),
            yaxis2=dict(title="Latency (ms)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-header">Recent Predictions</p>', unsafe_allow_html=True)
        recent = pd.DataFrame({
            "order_id":   [f"order_{random.randint(10000,99999)}" for _ in range(6)],
            "prob":       np.random.uniform(0.3, 0.95, 6).round(3),
            "latency_ms": np.random.randint(18, 45, 6),
            "timestamp":  [(datetime.now() - timedelta(seconds=i*10)).strftime("%H:%M:%S") for i in range(6)],
        })
        recent["result"] = recent["prob"].apply(lambda p: "✅ Complete" if p > 0.5 else "❌ Cancel")
        st.dataframe(recent, use_container_width=True, hide_index=True)
