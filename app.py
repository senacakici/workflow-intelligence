"""
Workflow Intelligence Dashboard
--------------------------------
Run with: streamlit run dashboard/app.py
"""

import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Workflow Intelligence",
    page_icon="🧠",
    layout="wide",
)

CATEGORY_COLORS = {
    "development": "#4C9BE8",
    "review":      "#F4A261",
    "meeting":     "#2A9D8F",
    "admin":       "#E76F51",
    "planning":    "#A8DADC",
}


@st.cache_data
def load_data():
    df = pd.read_csv("data/activities_scored.csv", parse_dates=["timestamp"])
    alerts = pd.read_csv("data/workload_alerts.csv")
    weekly = pd.read_csv("data/weekly_workload.csv")
    return df, alerts, weekly


@st.cache_resource
def load_model():
    path = Path("models/task_classifier.pkl")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


df, alerts, weekly = load_data()
model = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Workflow Intelligence")
st.sidebar.markdown("---")

teams = ["All"] + sorted(df["team"].unique().tolist())
selected_team = st.sidebar.selectbox("Filter by Team", teams)

categories = ["All"] + sorted(df["predicted_category"].dropna().unique().tolist())
selected_cat = st.sidebar.selectbox("Filter by Category", categories)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.metric("Total Records", len(df))
st.sidebar.metric("Labeled", df["predicted_category"].notna().sum())
st.sidebar.metric("Anomalies Detected", int(df["if_anomaly"].sum()))

# ── Filter data ────────────────────────────────────────────────────────────────
filtered = df.copy()
if selected_team != "All":
    filtered = filtered[filtered["team"] == selected_team]
if selected_cat != "All":
    filtered = filtered[filtered["predicted_category"] == selected_cat]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🧠 Workforce & Workflow Intelligence Platform")
st.markdown("*Automated task classification · Weak supervision pre-labeling · Anomaly detection*")
st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Activities", len(filtered))
col2.metric("Avg Duration (min)", f"{filtered['duration_min'].mean():.0f}")
col3.metric("Anomalous Sessions", int(filtered["if_anomaly"].sum()))
col4.metric("Unique Users", filtered["user_id"].nunique())

st.markdown("---")

# ── Row 1: Category Distribution + Team Workload ───────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Task Category Distribution")
    cat_counts = filtered["predicted_category"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [CATEGORY_COLORS.get(c, "#888") for c in cat_counts.index]
    bars = ax.barh(cat_counts.index, cat_counts.values, color=colors)
    ax.set_xlabel("Number of Tasks")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    st.pyplot(fig)

with col_b:
    st.subheader("⏱ Avg Duration by Category")
    avg_dur = filtered.groupby("predicted_category")["duration_min"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    colors2 = [CATEGORY_COLORS.get(c, "#888") for c in avg_dur.index]
    bars2 = ax2.barh(avg_dur.index, avg_dur.values, color=colors2)
    ax2.set_xlabel("Avg Minutes")
    ax2.bar_label(bars2, fmt="%.0f", padding=3, fontsize=9)
    ax2.set_facecolor("#0e1117")
    fig2.patch.set_facecolor("#0e1117")
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    st.pyplot(fig2)

# ── Row 2: Activity by Hour + Team Breakdown ───────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🕐 Activity by Hour of Day")
    hourly = filtered["hour"].value_counts().sort_index()
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.plot(hourly.index, hourly.values, color="#4C9BE8", linewidth=2, marker="o", markersize=4)
    ax3.fill_between(hourly.index, hourly.values, alpha=0.2, color="#4C9BE8")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Count")
    ax3.set_facecolor("#0e1117")
    fig3.patch.set_facecolor("#0e1117")
    ax3.tick_params(colors="white")
    ax3.xaxis.label.set_color("white")
    ax3.yaxis.label.set_color("white")
    st.pyplot(fig3)

with col_d:
    st.subheader("👥 Workload by Team")
    team_work = filtered.groupby("team")["duration_min"].sum().sort_values(ascending=False)
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    ax4.bar(team_work.index, team_work.values / 60, color="#2A9D8F")
    ax4.set_ylabel("Total Hours")
    ax4.set_facecolor("#0e1117")
    fig4.patch.set_facecolor("#0e1117")
    ax4.tick_params(colors="white", axis="both")
    ax4.yaxis.label.set_color("white")
    st.pyplot(fig4)

st.markdown("---")

# ── Anomaly Alerts ─────────────────────────────────────────────────────────────
st.subheader("🚨 Workload Anomaly Alerts")
if len(alerts) > 0:
    display_alerts = alerts[["user_id", "week", "weekly_minutes", "z_score", "alert_type", "description"]].copy()
    display_alerts["z_score"] = display_alerts["z_score"].round(2)
    display_alerts["weekly_minutes"] = display_alerts["weekly_minutes"].astype(int)
    st.dataframe(display_alerts.head(10), use_container_width=True)
else:
    st.success("No workload anomalies detected.")

st.markdown("---")

# ── Live Prediction ────────────────────────────────────────────────────────────
st.subheader("🤖 Live Task Classifier")
st.markdown("Enter a task description to auto-classify it:")

task_input = st.text_input("Task Description", placeholder="e.g. Weekly engineering standup with product team")

if task_input and model:
    proba = model.predict_proba([task_input])[0]
    classes = model.classes_
    pred_idx = np.argmax(proba)
    pred_label = classes[pred_idx]
    confidence = proba[pred_idx]

    col_pred, col_conf = st.columns(2)
    col_pred.metric("Predicted Category", pred_label.upper())
    col_conf.metric("Confidence", f"{confidence:.1%}")

    fig5, ax5 = plt.subplots(figsize=(5, 2.5))
    colors5 = [CATEGORY_COLORS.get(c, "#888") for c in classes]
    ax5.barh(classes, proba, color=colors5)
    ax5.set_xlim(0, 1)
    ax5.set_xlabel("Probability")
    ax5.set_facecolor("#0e1117")
    fig5.patch.set_facecolor("#0e1117")
    ax5.tick_params(colors="white")
    ax5.xaxis.label.set_color("white")
    st.pyplot(fig5)
elif task_input and not model:
    st.warning("Model not loaded. Run `python models/task_classifier.py` first.")

st.markdown("---")
st.markdown(
    "<small>Built with Python · scikit-learn · Weak Supervision · FastAPI · Streamlit</small>",
    unsafe_allow_html=True,
)
