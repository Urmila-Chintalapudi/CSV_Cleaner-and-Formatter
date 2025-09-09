from __future__ import annotations
import os
import io
import re
import math
import typing as T
from dataclasses import dataclass

import numpy as np
import pandas as pd
import duckdb
from dateutil.parser import parse as date_parse
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# HuggingFace (free models)
from transformers import pipeline

# ────────────────────────────────────────────────────────────────
# HuggingFace setup
# ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_hf_model():
    """Load Flan-T5 for offline narration/explanations."""
    try:
        pipe = pipeline("text2text-generation", model="google/flan-t5-base")
        return pipe
    except Exception as e:
        st.warning(f"⚠️ Could not load HuggingFace model: {e}")
        return None

hf_pipe = load_hf_model()

# ────────────────────────────────────────────────────────────────
# Utility dataclasses
# ────────────────────────────────────────────────────────────────
@dataclass
class Schema:
    date_col: str | None
    numeric_cols: list[str]
    categorical_cols: list[str]
    value_col: str | None

# ────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────
def load_dataframe(uploaded_file: io.BytesIO | None) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    name = getattr(uploaded_file, "name", "uploaded")
    try:
        if name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

def profile_dataframe(df: pd.DataFrame) -> Schema:
    # Detect date column
    date_col = None
    for c in df.columns:
        try:
            if df[c].dtype == "object" and df[c].dropna().apply(lambda x: _looks_like_date(str(x))).mean() > 0.6:
                date_col = c
                df[c] = pd.to_datetime(df[c], errors="coerce")
                break
        except Exception:
            continue

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    value_col = None
    for kw in ["sales", "revenue", "amount", "qty", "quantity", "price", "profit", "units", "count"]:
        for c in numeric_cols:
            if kw in c.lower():
                value_col = c
                break
        if value_col:
            break
    if not value_col and numeric_cols:
        value_col = numeric_cols[0]
    return Schema(date_col=date_col, numeric_cols=numeric_cols, categorical_cols=categorical_cols, value_col=value_col)

def _looks_like_date(s: str) -> bool:
    try:
        date_parse(s, fuzzy=True)
        return True
    except Exception:
        return False

# ────────────────────────────────────────────────────────────────
# Core analytics
# ────────────────────────────────────────────────────────────────
def summarize_overall(df: pd.DataFrame, schema: Schema) -> dict:
    out = {}
    if not schema.value_col:
        return {"text": "No numeric columns found."}
    val = schema.value_col
    out["count"] = int(df[val].count())
    out["mean"] = float(df[val].mean())
    out["sum"] = float(df[val].sum())
    out["min"] = float(df[val].min())
    out["max"] = float(df[val].max())
    if schema.date_col:
        by_month = (
            df.dropna(subset=[schema.date_col])
            .assign(month=lambda d: d[schema.date_col].dt.to_period("M").dt.to_timestamp())
            .groupby("month")[val]
            .sum()
        )
        out["by_month"] = by_month
    return out

def detect_anomalies(series: pd.Series, contamination: float = 0.1) -> pd.DataFrame:
    s = series.dropna()
    if len(s) < 8:
        return pd.DataFrame({"x": s.index, "y": s.values, "anomaly": [False]*len(s)})
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(s.values.reshape(-1, 1))
    return pd.DataFrame({"x": s.index, "y": s.values, "anomaly": labels == -1})

def top_contributors_change(df: pd.DataFrame, schema: Schema, m1: pd.Timestamp, m2: pd.Timestamp, by: str) -> pd.DataFrame:
    if not (schema.date_col and schema.value_col):
        return pd.DataFrame()
    dfa = df.dropna(subset=[schema.date_col]).assign(month=lambda d: d[schema.date_col].dt.to_period("M").dt.to_timestamp())
    grp = dfa.groupby(["month", by])[schema.value_col].sum().reset_index()
    A = grp[grp["month"] == m1].set_index(by)[schema.value_col]
    B = grp[grp["month"] == m2].set_index(by)[schema.value_col]
    delta = (B.reindex(A.index.union(B.index), fill_value=0) - A.reindex(A.index.union(B.index), fill_value=0)).sort_values()
    return pd.DataFrame({by: delta.index, "delta": delta.values})

# ────────────────────────────────────────────────────────────────
# Narration
# ────────────────────────────────────────────────────────────────
def generate_narrative(prompt: str, fallback: str) -> str:
    if hf_pipe:
        try:
            resp = hf_pipe(prompt, max_new_tokens=120)[0]["generated_text"]
            return resp.strip()
        except Exception:
            return fallback
    return fallback

# ────────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────────
def plot_series(series: pd.Series, title: str):
    fig, ax = plt.subplots()
    series.plot(ax=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_bar(df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.bar(df, x=x, y=y, title=title)
    st.plotly_chart(fig, use_container_width=True)

def plot_anomalies(anoms: pd.DataFrame, title: str):
    fig = px.scatter(anoms, x="x", y="y", color="anomaly", title=title, color_discrete_map={True: "red", False: "blue"})
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="InsightLens", page_icon="📊", layout="wide")
st.title("📊 InsightLens – Free, Explainable Data Explorer")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])
use_demo = st.checkbox("Use demo dataset")

if uploaded is None and use_demo:
    rng = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    df_demo = pd.DataFrame({
        "date": rng,
        "region": np.random.choice(["North", "South", "East", "West"], len(rng)),
        "sales": np.random.normal(200, 50, len(rng)).clip(50, 500)
    })
    buf = io.BytesIO()
    df_demo.to_csv(buf, index=False)
    buf.name = "demo.csv"
    buf.seek(0)
    uploaded = buf

df = load_dataframe(uploaded)
if df is None:
    st.info("👆 Upload a dataset or tick demo mode.")
    st.stop()

schema = profile_dataframe(df)
st.sidebar.write("Detected Schema:", schema)

TAB1, TAB2, TAB3 = st.tabs(["Ask", "Insights", "Explore"])

# ────────────────────────────────────────────────────────────────
# TAB1 – Ask
# ────────────────────────────────────────────────────────────────
with TAB1:
    st.subheader("💬 Ask a Question")
    q = st.text_input("Question", value="Give me an overview")
    if st.button("Analyze"):
        if "overview" in q.lower():
            summary = summarize_overall(df, schema)
            st.write(summary)
            if "by_month" in summary:
                plot_series(summary["by_month"], "Monthly Trend")
            narration = generate_narrative(f"Summarize these stats: {summary}", fallback="Overview generated from dataset.")
            st.success(narration)

        elif "trend" in q.lower():
            summary = summarize_overall(df, schema)
            if "by_month" in summary:
                plot_series(summary["by_month"], "Monthly Trend")
                st.success("📈 Trend plotted.")

        elif "anomaly" in q.lower():
            if schema.value_col:
                anoms = detect_anomalies(df[schema.value_col])
                plot_anomalies(anoms, "Anomaly Detection")
                st.success("🔎 Anomalies highlighted.")

        else:
            st.info("Try questions like 'overview', 'trend', 'anomaly detection'.")

# ────────────────────────────────────────────────────────────────
# TAB2 – Insights
# ────────────────────────────────────────────────────────────────
with TAB2:
    st.subheader("📌 Auto Insights")
    if schema.value_col:
        summary = summarize_overall(df, schema)
        narration = generate_narrative(f"Find key insights in: {summary}", fallback="Key insights auto-generated.")
        st.write(narration)

# ────────────────────────────────────────────────────────────────
# TAB3 – SQL
# ────────────────────────────────────────────────────────────────
with TAB3:
    st.subheader("🧪 Explore with SQL")
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    sql = st.text_area("SQL", "SELECT * FROM data LIMIT 10;")
    if st.button("Run SQL"):
        try:
            res = con.execute(sql).fetchdf()
            st.dataframe(res)
        except Exception as e:
            st.error(f"SQL error: {e}")
st.markdown("---")
st.caption("Built with ❤️ for fast, CPU-friendly insight discovery. © 2025 InsightLens")