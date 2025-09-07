from __future__ import annotations
import os
import io
import re
import duckdb
import typing as T
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dateutil.parser import parse as date_parse
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import streamlit as st

# ────────────────────────────────────────────────
# Optional OpenAI client (new API)
# ────────────────────────────────────────────────
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
except Exception:
    client = None

# ────────────────────────────────────────────────
# Utility dataclasses
# ────────────────────────────────────────────────
@dataclass
class Schema:
    date_col: str | None
    numeric_cols: list[str]
    categorical_cols: list[str]
    value_col: str | None

# ────────────────────────────────────────────────
# Data loading & profiling
# ────────────────────────────────────────────────
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
    date_col = None
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            date_col = c
            break
        except Exception:
            continue

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # pick value col
    value_col = None
    for kw in ["sales", "revenue", "amount", "price", "profit", "quantity", "spend"]:
        for c in numeric_cols:
            if kw in c.lower():
                value_col = c
                break
        if value_col:
            break
    if not value_col and numeric_cols:
        value_col = numeric_cols[0]

    return Schema(date_col=date_col, numeric_cols=numeric_cols,
                  categorical_cols=categorical_cols, value_col=value_col)

# ────────────────────────────────────────────────
# Core helpers
# ────────────────────────────────────────────────
def summarize_overall(df: pd.DataFrame, schema: Schema) -> dict:
    if not schema.value_col:
        return {"text": "No numeric column found"}
    val = schema.value_col
    out = {
        "value_col": val,
        "count": int(df[val].count()),
        "mean": float(df[val].mean()),
        "sum": float(df[val].sum()),
        "min": float(df[val].min()),
        "max": float(df[val].max())
    }
    if schema.date_col:
        by_month = (
            df.dropna(subset=[schema.date_col])
              .assign(month=lambda d: d[schema.date_col].dt.to_period("M").dt.to_timestamp())
              .groupby("month")[val].sum()
        )
        out["by_month"] = by_month
    return out


def detect_anomalies(series: pd.Series, contamination: float = 0.1) -> pd.DataFrame:
    s = series.dropna()
    if len(s) < 8:
        return pd.DataFrame({"x": s.index, "y": s.values, "anomaly": [False]*len(s)})
    X = s.values.reshape(-1, 1)
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)
    return pd.DataFrame({"x": s.index, "y": s.values, "anomaly": labels == -1})

# ────────────────────────────────────────────────
# Narration
# ────────────────────────────────────────────────
def generate_narrative(prompt: str, fallback: str = "") -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if _OPENAI_AVAILABLE and api_key:
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a concise data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=220,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(LLM error: {e}) " + fallback
    return fallback or "Insight not available."

# ────────────────────────────────────────────────
# Simple rule-based Q&A
# ────────────────────────────────────────────────
def rule_based_answer(q: str, df: pd.DataFrame, schema: Schema) -> str:
    ql = q.lower()

    # Which category spends more?
    for col in schema.categorical_cols:
        if "which" in ql and ("spend" in ql or "sales" in ql or "more" in ql):
            if schema.value_col and col:
                grp = df.groupby(col)[schema.value_col].mean().sort_values(ascending=False)
                top = grp.index[0]
                return f"📊 {col}: '{top}' spends the most on average ({grp.iloc[0]:.2f})."
    
    # Trend
    if "trend" in ql or "over time" in ql:
        if schema.date_col and schema.value_col:
            by_m = (df.dropna(subset=[schema.date_col])
                      .assign(month=lambda d: d[schema.date_col].dt.to_period("M").dt.to_timestamp())
                      .groupby("month")[schema.value_col].sum())
            return f"📈 Trend available for {schema.value_col}. Latest month = {by_m.index.max()}, value = {by_m.iloc[-1]:.2f}."

    return ""  # not matched

# ────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────
def plot_series(series: pd.Series, title: str = ""):
    fig, ax = plt.subplots(figsize=(7, 3))
    series.plot(ax=ax)
    ax.set_title(title or series.name)
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.set_page_config(page_title="InsightLens", page_icon="📊", layout="wide")
st.title("📊 InsightLens – Context-Aware Data Explorer")

with st.sidebar:
    use_llm = st.checkbox("Use OpenAI (if API key set)", value=bool(os.getenv("OPENAI_API_KEY")))
    contamination = st.slider("Anomaly sensitivity", 0.01, 0.3, 0.1, 0.01)

uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xls", "xlsx"])
use_demo = st.checkbox("Use demo dataset")

if uploaded is None and use_demo:
    rng = pd.date_range("2024-01-01", periods=180, freq="D")
    np.random.seed(42)
    data = pd.DataFrame({
        "date": rng,
        "region": np.random.choice(["North", "South", "East", "West"], len(rng)),
        "sales": np.random.randint(50, 200, len(rng))
    })
    buf = io.BytesIO()
    data.to_csv(buf, index=False)
    buf.seek(0)
    buf.name = "demo.csv"
    uploaded = buf

df = load_dataframe(uploaded)

if df is None:
    st.info("👆 Upload a dataset or tick demo option.")
    st.stop()

schema = profile_dataframe(df)

with st.expander("🔎 Schema"):
    st.write(schema)
    st.write({"rows": len(df), "cols": df.shape[1]})
    st.dataframe(df.head(10))

TAB1, TAB2, TAB3 = st.tabs(["Ask", "Insights", "SQL"])

with TAB1:
    st.subheader("💬 Ask a question")
    q = st.text_input("Enter question", value="Which region spends more?")
    if st.button("Analyze"):
        ans = rule_based_answer(q, df, schema)

        if ans:
            st.success(ans)
        else:
            if use_llm and os.getenv("OPENAI_API_KEY"):
                narration = generate_narrative(
                    f"Answer this question using the dataset:\n{q}\n\nSample:\n{df.head(20).to_string(index=False)}",
                    fallback="Could not interpret.")
                st.success(narration)
            else:
                st.info("I didn't recognize that question. Try asking about 'overview', 'trend', or 'which category spends more'.")

with TAB2:
    st.subheader("🔍 Auto Insights")
    summary = summarize_overall(df, schema)
    st.write(summary)
    if "by_month" in summary:
        plot_series(summary["by_month"], f"{schema.value_col} trend")

with TAB3:
    st.subheader("🧪 SQL Explorer")
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    sql = st.text_area("SQL", value="SELECT * FROM data LIMIT 10;")
    if st.button("Run SQL"):
        try:
            st.dataframe(con.execute(sql).fetchdf())
        except Exception as e:
            st.error(e)
st.markdown("---")
st.caption("Built with ❤️ for fast, CPU-friendly insight discovery. © 2025 InsightLens") 