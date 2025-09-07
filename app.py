from __future__ import annotations
import os
import io
import re
import json
import math
import duckdb
import typing as T
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dateutil.parser import parse as date_parse
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
# Load API key from Streamlit secrets if available
import streamlit as st
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)


# ────────────────────────────────────────────────────────────────
# Optional LLM clients (used only if API keys / flags provided)
# ────────────────────────────────────────────────────────────────
_OPENAI_AVAILABLE = False
try:
    import openai  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    pass

# ────────────────────────────────────────────────────────────────
# Utility dataclasses
# ────────────────────────────────────────────────────────────────
@dataclass
class Schema:
    date_col: str | None
    numeric_cols: list[str]
    categorical_cols: list[str]
    value_col: str | None  # heuristic pick for primary metric (e.g., sales)

# ────────────────────────────────────────────────────────────────
# Data loading & profiling
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
    # Guess date column
    date_candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["date", "time", "month", "year", "day"]):
            date_candidates.append(c)
    date_col = None
    for c in date_candidates + list(df.columns):
        try:
            sample = df[c].dropna().astype(str).head(20)
            if len(sample) and sum([_looks_like_date(s) for s in sample]) / len(sample) > 0.6:
                date_col = c
                break
        except Exception:
            continue

    if date_col is not None:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            date_col = None

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    value_keywords = [
        "sales", "revenue", "amount", "qty", "quantity", "price", "profit",
        "units", "count", "visits", "clicks", "impressions", "score"
    ]
    value_col = None
    for kw in value_keywords:
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
# Core analytics helpers
# ────────────────────────────────────────────────────────────────
def summarize_overall(df: pd.DataFrame, schema: Schema) -> dict:
    out: dict[str, T.Any] = {}
    if not schema.value_col:
        return {"text": "No numeric columns found for summary."}
    val = schema.value_col
    out["value_col"] = val
    out["count"] = int(df[val].count())
    out["mean"] = float(df[val].mean())
    out["sum"] = float(df[val].sum())
    out["min"] = float(df[val].min())
    out["max"] = float(df[val].max())
    if schema.date_col:
        by_month = (df.dropna(subset=[schema.date_col])
                      .assign(__month=lambda d: d[schema.date_col].dt.to_period('M').dt.to_timestamp())
                      .groupby("__month")[val]
                      .sum()
                      .sort_index())
        out["by_month"] = by_month
        out["last_month"] = by_month.index.max() if len(by_month) else None
        out["last_month_value"] = float(by_month.iloc[-1]) if len(by_month) else None
    return out


def detect_anomalies(series: pd.Series, contamination: float = 0.1) -> pd.DataFrame:
    s = series.dropna()
    if len(s) < 8:
        return pd.DataFrame({"x": s.index, "y": s.values, "anomaly": [False]*len(s)})
    X = s.values.reshape(-1, 1)
    iso = IsolationForest(n_estimators=200, contamination=min(max(contamination, 0.01), 0.3), random_state=42)
    labels = iso.fit_predict(X)
    df_out = pd.DataFrame({"x": s.index, "y": s.values, "anomaly": labels == -1})
    return df_out


def top_contributors_change(df: pd.DataFrame, schema: Schema, period_a: pd.Timestamp, period_b: pd.Timestamp, by: str | None = None, top_k: int = 5) -> pd.DataFrame:
    if not (schema.date_col and schema.value_col):
        return pd.DataFrame()
    if by is None:
        cats = [c for c in schema.categorical_cols if df[c].nunique(dropna=True) <= 25 and c != schema.date_col]
        by = cats[0] if cats else None
        if by is None:
            return pd.DataFrame()
    dfa = (
        df.dropna(subset=[schema.date_col])
          .assign(__m=lambda d: d[schema.date_col].dt.to_period('M').dt.to_timestamp())
    )
    grp = dfa.groupby(["__m", by])[schema.value_col].sum().reset_index()
    A = grp[grp["__m"] == period_a].set_index(by)[schema.value_col]
    B = grp[grp["__m"] == period_b].set_index(by)[schema.value_col]
    all_idx = A.index.union(B.index)
    delta = (B.reindex(all_idx).fillna(0) - A.reindex(all_idx).fillna(0)).sort_values(ascending=True)
    out = pd.DataFrame({by: delta.index, "delta": delta.values})
    return  pd.concat([out.head(top_k),out.tail(top_k)]).drop_duplicates().reset_index(drop=True)

# ────────────────────────────────────────────────────────────────
# Narration (LLM optional, rule-based fallback)
# ────────────────────────────────────────────────────────────────
def generate_narrative(prompt: str, fallback: str = "") -> str:
    if _OPENAI_AVAILABLE and openai_api_key:
        try:
            openai.api_key = openai_api_key
            msg = [
                {"role": "system", "content": "You are a concise data analyst. Explain insights clearly with numbers and comparisons."},
                {"role": "user", "content": prompt},
            ]
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=msg,
                temperature=0.2,
                max_tokens=220
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"(LLM error: {e})"
    if fallback:
        return fallback
    return "Insight: Key changes identified. Consider drilling down by top categories and recent months for drivers."


# ────────────────────────────────────────────────────────────────
# Question understanding (regex patterns)
# ────────────────────────────────────────────────────────────────
_INTENT_PATTERNS = {
    "overview": re.compile(r"^(overview|summary|describe|show summary)$", re.I),
    "why_drop": re.compile(r"why .* (drop|decline|decrease).*(\bjan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}-\d{2})", re.I),
    "compare_months": re.compile(r"compare .* (\bjan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}-\d{2}).* (with|vs|to) .* (\bjan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}-\d{2})", re.I),
    "show_trend": re.compile(r"(trend|time series|month over month|mom|line chart)", re.I),
}

_MONTHS = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}


def _parse_month_token(tok: str, df: pd.DataFrame, date_col: str) -> pd.Timestamp | None:
    tok = tok.lower()
    m = re.match(r"(\d{4})[-/](\d{1,2})", tok)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(year=y, month=mo, day=1)
    if tok[:3] in _MONTHS:
        mo = _MONTHS[tok[:3]]
        years = df[date_col].dropna().dt.year.unique()
        if len(years) == 0:
            return None
        y = sorted(years)[-1]
        return pd.Timestamp(year=y, month=mo, day=1)
    return None

# ────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────
def plot_series(series: pd.Series, title: str = ""):
    fig, ax = plt.subplots(figsize=(7, 3))
    series.plot(ax=ax)
    ax.set_title(title or series.name)
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def plot_bar(index: T.Sequence, values: T.Sequence, title: str = "", xlab: str = "", ylab: str = ""):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(index, values)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig, clear_figure=True)

# ────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="InsightLens", page_icon="📊", layout="wide")
st.title("📊 InsightLens – Context-Aware Data Explorer")
st.caption("Upload a dataset → ask questions → get answers, charts, and explanations.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("LLM narration is optional. Without an API key, the app uses concise rule-based text.")
    use_llm = st.checkbox("Use OpenAI for narration (requires OPENAI_API_KEY)", value=bool(os.getenv("OPENAI_API_KEY")))
    contamination = st.slider("Anomaly sensitivity (IsolationForest contamination)", 0.01, 0.3, 0.1, 0.01)
    st.markdown("---")
    st.markdown("**Tip:** Start with a tidy CSV that has a date column and at least one numeric value column (e.g., sales).")

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])
use_demo = st.checkbox("Use demo sales dataset (if no file uploaded)")

if uploaded is None and use_demo:
    rng = pd.date_range("2024-01-01", periods=240, freq="D")
    np.random.seed(42)
    regions = ["North", "South", "East", "West"]
    products = ["Alpha", "Beta", "Gamma"]
    data = []
    for dt in rng:
        for r in regions:
            for p in products:
                base = 100 + 20*np.sin((dt.dayofyear/365)*2*np.pi)
                shock = np.random.normal(0, 8)
                adj = -35 if (dt.month == 3 and r == "West" and p == "Beta") else 0
                data.append({
                    "date": dt,
                    "region": r,
                    "product": p,
                    "sales": max(5, base + shock + adj)
                })
    demo_df = pd.DataFrame(data)
    buffer = io.BytesIO()
    demo_df.to_csv(buffer, index=False)
    buffer.name = "demo.csv"
    buffer.seek(0)
    uploaded = buffer

df = load_dataframe(uploaded)

if df is None:
    st.info("👆 Upload a dataset or tick 'Use demo sales dataset' to try the app.")
    st.stop()

schema = profile_dataframe(df)

with st.expander("🔎 Detected Schema", expanded=False):
    st.write(schema)
    st.write({"rows": len(df), "cols": df.shape[1]})
    st.dataframe(df.head(10))

TAB1, TAB2, TAB3 = st.tabs(["Ask", "Insights", "Explore (SQL)"])

# ────────────────────────────────────────────────────────────────
# TAB 1 – Ask
# ────────────────────────────────────────────────────────────────
with TAB1:
    st.subheader("💬 Ask a question")
    default_q = "Why did sales drop in March?" if schema.date_col and schema.value_col else "Give me an overview."
    q = st.text_input("Enter your question:", value=default_q)
    btn = st.button("Analyze")

    if btn and q.strip():
        # Rule-based checks
        if _INTENT_PATTERNS["overview"].match(q.strip()) or not (schema.date_col and schema.value_col):
            summary = summarize_overall(df, schema)
            st.write({k: v for k, v in summary.items() if k not in ("by_month",)})
            if "by_month" in summary and isinstance(summary["by_month"], pd.Series):
                st.markdown("**Monthly Trend**")
                plot_series(summary["by_month"], title=f"{schema.value_col} by month")
            text = f"Overall {schema.value_col or 'value'}: total={summary.get('sum'):.2f}, mean={summary.get('mean'):.2f}."
            if schema.date_col and summary.get("last_month") is not None:
                lm = pd.to_datetime(summary["last_month"]).strftime('%b %Y')
                text += f" Latest month ({lm}) value={summary.get('last_month_value'):.2f}."
            narration = generate_narrative(
                prompt=f"Summarize the dataset focusing on {schema.value_col} with monthly trend if present.",
                fallback=text,
            )
            st.success(narration)

        elif _INTENT_PATTERNS["why_drop"].search(q) and schema.date_col and schema.value_col:
            st.warning("Why-drop intent not expanded here for brevity.")

        elif _INTENT_PATTERNS["compare_months"].search(q) and schema.date_col and schema.value_col:
            st.warning("Compare-months intent not expanded here for brevity.")

        elif _INTENT_PATTERNS["show_trend"].search(q) and schema.date_col and schema.value_col:
            st.warning("Show-trend intent not expanded here for brevity.")

        else:
            # Fallback → OpenAI
            if use_llm and os.getenv("OPENAI_API_KEY"):
                prompt = f"""You are a helpful data analyst.
                The user asked: {q}
                Dataset sample:
                {df.head(20).to_string(index=False)}

                Answer the question using the dataset.
                If the question is about comparisons, calculate directly from the numbers.
                Be concise and clear."""
                narration = generate_narrative(prompt, fallback="I could not interpret the question with built-in rules.")
                st.success(narration)
            else:
                st.info("I didn't recognize that question. Try: 'overview', 'Why did sales drop in March?', 'Compare 2024-02 vs 2024-03', or 'trend'.")
if use_llm and openai_api_key:
    prompt = f"""You are a helpful data analyst.
    The user asked: {q}
    Dataset sample:
    {df.head(20).to_string(index=False)}

    Answer the question using the dataset.
    If the question is about comparisons, calculate directly from the numbers.
    Be concise and clear."""
    narration = generate_narrative(prompt, fallback="I could not interpret the question with built-in rules.")
    st.success(narration)
else:
    st.info("I didn't recognize that question. Try: 'overview', 'Why did sales drop in March?', 'Compare 2024-02 vs 2024-03', or 'trend'.")

# ────────────────────────────────────────────────────────────────
# TAB 2 – Insights
# ────────────────────────────────────────────────────────────────
with TAB2:
    st.subheader("🔍 Automatic Insights")
    st.info("Auto insights go here (same as before).")

# ────────────────────────────────────────────────────────────────
# TAB 3 – Explore
# ────────────────────────────────────────────────────────────────
with TAB3:
    st.subheader("🧪 Explore with SQL (DuckDB)")
    con = duckdb.connect(database=':memory:')
    con.register('data', df)
    sql = st.text_area("SQL", height=140, value="SELECT * FROM data LIMIT 10;")
    if st.button("Run SQL"):
        try:
            res = con.execute(sql).fetchdf()
            st.dataframe(res)
        except Exception as e:
            st.error(f"SQL error: {e}")

st.markdown("---")
st.caption("Built with ❤️ for fast, CPU-friendly insight discovery. © 2025 InsightLens")




