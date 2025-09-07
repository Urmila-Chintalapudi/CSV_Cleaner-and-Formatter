
"""
InsightLens – Context-Aware Data Explorer (Streamlit, single-file)
-----------------------------------------------------------------
A lightweight app to upload a CSV/Excel file, ask natural questions,
and get:
  • Numeric answers & tables
  • Auto-generated visualizations
  • Explanations of trends, anomalies, and top drivers

Designed to run on CPU (Intel i3 friendly). Uses rule-based analytics
for reliability and optional LLM narration via OpenAI/HF (if key set).

──────────────────────────────────────────────────────────────────
Installation
──────────────────────────────────────────────────────────────────
# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -U streamlit pandas numpy duckdb scikit-learn matplotlib plotly python-dateutil
# Optional (only if you want OpenAI narration)
pip install openai
# Optional (only if you want Hugging Face Transformers narration)
pip install transformers sentencepiece accelerate safetensors

──────────────────────────────────────────────────────────────────
Run
──────────────────────────────────────────────────────────────────
streamlit run app.py

Open in browser: http://localhost:8501

──────────────────────────────────────────────────────────────────
Optional environment variables
──────────────────────────────────────────────────────────────────
# For OpenAI API (narrative generation)
setx OPENAI_API_KEY "your_key_here"          # Windows (PowerShell: $env:OPENAI_API_KEY="...")
export OPENAI_API_KEY="your_key_here"        # macOS/Linux

# For Hugging Face Inference (local small model)
setx HF_LOCAL_NARRATION "1"

If no key is set, the app falls back to concise rule-based narration.

"""
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
        # default try CSV
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
            # "coerce" parse check on a sample
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

    # Heuristic for primary value column
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
    """Compare sum(value_col) between two months and show category drivers."""
    if not (schema.date_col and schema.value_col):
        return pd.DataFrame()
    if by is None:
        # pick a categorical with low cardinality
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
    return pd.concat([out.head(top_k),out.tail(top_k)]).drop_duplicates().reset_index(drop=True)
# ────────────────────────────────────────────────────────────────
# Narration (LLM optional, rule-based fallback)
# ────────────────────────────────────────────────────────────────
def generate_narrative(prompt: str, fallback: str = "") -> str:
    """Try OpenAI first (if key + lib present), else return fallback or a concise heuristic."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if _OPENAI_AVAILABLE and api_key:
        try:
            openai.api_key = api_key
            msg = [
                {"role": "system", "content": "You are a concise data analyst. Explain insights clearly with numbers and comparisons."},
                {"role": "user", "content": prompt},
            ]
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=msg, temperature=0.2, max_tokens=220)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
    # Fallback
    if fallback:
        return fallback
    return "Insight: Key changes identified. Consider drilling down by top categories and recent months for drivers."

# ────────────────────────────────────────────────────────────────
# Question understanding (simple router for common intents)
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
    # YYYY-MM
    m = re.match(r"(\d{4})[-/](\d{1,2})", tok)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(year=y, month=mo, day=1)
    # month name → pick latest year in data
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
st.title("📊 InsightLens – Context‑Aware Data Explorer")
st.caption("Upload a dataset → ask questions → get answers, charts, and explanations.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("LLM narration is optional. Without an API key, the app uses concise rule‑based text.")
    use_llm = st.checkbox("Use OpenAI for narration (requires OPENAI_API_KEY)", value=bool(os.getenv("OPENAI_API_KEY")))
    contamination = st.slider("Anomaly sensitivity (IsolationForest contamination)", 0.01, 0.3, 0.1, 0.01)
    st.markdown("---")
    st.markdown("**Tip:** Start with a tidy CSV that has a date column and at least one numeric value column (e.g., sales).")

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

# Demo dataset toggle
use_demo = st.checkbox("Use demo sales dataset (if no file uploaded)")

if uploaded is None and use_demo:
    # Create a small synthetic dataset
    rng = pd.date_range("2024-01-01", periods=240, freq="D")
    np.random.seed(42)
    regions = ["North", "South", "East", "West"]
    products = ["Alpha", "Beta", "Gamma"]
    data = []
    for dt in rng:
        for r in regions:
            for p in products:
                base = 100 + 20*np.sin((dt.dayofyear/365)*2*np.pi)  # seasonal
                shock = np.random.normal(0, 8)
                # Inject a March dip for Region West, Product Beta
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

# Tabs
TAB1, TAB2, TAB3 = st.tabs(["Ask", "Insights", "Explore (SQL)"])

# ────────────────────────────────────────────────────────────────
# TAB 1 – Ask (natural-ish questions with a small intent router)
# ────────────────────────────────────────────────────────────────
with TAB1:
    st.subheader("💬 Ask a question")
    default_q = "Why did sales drop in March?" if schema.date_col and schema.value_col else "Give me an overview."
    q = st.text_input("Enter your question:", value=default_q)
    btn = st.button("Analyze")

    if btn and q.strip():
        # Overview intent
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

        # Why drop intent
        elif _INTENT_PATTERNS["why_drop"].search(q) and schema.date_col and schema.value_col:
            m = _INTENT_PATTERNS["why_drop"].search(q)
            tok = m.group(2) if m else ""
            month_target = _parse_month_token(tok, df, schema.date_col)
            if month_target is None:
                st.warning("Couldn't parse the month. Try 'Why did sales drop in 2024-03?'")
            else:
                by_month = (df.dropna(subset=[schema.date_col])
                              .assign(__m=lambda d: d[schema.date_col].dt.to_period('M').dt.to_timestamp())
                              .groupby("__m")[schema.value_col]
                              .sum()
                              .sort_index())
                if month_target not in by_month.index:
                    st.warning("Requested month not found in data.")
                else:
                    prev_idx = list(by_month.index).index(month_target) - 1
                    if prev_idx < 0:
                        st.warning("No previous month to compare against.")
                    else:
                        prev_month = by_month.index[prev_idx]
                        curr_val = float(by_month.loc[month_target])
                        prev_val = float(by_month.loc[prev_month])
                        change = curr_val - prev_val
                        pct = (change / prev_val * 100) if prev_val else float('nan')
                        st.markdown(f"**Month:** {month_target.strftime('%b %Y')} | **Value:** {curr_val:.2f} | **Change vs {prev_month.strftime('%b %Y')}:** {change:+.2f} ({pct:+.1f}%)")
                        plot_series(by_month, title=f"{schema.value_col} by month")
                        # Drivers
                        drivers = top_contributors_change(df, schema, prev_month, month_target, by=None, top_k=5)
                        if not drivers.empty:
                            st.markdown("**Top Drivers of Change (category deltas)**")
                            st.dataframe(drivers)
                            plot_bar(drivers.iloc[:5, 0], drivers.iloc[:5, 1], title="Largest Negative Contributors", xlab=drivers.columns[0], ylab="Δ value")
                        fallback = (
                            f"{schema.value_col} changed by {pct:+.1f}% in {month_target.strftime('%b %Y')} vs {prev_month.strftime('%b %Y')}. "
                            f"Top categories contributed to the decline as shown."
                        )
                        prompt = (
                            f"Explain why {schema.value_col} changed in {month_target.strftime('%b %Y')} versus {prev_month.strftime('%b %Y')}.\n"
                            f"Current value: {curr_val:.2f}, previous: {prev_val:.2f}, change: {change:+.2f} ({pct:+.1f}%).\n"
                            f"Use the category deltas to identify key drivers. Be concise and business-friendly."
                        )
                        narration = generate_narrative(prompt, fallback=fallback)
                        st.success(narration)

        # Compare months intent
        elif _INTENT_PATTERNS["compare_months"].search(q) and schema.date_col and schema.value_col:
            m = _INTENT_PATTERNS["compare_months"].search(q)
            tok1, tok2 = m.group(1), m.group(3)
            m1 = _parse_month_token(tok1, df, schema.date_col)
            m2 = _parse_month_token(tok2, df, schema.date_col)
            if not (m1 and m2):
                st.warning("Couldn't parse both months. Try 'Compare 2024-02 vs 2024-03'.")
            else:
                by_month = (df.dropna(subset=[schema.date_col])
                              .assign(__m=lambda d: d[schema.date_col].dt.to_period('M').dt.to_timestamp())
                              .groupby("__m")[schema.value_col]
                              .sum())
                if not (m1 in by_month.index and m2 in by_month.index):
                    st.warning("One of the months not present in data.")
                else:
                    v1, v2 = float(by_month.loc[m1]), float(by_month.loc[m2])
                    change = v2 - v1
                    pct = (change / v1 * 100) if v1 else float('nan')
                    st.markdown(f"**{m1.strftime('%b %Y')} → {m2.strftime('%b %Y')}** | {schema.value_col}: {v1:.2f} → {v2:.2f} ({change:+.2f}, {pct:+.1f}%)")
                    plot_series(by_month.sort_index(), title=f"{schema.value_col} by month")
                    drivers = top_contributors_change(df, schema, m1, m2, by=None, top_k=5)
                    if not drivers.empty:
                        st.markdown("**Top Drivers of Change**")
                        st.dataframe(drivers)
                    narration = generate_narrative(
                        prompt=(
                            f"Compare {schema.value_col} between {m1.strftime('%b %Y')} and {m2.strftime('%b %Y')}. "
                            f"Explain key differences and top category drivers briefly."
                        ),
                        fallback=f"Change = {change:+.2f} ({pct:+.1f}%). See category deltas above for drivers.",
                    )
                    st.success(narration)

        # Show trend intent
        elif _INTENT_PATTERNS["show_trend"].search(q) and schema.date_col and schema.value_col:
            by_month = (df.dropna(subset=[schema.date_col])
                          .assign(__m=lambda d: d[schema.date_col].dt.to_period('M').dt.to_timestamp())
                          .groupby("__m")[schema.value_col]
                          .sum()
                          .sort_index())
            st.markdown("**Monthly Trend**")
            plot_series(by_month, title=f"{schema.value_col} by month")
            an = detect_anomalies(by_month, contamination=contamination)
            if an["anomaly"].any():
                st.markdown("**Detected Anomalies**")
                st.dataframe(an[an["anomaly"]])
            narration = generate_narrative(
                prompt=f"Provide a concise trend summary for monthly {schema.value_col}, highlighting anomalies if any.",
                fallback="Trend shown with anomalies flagged where values deviate significantly from typical levels.",
            )
            st.success(narration)

        else:
            st.info("I didn't recognize that question. Try: 'overview', 'Why did sales drop in March?', 'Compare 2024-02 vs 2024-03', or 'trend'.")

# ────────────────────────────────────────────────────────────────
# TAB 2 – Insights (auto summaries + anomalies without a question)
# ────────────────────────────────────────────────────────────────
with TAB2:
    st.subheader("🔍 Automatic Insights")
    if not (schema.date_col and schema.value_col):
        st.info("Auto insights require a detected date column and a numeric value column.")
    else:
        by_month = (df.dropna(subset=[schema.date_col])
                      .assign(__m=lambda d: d[schema.date_col].dt.to_period('M').dt.to_timestamp())
                      .groupby("__m")[schema.value_col]
                      .sum()
                      .sort_index())
        col1, col2 = st.columns([2, 1])
        with col1:
            plot_series(by_month, title=f"{schema.value_col} by month")
        with col2:
            st.metric("Latest month value", f"{by_month.iloc[-1]:.2f}")
            if len(by_month) > 1:
                change = by_month.iloc[-1] - by_month.iloc[-2]
                pct = change / by_month.iloc[-2] * 100 if by_month.iloc[-2] else float('nan')
                st.metric("Change vs prev month", f"{change:+.2f}", f"{pct:+.1f}%")
        an = detect_anomalies(by_month, contamination=contamination)
        st.markdown("**Anomalies**")
        st.dataframe(an[an["anomaly"]])
        narration = generate_narrative(
            prompt=(
                f"Summarize monthly {schema.value_col} trend, latest value, MoM change, and anomalies in 3-4 bullet points."
            ),
            fallback=(
                "• Trend visualized with latest month highlighted.\n"
                "• Month-over-month change computed.\n"
                "• Anomalies flagged using IsolationForest."
            ),
        )
        st.success(narration)

# ────────────────────────────────────────────────────────────────
# TAB 3 – Explore (SQL via DuckDB, power users)
# ────────────────────────────────────────────────────────────────
with TAB3:
    st.subheader("🧪 Explore with SQL (DuckDB)")
    st.caption("Run quick SQL queries in-memory. Your dataframe is available as table 'data'.")
    con = duckdb.connect(database=':memory:')
    con.register('data', df)
    default_sql = f"SELECT * FROM data LIMIT 10;"
    sql = st.text_area("SQL", height=140, value=default_sql)
    if st.button("Run SQL"):
        try:
            res = con.execute(sql).fetchdf()
            st.dataframe(res)
            # Auto chart for simple 2-col numeric/grouped queries
            if res.shape[1] >= 2 and pd.api.types.is_numeric_dtype(res.iloc[:,1]):
                try:
                    fig = px.bar(res, x=res.columns[0], y=res.columns[1])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"SQL error: {e}")

st.markdown("---")
st.caption("Built with ❤️ for fast, CPU‑friendly insight discovery. © 2025 InsightLens")
