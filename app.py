import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler





# ---------------------------------
# Helper Functions
# ---------------------------------

def detect_schema(df):
    """Classify columns into numerical, categorical, and date."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    return num_cols, cat_cols, date_cols

def summarize_data(df):
    """Return simple stats summary."""
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return summary

def detect_outliers(df, num_cols):
    """Z-score method for anomaly detection."""
    outlier_report = {}
    if not num_cols:
        return outlier_report
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[num_cols].dropna())
    z_scores = np.abs(scaled)
    for i, col in enumerate(num_cols):
        outliers = (z_scores[:, i] > 3).sum()
        outlier_report[col] = int(outliers)
    return outlier_report

def generate_charts(df, num_cols, cat_cols):
    """Generate some charts dynamically."""
    charts = []

    if num_cols:
        fig = px.histogram(df, x=num_cols[0], nbins=30, title=f"Distribution of {num_cols[0]}")
        charts.append(fig)

    if cat_cols:
        fig = px.bar(df[cat_cols[0]].value_counts().reset_index(),
                     x="index", y=cat_cols[0],
                     title=f"Counts of {cat_cols[0]}")
        charts.append(fig)

    if len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                         title=f"Scatter: {num_cols[0]} vs {num_cols[1]}")
        charts.append(fig)

    return charts

# ---------------------------------
# Streamlit App
# ---------------------------------

st.set_page_config(page_title="InsightLens", layout="wide")
st.title("🔎 InsightLens – Data Exploration without DuckDB")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Auto parse dates
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            continue

    num_cols, cat_cols, date_cols = detect_schema(df)

    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    st.write("### Dataset Summary")
    summary = summarize_data(df)
    st.json(summary)

    st.write("### Column Types")
    st.write({
        "Numerical": num_cols,
        "Categorical": cat_cols,
        "Dates": date_cols
    })

    st.write("### Outlier Detection")
    outliers = detect_outliers(df, num_cols)
    st.json(outliers)

    st.write("### Visual Insights")
    charts = generate_charts(df, num_cols, cat_cols)
    for fig in charts:
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to get started.")

# Quick Plotly test
st.subheader("🔍 Quick Plotly Test")
test_df = pd.DataFrame({
    "Category": ["A", "B", "C"],
    "Value": [10, 20, 15]
})
fig = px.bar(test_df, x="Category", y="Value", title="Test Chart")
st.plotly_chart(fig)

st.markdown("---")
st.caption("Built with ❤️ for fast, CPU-friendly insight discovery. © 2025 InsightLens")
