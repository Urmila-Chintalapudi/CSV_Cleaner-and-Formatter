import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Helper Functions
# -------------------------------
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_missing_values(df, strategy="drop", fill_value=None):
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df
    elif strategy == "custom" and fill_value is not None:
        return df.fillna(fill_value)
    return df

def generate_summary(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values (%)": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
        "Duplicate Rows": df.duplicated().sum()
    }

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="CSV Cleaner & Formatter", layout="wide")
st.title("🧹 CSV Cleaner & Formatter – Upgraded Version")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Data summary
    st.subheader("📌 Data Summary Report")
    summary = generate_summary(df)
    st.json(summary)

    # Cleaning options
    st.subheader("🛠 Data Cleaning Options")
    if st.checkbox("Normalize Column Names"):
        df = clean_column_names(df)
        st.success("✅ Column names normalized!")

    if st.checkbox("Remove Duplicate Rows"):
        df = remove_duplicates(df)
        st.success("✅ Duplicates removed!")

    missing_option = st.radio(
        "Handle Missing Values",
        ["None", "Drop", "Mean (numeric)", "Mode (categorical)", "Custom Fill"]
    )
    if missing_option == "Drop":
        df = handle_missing_values(df, "drop")
    elif missing_option == "Mean (numeric)":
        df = handle_missing_values(df, "mean")
    elif missing_option == "Mode (categorical)":
        df = handle_missing_values(df, "mode")
    elif missing_option == "Custom Fill":
        value = st.text_input("Enter custom fill value:")
        if value:
            df = handle_missing_values(df, "custom", fill_value=value)

    # Missing value heatmap
    st.subheader("🔎 Missing Value Heatmap")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("📈 Correlation Matrix (Numeric Columns)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for correlation matrix.")

    # Download cleaned data
    st.subheader("⬇️ Download Cleaned CSV")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")

else:
    st.info("Please upload a CSV file to start cleaning.")

st.markdown("---")
st.caption("Built with ❤️ for easy, visual data cleaning. © 2025")
