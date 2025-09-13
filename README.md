# CSV Cleaner & Formatter – Smart Data Preprocessing Tool
CSV Cleaner & Formatter is a Streamlit app that lets you quickly clean messy datasets by removing duplicates, handling missing values, and standardizing column names. It also provides quick data summaries, visualizations, and an option to download the cleaned CSV for easy use in analysis or machine learning.

The app is built using Streamlit and Pandas, with visualization support from Matplotlib and Seaborn.
Upload CSV → Start by uploading any CSV file.
Data Summary → Instantly view number of rows, columns, missing values percentage, and duplicate rows.

Cleaning Options:
Normalize column names (lowercase, underscores instead of spaces).
Remove duplicate rows.

Handle missing values with multiple strategies:
Drop missing rows
Fill with mean (numeric columns)
Fill with mode (categorical columns)
Custom fill value (user-defined).

Visual Insights:
Missing value heatmap for spotting gaps in data.
Correlation matrix for numeric columns.

Download → Export the cleaned dataset as a ready-to-use CSV file.
