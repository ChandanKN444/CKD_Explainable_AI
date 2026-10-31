"""
Exploratory Data Analysis (EDA) Page
Visualizes dataset statistics, missing value rates, biomarker distributions, and correlations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from pathlib import Path
import sys

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data_loader import load_raw_data, get_feature_and_target_matrices
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, FEATURE_METADATA

st.set_page_config(page_title="EDA | CKD Explainable AI", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis & Clinical Dataset Insights")
st.markdown("Inspect distributions, correlations, and biomarker characteristics of the Chronic Kidney Disease cohort.")

@st.cache_data
def load_data():
    raw_df = load_raw_data()
    X, y = get_feature_and_target_matrices(save_cleaned=False)
    combined = X.copy()
    combined["class"] = y.map({1: "CKD", 0: "Not CKD"})
    return raw_df, combined

try:
    raw_df, df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Key Cohort Statistics
st.subheader("📌 Cohort Summary")
m1, m2, m3, m4 = st.columns(4)
total_patients = len(df)
ckd_count = (df["class"] == "CKD").sum()
not_ckd_count = (df["class"] == "Not CKD").sum()
missing_cells_pct = (raw_df.isna().sum().sum() / (raw_df.shape[0] * raw_df.shape[1])) * 100

m1.metric("Total Patient Records", total_patients)
m2.metric("CKD Diagnosed Cases", f"{ckd_count} ({ckd_count/total_patients:.1%})")
m3.metric("Healthy / Non-CKD", f"{not_ckd_count} ({not_ckd_count/total_patients:.1%})")
m4.metric("Dataset Missing Values", f"{missing_cells_pct:.1f}%")

st.divider()

# Tabbed In-Depth Analysis
tab_overview, tab_dist, tab_corr, tab_cat, tab_missing = st.tabs([
    "📋 Data Explorer",
    "📈 Biomarker Distributions",
    "🔥 Correlation Heatmap",
    "🏥 Comorbidities & Symptoms",
    "❓ Missing Data Analysis"
])

with tab_overview:
    st.subheader("Raw Clinical Records Preview")
    st.dataframe(df, use_container_width=True, height=350)
    st.caption("Showing full dataset with cleaned values and target labels.")

with tab_dist:
    st.subheader("Biomarker Comparison by CKD Status")
    col_feat, col_plot_type = st.columns([2, 1])
    with col_feat:
        selected_num = st.selectbox(
            "Select Clinical Biomarker to Analyze:",
            options=NUMERICAL_FEATURES,
            format_func=lambda x: f"{FEATURE_METADATA.get(x, {}).get('name', x)} ({x})"
        )
    with col_plot_type:
        plot_type = st.radio("Chart Type", ["Box Plot", "Histogram / Distribution"], horizontal=True)

    meta = FEATURE_METADATA.get(selected_num, {})
    unit = meta.get("unit", "")
    norm_range = meta.get("normal_range", "N/A")

    st.info(f"**Parameter:** {meta.get('name', selected_num)} | **Normal Reference Interval:** {norm_range} {unit}")

    if plot_type == "Box Plot":
        fig = px.box(
            df,
            x="class",
            y=selected_num,
            color="class",
            points="all",
            color_discrete_map={"CKD": "#E53E3E", "Not CKD": "#38A169"},
            labels={"class": "Diagnostic Group", selected_num: f"{meta.get('name', selected_num)} ({unit})"},
            title=f"Distribution of {meta.get('name', selected_num)} by Disease Status"
        )
    else:
        fig = px.histogram(
            df,
            x=selected_num,
            color="class",
            barmode="overlay",
            marginal="rug",
            color_discrete_map={"CKD": "#E53E3E", "Not CKD": "#38A169"},
            labels={"class": "Diagnostic Group", selected_num: f"{meta.get('name', selected_num)} ({unit})"},
            title=f"Histogram of {meta.get('name', selected_num)}"
        )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab_corr:
    st.subheader("Correlation Matrix (Numerical Features)")
    num_df = df[NUMERICAL_FEATURES].dropna()
    corr = num_df.corr().round(2)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pairwise Pearson Correlation Matrix"
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Strong negative correlations are visible between Serum Creatinine and Hemoglobin/PCV, reflecting renal erythropoietin depletion.")

with tab_cat:
    st.subheader("Categorical Feature Prevalence in CKD vs Non-CKD")
    selected_cat = st.selectbox(
        "Select Clinical Categorical Factor:",
        options=CATEGORICAL_FEATURES,
        format_func=lambda x: f"{FEATURE_METADATA.get(x, {}).get('name', x)} ({x})"
    )

    ct = pd.crosstab(df[selected_cat], df["class"], normalize="index") * 100
    ct_reset = ct.reset_index().melt(id_vars=selected_cat, var_name="Class", value_name="Percentage")

    fig_cat = px.bar(
        ct_reset,
        x=selected_cat,
        y="Percentage",
        color="Class",
        barmode="group",
        color_discrete_map={"CKD": "#E53E3E", "Not CKD": "#38A169"},
        title=f"CKD Prevalence by {FEATURE_METADATA.get(selected_cat, {}).get('name', selected_cat)}",
        labels={"Percentage": "% within Category"}
    )
    fig_cat.update_layout(height=400)
    st.plotly_chart(fig_cat, use_container_width=True)

with tab_missing:
    st.subheader("Missing Value Frequency by Feature")
    missing_df = raw_df.isna().sum().reset_index()
    missing_df.columns = ["Feature", "Missing_Count"]
    missing_df["Missing_Percentage"] = (missing_df["Missing_Count"] / len(raw_df)) * 100
    missing_df = missing_df.sort_values(by="Missing_Percentage", ascending=False)

    fig_miss = px.bar(
        missing_df[missing_df["Missing_Count"] > 0],
        x="Feature",
        y="Missing_Percentage",
        text_auto=".1f",
        title="Percentage of Missing Values per Feature in Raw Dataset",
        color="Missing_Percentage",
        color_continuous_scale="Reds"
    )
    fig_miss.update_layout(height=400, yaxis_title="% Missing Values")
    st.plotly_chart(fig_miss, use_container_width=True)
    st.info("Our preprocessor addresses these missing rates via multivariate median and mode imputation without data leakage.")
