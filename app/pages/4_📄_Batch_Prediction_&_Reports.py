"""
Batch Patient Prediction and Clinical Screening Page
Allows hospital teams to upload multiple patient records via CSV,
receive instant risk stratification, and export enriched diagnostic results.
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.config import (
    SAVED_MODELS_DIR,
    SCALER_DIR,
    SAMPLE_PATIENTS_DIR,
    MODELS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.preprocessor import load_preprocessor, transform_data

st.set_page_config(page_title="Batch Screening | CKD Explainable AI", page_icon="📄", layout="wide")

st.title("📄 High-Throughput Batch Screening & Risk Stratification")
st.markdown("Upload hospital cohort CSV files to screen multiple patients simultaneously and identify high-risk individuals.")

preprocessor_path = SCALER_DIR / "preprocessor.joblib"
if not preprocessor_path.exists():
    st.warning("⚠️ Preprocessor pipeline not found. Please train models first on the 'Model Comparison' page.")
    st.stop()

col_up, col_info = st.columns([2, 1])

with col_info:
    st.subheader("💡 Template & Sample Data")
    sample_csv_path = SAMPLE_PATIENTS_DIR / "batch_test_patients.csv"
    if sample_csv_path.exists():
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                "📥 Download Sample CSV Template",
                data=f.read(),
                file_name="sample_patient_cohort.csv",
                mime="text/csv",
                use_container_width=True
            )
    st.caption("Uploaded CSV must include the standard 24 clinical features (age, bp, sg, al, sc, hemo, etc.).")

with col_up:
    st.subheader("📤 Upload Cohort File")
    uploaded_file = st.file_uploader("Select Patient Records (CSV):", type=["csv"])

    selected_model = st.selectbox(
        "Screening Algorithm:",
        options=list(MODELS.keys()),
        index=0
    )

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(input_df)} patient records.")

        # Ensure all required features are present
        required_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        missing = [c for c in required_cols if c not in input_df.columns]

        if missing:
            st.error(f"Missing required columns in uploaded CSV: `{missing}`")
            st.stop()

        # Run Predictions
        model_slug = selected_model.lower().replace(" ", "_")
        model = joblib.load(SAVED_MODELS_DIR / f"{model_slug}_model.joblib")
        preprocessor = load_preprocessor(preprocessor_path)

        features_df = input_df[required_cols].copy()
        X_trans, feat_names = transform_data(features_df, preprocessor)

        y_pred = model.predict(X_trans)
        y_proba = model.predict_proba(X_trans)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        results_df = input_df.copy()
        results_df["CKD_Prediction"] = ["CKD Detected" if p == 1 else "Normal / No CKD" for p in y_pred]
        results_df["Risk_Probability"] = (y_proba * 100).round(1)

        def assign_risk_tier(p):
            if p >= 70.0:
                return "High Risk"
            elif p >= 40.0:
                return "Moderate Risk"
            return "Low Risk"

        results_df["Risk_Tier"] = results_df["Risk_Probability"].apply(assign_risk_tier)

        st.divider()
        st.subheader("📊 Cohort Screening Summary")

        k1, k2, k3, k4 = st.columns(4)
        total = len(results_df)
        high_risk = (results_df["Risk_Tier"] == "High Risk").sum()
        mod_risk = (results_df["Risk_Tier"] == "Moderate Risk").sum()
        low_risk = (results_df["Risk_Tier"] == "Low Risk").sum()

        k1.metric("Total Screened", total)
        k2.metric("High Risk Flagged", high_risk, delta=f"{high_risk/total:.1%}", delta_color="inverse")
        k3.metric("Moderate Risk", mod_risk)
        k4.metric("Low Risk / Normal", low_risk)

        # Reorder so predictions are visible upfront
        lead_cols = [c for c in ["patient_id", "name", "CKD_Prediction", "Risk_Probability", "Risk_Tier"] if c in results_df.columns]
        other_cols = [c for c in results_df.columns if c not in lead_cols]
        display_df = results_df[lead_cols + other_cols]

        st.subheader("📑 Enriched Patient Roster")
        st.dataframe(display_df, use_container_width=True, height=400)

        # CSV Download Button
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Annotated Screening Results (CSV)",
            data=csv_bytes,
            file_name="CKD_Screening_Annotated_Results.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Batch processing error: {e}")
