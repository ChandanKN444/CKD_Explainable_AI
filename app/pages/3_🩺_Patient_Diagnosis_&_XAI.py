"""
Patient Diagnosis & Explainable AI (SHAP) Page
Interactive clinical screening interface with live SHAP attribution and instant PDF report generation.
"""
import streamlit as st
import json
import joblib
import pandas as pd
import numpy as np
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
    FEATURE_METADATA
)
from src.preprocessor import load_preprocessor, transform_data
from src.explainability import explain_patient_prediction
from src.pdf_generator import generate_patient_pdf_report
from app.components.inputs import render_patient_input_form
from app.components.visualizers import create_risk_gauge, create_shap_horizontal_bar
from app.components.shap_plots import render_shap_waterfall_plot

st.set_page_config(page_title="Patient Diagnosis & XAI | CKD Explainable AI", page_icon="🩺", layout="wide")

st.title("🩺 Patient Diagnostic Screening & Explainable AI (SHAP)")
st.markdown("Input patient clinical laboratory values to compute personalized CKD probability and feature-level SHAP risk explanations.")

# Check for preprocessor & models
preprocessor_path = SCALER_DIR / "preprocessor.joblib"
if not preprocessor_path.exists():
    st.warning("⚠️ Preprocessor pipeline not found. Please train models first in the 'Model Comparison' page.")
    st.stop()

# Preset Sample Patient Loader
st.subheader("👤 Patient Profile Setup")
col_preset, col_model = st.columns([2, 1])

with col_preset:
    preset_choice = st.selectbox(
        "Load Pre-configured Clinical Profile (or customize below):",
        options=["Custom / Manual Input", "Healthy Patient (Normal Vitals)", "Early-Stage CKD Risk", "Severe Renal Impairment"]
    )

initial_vals = {}
patient_id = "PT-89421"
patient_name = "Patient Profile"

if preset_choice == "Healthy Patient (Normal Vitals)":
    with open(SAMPLE_PATIENTS_DIR / "healthy_patient.json", "r") as f:
        initial_vals = json.load(f)
        patient_id = initial_vals.get("id", "PT-NORM-01")
        patient_name = initial_vals.get("name", "Jane Doe")
elif preset_choice == "Early-Stage CKD Risk":
    with open(SAMPLE_PATIENTS_DIR / "early_stage_ckd_patient.json", "r") as f:
        initial_vals = json.load(f)
        patient_id = initial_vals.get("id", "PT-EARLY-02")
        patient_name = initial_vals.get("name", "Robert Smith")
elif preset_choice == "Severe Renal Impairment":
    with open(SAMPLE_PATIENTS_DIR / "severe_ckd_patient.json", "r") as f:
        initial_vals = json.load(f)
        patient_id = initial_vals.get("id", "PT-SEV-03")
        patient_name = initial_vals.get("name", "Arthur Pendelton")

with col_model:
    available_models = list(MODELS.keys())
    selected_model_name = st.selectbox(
        "Select AI Screening Algorithm:",
        options=available_models,
        index=0,
        help="Random Forest and XGBoost provide superior non-linear biomarker interaction capture."
    )

st.divider()

# Patient Data Entry Form
with st.form("patient_diagnostic_form"):
    patient_inputs = render_patient_input_form(initial_vals)
    submitted = st.form_submit_button("🔍 Run AI Diagnosis & Compute SHAP Explanations", use_container_width=True)

if submitted:
    # Load Model & Preprocessor
    model_slug = selected_model_name.lower().replace(" ", "_")
    model_path = SAVED_MODELS_DIR / f"{model_slug}_model.joblib"
    if not model_path.exists():
        st.error(f"Selected model '{selected_model_name}' artifact not found. Please train models first.")
        st.stop()

    model = joblib.load(model_path)
    preprocessor = load_preprocessor(preprocessor_path)

    # Convert patient dictionary into 1-row DataFrame
    patient_df = pd.DataFrame([patient_inputs])
    patient_trans, feature_names = transform_data(patient_df, preprocessor)

    # Load background data for SHAP if available
    background_trans = None
    test_data_path = SAVED_MODELS_DIR / "test_data.joblib"
    if test_data_path.exists():
        try:
            bg_data = joblib.load(test_data_path)
            background_trans = bg_data.get("X_test_trans")[:30] # small representative background
        except Exception:
            pass

    # Run Prediction & SHAP
    with st.spinner("Analyzing renal biomarkers and calculating SHAP values..."):
        try:
            prediction_result = explain_patient_prediction(
                model=model,
                patient_trans=patient_trans,
                feature_names=feature_names,
                background_trans=background_trans,
                raw_feature_values=patient_inputs
            )
        except Exception as e:
            st.error(f"Explainability computation failed: {e}")
            st.stop()

    st.divider()
    st.header("📋 Diagnostic Screening Outcome")

    is_ckd = prediction_result["is_ckd"]
    proba = prediction_result["prediction_probability"]

    res_col1, res_col2 = st.columns([3, 2])

    with res_col1:
        if is_ckd:
            st.error(f"""
            ### 🚨 Chronic Kidney Disease (CKD) Detected
            **Risk Assessment:** The model predicts high likelihood of CKD with a probability of **{proba * 100:.1f}%**.
            Immediate clinical correlation with serum creatinine clearance, eGFR, and nephrology consultation is strongly advised.
            """)
        else:
            st.success(f"""
            ### ✅ No Chronic Kidney Disease Detected
            **Risk Assessment:** The model predicts a normal renal profile with low CKD probability (**{proba * 100:.1f}%**).
            Continue routine preventive monitoring of blood pressure and blood glucose.
            """)

        # Out-of-Range Clinical Alerts
        out_of_range = []
        if patient_inputs.get("sc", 0) > 1.2:
            out_of_range.append(f"• **Serum Creatinine:** {patient_inputs['sc']} mg/dL (Elevated > 1.2)")
        if patient_inputs.get("al", 0) > 0:
            out_of_range.append(f"• **Proteinuria (Albumin):** Grade {int(patient_inputs['al'])} (Abnormal)")
        if patient_inputs.get("hemo", 15) < 12.0:
            out_of_range.append(f"• **Hemoglobin:** {patient_inputs['hemo']} g/dL (Anemic < 12.0)")
        if patient_inputs.get("bu", 0) > 45.0:
            out_of_range.append(f"• **Blood Urea:** {patient_inputs['bu']} mg/dL (Elevated > 45)")
        if patient_inputs.get("sg", 1.020) < 1.015:
            out_of_range.append(f"• **Specific Gravity:** {patient_inputs['sg']} (Subnormal < 1.015)")

        if out_of_range:
            st.warning("⚠️ **Abnormal Laboratory Flags:**\n\n" + "\n".join(out_of_range))

    with res_col2:
        gauge_fig = create_risk_gauge(proba)
        st.plotly_chart(gauge_fig, use_container_width=True)

    # SHAP Explainability Visualizations
    st.divider()
    st.header("🧠 Explainable AI: Why did the model make this prediction?")
    st.markdown("""
    SHAP values break down how each patient biomarker drove the prediction from the population baseline expected value.
    - **Red / Positive values:** Pushed risk **higher** towards CKD.
    - **Green / Negative values:** Pushed risk **lower** towards healthy.
    """)

    shap_col1, shap_col2 = st.columns([1, 1])

    with shap_col1:
        st.subheader("Feature Impact Breakdown")
        bar_fig = create_shap_horizontal_bar(prediction_result["top_contributions"])
        st.plotly_chart(bar_fig, use_container_width=True)

    with shap_col2:
        st.subheader("SHAP Waterfall Attribution Chart")
        render_shap_waterfall_plot(prediction_result["waterfall_figure"])

    # Detailed Table of Top Risk Contributors
    st.subheader("Top Contributing Biomarkers")
    top_items = []
    for it in prediction_result["top_contributions"][:6]:
        meta_f = FEATURE_METADATA.get(it["feature"], {})
        top_items.append({
            "Biomarker Feature": meta_f.get("name", it["feature"]),
            "Patient Observed Value": it["raw_value"],
            "Reference Interval": meta_f.get("normal_range", "N/A"),
            "SHAP Contribution Value": f"{it['shap_value']:+.4f}",
            "Risk Attribution Direction": it["impact"]
        })
    st.dataframe(pd.DataFrame(top_items), use_container_width=True, hide_index=True)

    # PDF Report Generation & Download
    st.divider()
    st.subheader("📄 Export Official Clinical Diagnostic Report")
    st.markdown("Generate a print-ready clinical PDF containing patient demographics, test results, risk scores, and SHAP attribution.")

    with st.spinner("Generating official clinical PDF report..."):
        try:
            pdf_path = generate_patient_pdf_report(
                patient_data=patient_inputs,
                prediction_result=prediction_result,
                model_name=selected_model_name,
                patient_id=patient_id,
                patient_name=patient_name
            )

            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.download_button(
                label="📥 Download Clinical PDF Diagnostic Report",
                data=pdf_bytes,
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True
            )
            st.success(f"Report generated successfully: `{pdf_path.name}`")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
