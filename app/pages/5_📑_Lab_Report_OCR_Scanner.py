"""
Medical Lab Report OCR Scanner & Clinician Verification Page
Extracts patient test values from digital PDFs or lab images, presents them for
mandatory clinician manual verification, and performs instant XAI diagnosis.
"""
import streamlit as st
import pandas as pd
import json
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
    FEATURE_METADATA,
    MODELS
)
from src.ocr_extractor import process_lab_document
from src.preprocessor import load_preprocessor, transform_data
from src.explainability import explain_patient_prediction
from src.pdf_generator import generate_patient_pdf_report
from app.components.inputs import render_patient_input_form
from app.components.visualizers import create_risk_gauge, create_shap_horizontal_bar
from app.components.shap_plots import render_shap_waterfall_plot

st.set_page_config(page_title="Lab OCR Scanner | CKD Explainable AI", page_icon="📑", layout="wide")

st.title("📑 Medical Lab Report OCR Scanner & Clinician Verification")
st.markdown("""
Extract clinical biomarkers directly from digital laboratory PDF reports or scanned images.
**Clinical Safeguard:** In adherence to healthcare AI standards, all automated extractions require 
**mandatory clinician verification** before diagnostic inference.
""")

preprocessor_path = SCALER_DIR / "preprocessor.joblib"
if not preprocessor_path.exists():
    st.warning("⚠️ Preprocessor pipeline not found. Please train models first on the 'Model Comparison' page.")
    st.stop()

# Document Upload Section
col_upload, col_sample = st.columns([3, 2])

with col_sample:
    st.subheader("💡 Testing Sample Lab Report")
    sample_pdf_path = SAMPLE_PATIENTS_DIR / "sample_lab_report.pdf"
    load_sample = st.button("🧪 Load Sample Electronic Lab Report (PDF)", use_container_width=True)

    if sample_pdf_path.exists():
        with open(sample_pdf_path, "rb") as f:
            st.download_button(
                "📥 Download Sample Lab Report (PDF)",
                data=f.read(),
                file_name="sample_lab_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

with col_upload:
    st.subheader("📤 Upload Patient Lab Report")
    uploaded_file = st.file_uploader(
        "Upload PDF or Scanned Image (PNG, JPG):",
        type=["pdf", "png", "jpg", "jpeg"]
    )

file_bytes = None
file_name = None

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
elif load_sample and sample_pdf_path.exists():
    with open(sample_pdf_path, "rb") as f:
        file_bytes = f.read()
    file_name = "sample_lab_report.pdf"

if file_bytes is not None:
    with st.spinner("Processing document and extracting laboratory biomarkers..."):
        ocr_result = process_lab_document(file_bytes, file_name)

    extracted_vals = ocr_result.get("extracted_features", {})
    match_count = ocr_result.get("match_count", 0)

    st.divider()
    st.subheader(f"🔍 Automated Extraction Results for: `{file_name}`")

    sum_col1, sum_col2 = st.columns([1, 2])
    with sum_col1:
        st.metric("Biomarkers Detected", f"{match_count} / 13 key parameters")
        if ocr_result.get("error_notice"):
            st.info(ocr_result["error_notice"])

    with sum_col2:
        with st.expander("📄 View Raw Extracted Document Text"):
            st.text(ocr_result.get("extracted_text", "No text extracted."))

    st.subheader("🩺 Clinician Verification Form (Human-in-the-Loop)")
    st.info("Please review the pre-populated values extracted from the document. Verify accuracy and supply any clinical history or symptoms.")

    # Form with extracted values pre-filled
    with st.form("ocr_verification_form"):
        verified_inputs = render_patient_input_form(extracted_vals)
        selected_model = st.selectbox(
            "Screening Algorithm:",
            options=list(MODELS.keys()),
            index=0
        )
        verified_submit = st.form_submit_button("✅ Verify Values & Run AI Diagnosis", use_container_width=True)

    if verified_submit:
        model_slug = selected_model.lower().replace(" ", "_")
        model = joblib.load(SAVED_MODELS_DIR / f"{model_slug}_model.joblib")
        preprocessor = load_preprocessor(preprocessor_path)

        patient_df = pd.DataFrame([verified_inputs])
        patient_trans, feature_names = transform_data(patient_df, preprocessor)

        background_trans = None
        test_data_path = SAVED_MODELS_DIR / "test_data.joblib"
        if test_data_path.exists():
            bg_data = joblib.load(test_data_path)
            background_trans = bg_data.get("X_test_trans")[:30]

        with st.spinner("Running diagnosis and SHAP explanation on verified data..."):
            pred_res = explain_patient_prediction(
                model=model,
                patient_trans=patient_trans,
                feature_names=feature_names,
                background_trans=background_trans,
                raw_feature_values=verified_inputs
            )

        st.divider()
        st.header("📋 Diagnostic Screening Outcome (Verified)")

        is_ckd = pred_res["is_ckd"]
        proba = pred_res["prediction_probability"]

        r_col1, r_col2 = st.columns([3, 2])
        with r_col1:
            if is_ckd:
                st.error(f"""
                ### 🚨 Chronic Kidney Disease (CKD) Detected
                **Risk Assessment:** The model predicts high probability of CKD (**{proba * 100:.1f}%**).
                """)
            else:
                st.success(f"""
                ### ✅ Normal Renal Profile (Low Risk)
                **Risk Assessment:** The model predicts low CKD probability (**{proba * 100:.1f}%**).
                """)

        with r_col2:
            st.plotly_chart(create_risk_gauge(proba), use_container_width=True)

        st.subheader("🧠 SHAP Feature Attribution for Verified Patient")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.plotly_chart(create_shap_horizontal_bar(pred_res["top_contributions"]), use_container_width=True)
        with v_col2:
            render_shap_waterfall_plot(pred_res["waterfall_figure"])

        # PDF Download
        pdf_path = generate_patient_pdf_report(
            patient_data=verified_inputs,
            prediction_result=pred_res,
            model_name=selected_model,
            patient_id="PT-OCR-VERIFIED",
            patient_name="Lab Report Patient"
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download Verified Clinical PDF Report",
                data=f.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True
            )
