"""
CKD Explainable AI - Streamlit Web Application Main Entrypoint
"""
import streamlit as st
import pandas as pd
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="CKD Explainable AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
css_file = Path(__file__).resolve().parent.parent / "assets" / "css" / "custom.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🩺 Chronic Kidney Disease (CKD) Prediction with Explainable AI")
st.markdown(
    "**An Advanced Clinical Decision Support System leveraging Machine Learning and SHapley Additive exPlanations (SHAP)**"
)

st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💡 Clinical Problem & Purpose")
    st.markdown("""
    **Chronic Kidney Disease (CKD)** affects more than **10% of the global population**, with millions remaining undiagnosed until irreversible renal decline occurs. 

    Traditional machine learning models often behave as **'black boxes'**, predicting patient outcomes without transparent clinical justifications. In high-stakes medicine, clinicians need to know **why** an algorithm made a recommendation.

    This project delivers:
    - **High-Accuracy Multi-Model Prediction:** Benchmarking **Random Forest**, **XGBoost**, **Decision Tree**, and **Logistic Regression**.
    - **Leakage-Free Preprocessing:** Medical range validation, multivariate imputation, and feature scaling.
    - **Dual-Layer Explainable AI (SHAP):** Providing both global epidemiological biomarker importance and patient-level waterfall attribution.
    - **Automated Clinical Documentation:** Instant generation of downloadable, print-ready PDF diagnostic reports.
    """)

    st.subheader("🚀 Quick Navigation Guide")
    st.markdown("""
    Use the sidebar to explore the full diagnostic pipeline:
    1. **📊 Exploratory Data Analysis:** Interactive dataset visualization, correlations, and biomarker distributions.
    2. **🧪 Model Comparison:** Side-by-side performance benchmarking, ROC-AUC curves, and confusion matrices.
    3. **🩺 Patient Diagnosis & XAI:** Enter patient parameters, run live diagnosis, examine SHAP waterfall plots, and download PDF reports.
    4. **📄 Batch Screening:** Upload a multi-patient hospital CSV for rapid automated screening.
    """)

with col2:
    st.subheader("🔬 Technology Stack")
    st.info("""
    - **Core ML:** Scikit-Learn, XGBoost, Pandas, NumPy
    - **Explainability:** SHAP (TreeExplainer & LinearExplainer)
    - **Frontend:** Streamlit & Plotly
    - **Reporting:** ReportLab Clinical Engine
    - **Validation:** 5-Fold Stratified Cross-Validation
    """)

    st.subheader("🏥 Key Monitored Biomarkers")
    st.markdown("""
    | Biomarker | Clinical Importance | Normal Range |
    | :--- | :--- | :--- |
    | **Serum Creatinine** | Glomerular Filtration | 0.6 - 1.2 mg/dL |
    | **Hemoglobin** | Renal Erythropoietin | 12.0 - 16.5 g/dL |
    | **Blood Urea** | Nitrogen Waste Clearance | 15 - 45 mg/dL |
    | **Albumin in Urine** | Glomerular Barrier Integrity | Grade 0 (Negative) |
    | **Specific Gravity** | Urine Concentration Ability | 1.010 - 1.025 |
    """)

st.divider()
st.caption("College Capstone Project: Explainable AI in Healthcare | Clinical Decision Support System")
