"""
Script to pre-compute and export global SHAP figures and test PDF generator.
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import joblib
from src.config import SAVED_MODELS_DIR, SAMPLE_PATIENTS_DIR
from src.explainability import compute_global_shap, explain_patient_prediction
from src.pdf_generator import generate_patient_pdf_report
from src.preprocessor import load_preprocessor, transform_data
import pandas as pd
import json

def run_asset_generation():
    print("Pre-computing Global SHAP explanations...")
    test_data = joblib.load(SAVED_MODELS_DIR / "test_data.joblib")
    model = joblib.load(SAVED_MODELS_DIR / "best_model.joblib")

    X_test_trans = test_data["X_test_trans"]
    feature_names = test_data["feature_names"]

    # Compute Global SHAP & Save figures
    compute_global_shap(model, X_test_trans, feature_names, save_figures=True)
    print("Global SHAP figures saved.")

    # Test PDF Generation with Sample Patient
    with open(SAMPLE_PATIENTS_DIR / "early_stage_ckd_patient.json", "r") as f:
        patient_data = json.load(f)

    preprocessor = load_preprocessor()
    patient_df = pd.DataFrame([patient_data])
    patient_trans, _ = transform_data(patient_df, preprocessor)

    pred_res = explain_patient_prediction(
        model=model,
        patient_trans=patient_trans,
        feature_names=feature_names,
        background_trans=X_test_trans[:30],
        raw_feature_values=patient_data
    )

    pdf_out = generate_patient_pdf_report(
        patient_data=patient_data,
        prediction_result=pred_res,
        model_name="Random Forest",
        patient_id="PT-EARLY-02",
        patient_name="Robert Smith"
    )
    print(f"Sample PDF successfully generated at: {pdf_out}")

if __name__ == "__main__":
    run_asset_generation()
