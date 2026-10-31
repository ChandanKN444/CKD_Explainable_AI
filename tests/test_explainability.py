"""
Tests for SHAP Explainability Calculations
"""
import pytest
import joblib
import numpy as np
import matplotlib.pyplot as plt
from src.config import SAVED_MODELS_DIR, FIGURES_DIR
from src.explainability import (
    get_explainer,
    compute_global_shap,
    explain_patient_prediction
)


def test_explainer_initialization():
    model = joblib.load(SAVED_MODELS_DIR / "random_forest_model.joblib")
    explainer = get_explainer(model)
    assert explainer is not None


def test_explain_patient_prediction():
    model = joblib.load(SAVED_MODELS_DIR / "random_forest_model.joblib")
    test_data = joblib.load(SAVED_MODELS_DIR / "test_data.joblib")
    X_test_trans = test_data["X_test_trans"]
    feature_names = test_data["feature_names"]

    patient_vector = X_test_trans[0:1]
    res = explain_patient_prediction(
        model=model,
        patient_trans=patient_vector,
        feature_names=feature_names
    )

    assert "base_value" in res
    assert "prediction_probability" in res
    assert 0.0 <= res["prediction_probability"] <= 1.0
    assert "top_contributions" in res
    assert len(res["top_contributions"]) == len(feature_names)
    assert res["waterfall_figure"] is not None
    plt.close(res["waterfall_figure"])


def test_global_figures_exist():
    assert (FIGURES_DIR / "shap_summary_beeswarm.png").exists()
    assert (FIGURES_DIR / "shap_feature_importance_bar.png").exists()
    assert (FIGURES_DIR / "roc_curves_comparison.png").exists()
