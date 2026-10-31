"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations)
Provides both Global (dataset-level) and Local (patient-level) explanations
for Tree-based models (Random Forest, XGBoost, Decision Tree) and Linear models.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

try:
    from src.config import FIGURES_DIR
except ImportError:
    from config import FIGURES_DIR


def get_explainer(model: Any, background_data: Optional[np.ndarray] = None) -> shap.Explainer:
    """
    Selects the optimal SHAP explainer depending on model family:
    - TreeExplainer for Random Forest, Decision Tree, XGBoost
    - LinearExplainer for Logistic Regression
    - General Explainer as fallback
    """
    model_type = type(model).__name__.lower()

    if any(k in model_type for k in ["randomforest", "xgb", "decisiontree"]):
        return shap.TreeExplainer(model)
    elif "logistic" in model_type:
        if background_data is None:
            # Masker with zeros if no background data passed
            background_data = np.zeros((1, model.coef_.shape[1]))
        return shap.LinearExplainer(model, background_data)
    else:
        # Fallback KernelExplainer / Explainer
        if background_data is None:
            raise ValueError("Background data required for general Explainer.")
        return shap.Explainer(model, background_data)


def compute_global_shap(
    model: Any,
    X_trans: np.ndarray,
    feature_names: List[str],
    save_figures: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Computes global SHAP values across an entire dataset/test split.
    Generates summary beeswarm and feature importance bar plots.
    Returns:
    - shap_values matrix
    - DataFrame of sorted global feature importances
    """
    explainer = get_explainer(model, X_trans)
    shap_explanation = explainer(X_trans)

    # If explanation is multi-dimensional (e.g. RF 2 classes), extract class 1 (CKD positive)
    if len(shap_explanation.values.shape) == 3 and shap_explanation.values.shape[2] == 2:
        values = shap_explanation.values[:, :, 1]
    else:
        values = shap_explanation.values

    # Mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    if save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Beeswarm / Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            values,
            X_trans,
            feature_names=feature_names,
            show=False,
            plot_type="dot"
        )
        plt.title("SHAP Global Summary (Feature Impact on CKD Risk)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        beeswarm_path = FIGURES_DIR / "shap_summary_beeswarm.png"
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Feature Importance Bar Plot
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            values,
            X_trans,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        bar_path = FIGURES_DIR / "shap_feature_importance_bar.png"
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()

    return values, importance_df


def explain_patient_prediction(
    model: Any,
    patient_trans: np.ndarray,
    feature_names: List[str],
    background_trans: Optional[np.ndarray] = None,
    raw_feature_values: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes local SHAP explanation for a single patient record.
    Returns:
    - base_value: expected baseline model risk
    - prediction_proba: model's predicted probability of CKD
    - feature_contributions: list of dicts with feature names, values, shap values, and risk direction
    - waterfall_fig: Matplotlib figure for UI and PDF embedding
    """
    if patient_trans.ndim == 1:
        patient_trans = patient_trans.reshape(1, -1)

    explainer = get_explainer(model, background_trans)
    explanation = explainer(patient_trans)

    # Extract class 1 (CKD positive) values if binary output
    if len(explanation.values.shape) == 3 and explanation.values.shape[2] == 2:
        shap_vals = explanation.values[0, :, 1]
        base_val = explanation.base_values[0, 1] if hasattr(explanation.base_values, "__len__") else explanation.base_values
    else:
        shap_vals = explanation.values[0]
        base_val = explanation.base_values[0] if hasattr(explanation.base_values, "__len__") else explanation.base_values

    # Model probability
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(patient_trans)[0, 1])
    else:
        proba = float(model.predict(patient_trans)[0])

    # Rank contributions
    contributions = []
    for idx, (f_name, s_val) in enumerate(zip(feature_names, shap_vals)):
        raw_val = raw_feature_values.get(f_name, patient_trans[0, idx]) if raw_feature_values else patient_trans[0, idx]
        contributions.append({
            "feature": f_name,
            "raw_value": raw_val,
            "shap_value": float(s_val),
            "abs_shap": abs(float(s_val)),
            "impact": "Increases CKD Risk" if s_val > 0 else "Decreases CKD Risk"
        })

    # Sort by absolute impact descending
    contributions = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)

    # Generate Waterfall Plot Figure
    fig, ax = plt.subplots(figsize=(9, 6))
    single_exp = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=patient_trans[0],
        feature_names=feature_names
    )
    shap.plots.waterfall(single_exp, max_display=10, show=False)
    plt.title("Patient-Specific SHAP Risk Breakdown", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    result = {
        "base_value": float(base_val),
        "prediction_probability": proba,
        "is_ckd": bool(proba >= 0.50),
        "top_contributions": contributions,
        "waterfall_figure": fig
    }
    return result
