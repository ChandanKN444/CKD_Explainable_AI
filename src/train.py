"""
Model Training and Benchmarking Pipeline
Trains Random Forest, Logistic Regression, Decision Tree, and XGBoost models
with leakage-free preprocessing, metrics evaluation, and model serialization.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from src.config import (
        RANDOM_STATE,
        TEST_SIZE,
        MODELS,
        SAVED_MODELS_DIR,
        SCALER_DIR
    )
    from src.data_loader import get_feature_and_target_matrices
    from src.preprocessor import fit_and_save_preprocessor
    from src.evaluate import (
        evaluate_predictions,
        save_benchmark_metrics,
        plot_and_save_roc_curves,
        plot_and_save_confusion_matrices
    )
except ImportError:
    from config import (
        RANDOM_STATE,
        TEST_SIZE,
        MODELS,
        SAVED_MODELS_DIR,
        SCALER_DIR
    )
    from data_loader import get_feature_and_target_matrices
    from preprocessor import fit_and_save_preprocessor
    from evaluate import (
        evaluate_predictions,
        save_benchmark_metrics,
        plot_and_save_roc_curves,
        plot_and_save_confusion_matrices
    )


def initialize_model(model_name: str, custom_params: Dict[str, Any] = None):
    """Instantiate a machine learning model by name with default or custom hyperparameters."""
    config_entry = MODELS.get(model_name)
    if not config_entry:
        raise ValueError(f"Unknown model name: {model_name}. Allowed: {list(MODELS.keys())}")

    params = config_entry["params"].copy()
    if custom_params:
        params.update(custom_params)

    if model_name == "Random Forest":
        return RandomForestClassifier(**params)
    elif model_name == "Logistic Regression":
        return LogisticRegression(**params)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**params)
    elif model_name == "XGBoost":
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def train_and_evaluate_all_models() -> Tuple[Dict[str, Any], str]:
    """
    Complete end-to-end training workflow:
    1. Load data
    2. Stratified train-test split
    3. Leak-free preprocessor fit on train split only
    4. Model fitting & test evaluation
    5. Artifact serialization
    6. Performance comparison visualization
    """
    print("=" * 60)
    print("Starting CKD Model Training & Benchmarking Workflow")
    print("=" * 60)

    # 1. Load Data
    X, y = get_feature_and_target_matrices()
    print(f"Dataset successfully loaded. Total samples: {X.shape[0]}, Features: {X.shape[1]}")

    # 2. Stratified Train-Test Split (strictly 80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Fit Preprocessor on X_train ONLY
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor, X_train_trans, feature_names = fit_and_save_preprocessor(X_train)
    X_test_trans = preprocessor.transform(X_test)
    print(f"Preprocessing pipeline fitted on training data. Transformed feature count: {len(feature_names)}")

    # Save X_test and y_test for quick evaluation / SHAP notebook
    test_data_path = SAVED_MODELS_DIR / "test_data.joblib"
    joblib.dump({"X_test": X_test, "y_test": y_test, "X_test_trans": X_test_trans, "feature_names": feature_names}, test_data_path)

    # 4. Train and benchmark each model
    benchmarks = {}
    fitted_models = {}

    for model_name in MODELS.keys():
        print(f"\n--- Training: {model_name} ---")
        model = initialize_model(model_name)
        model.fit(X_train_trans, y_train)

        # Predictions
        y_pred = model.predict(X_test_trans)
        y_proba = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else None

        # Evaluation
        metrics = evaluate_predictions(y_test.values, y_pred, y_proba)
        benchmarks[model_name] = metrics
        fitted_models[model_name] = model

        # Serialize Model
        slug = model_name.lower().replace(" ", "_")
        model_save_path = SAVED_MODELS_DIR / f"{slug}_model.joblib"
        joblib.dump(model, model_save_path)
        print(f"  Accuracy : {metrics['accuracy'] * 100:.2f}%")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc'] if metrics['roc_auc'] else 'N/A'}")
        print(f"  Artifact saved -> {model_save_path}")

    # 5. Determine Best Model (by F1-score, tiebreaker ROC-AUC)
    best_model_name = max(
        benchmarks.keys(),
        key=lambda m: (benchmarks[m]["f1_score"], benchmarks[m]["roc_auc"] or 0.0)
    )
    print("\n" + "=" * 60)
    print(f"Best Performing Model: {best_model_name}")
    print(f"F1-Score: {benchmarks[best_model_name]['f1_score']:.4f} | Accuracy: {benchmarks[best_model_name]['accuracy'] * 100:.2f}%")
    print("=" * 60)

    # Save Best Model reference
    best_slug = best_model_name.lower().replace(" ", "_")
    best_model_file = SAVED_MODELS_DIR / f"{best_slug}_model.joblib"
    joblib.dump(fitted_models[best_model_name], SAVED_MODELS_DIR / "best_model.joblib")

    # 6. Save Benchmark Metrics & Plots
    save_benchmark_metrics(benchmarks)
    plot_and_save_roc_curves(benchmarks)
    plot_and_save_confusion_matrices(benchmarks)

    print("All models, preprocessors, evaluation metrics, and figures saved successfully.")
    return benchmarks, best_model_name


if __name__ == "__main__":
    train_and_evaluate_all_models()
