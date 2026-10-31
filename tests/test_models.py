"""
Tests for Model Loading, Inference, and Metrics
"""
import pytest
import joblib
import numpy as np
from src.config import SAVED_MODELS_DIR, METRICS_DIR, MODELS


def test_saved_models_exist():
    for model_name in MODELS.keys():
        slug = model_name.lower().replace(" ", "_")
        model_file = SAVED_MODELS_DIR / f"{slug}_model.joblib"
        assert model_file.exists(), f"Model file missing: {model_file}"


def test_model_inference():
    test_data = joblib.load(SAVED_MODELS_DIR / "test_data.joblib")
    X_test_trans = test_data["X_test_trans"]

    for model_name in MODELS.keys():
        slug = model_name.lower().replace(" ", "_")
        model = joblib.load(SAVED_MODELS_DIR / f"{slug}_model.joblib")

        preds = model.predict(X_test_trans)
        assert len(preds) == len(X_test_trans)
        assert set(np.unique(preds)).issubset({0, 1})

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test_trans)
            assert probas.shape == (len(X_test_trans), 2)
            assert (probas >= 0.0).all() and (probas <= 1.0).all()


def test_metrics_json():
    metrics_path = METRICS_DIR / "model_comparison.json"
    assert metrics_path.exists(), "Benchmark metrics JSON must exist."
    import json
    with open(metrics_path, "r") as f:
        benchmarks = json.load(f)

    for model_name in MODELS.keys():
        assert model_name in benchmarks
        m = benchmarks[model_name]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_score"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
