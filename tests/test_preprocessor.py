"""
Tests for Preprocessing Pipeline & Feature Names
"""
import pytest
import numpy as np
import pandas as pd
from src.data_loader import get_feature_and_target_matrices
from src.preprocessor import (
    create_preprocessor,
    fit_and_save_preprocessor,
    load_preprocessor,
    transform_data
)
from src.config import SCALER_DIR


def test_create_preprocessor():
    preprocessor = create_preprocessor()
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2


def test_fit_and_transform_pipeline():
    X, y = get_feature_and_target_matrices(save_cleaned=False)
    preprocessor, X_trans, feat_names = fit_and_save_preprocessor(X)

    assert isinstance(X_trans, np.ndarray)
    assert not np.isnan(X_trans).any(), "Transformed array should not contain NaNs."
    assert len(feat_names) == X_trans.shape[1]
    assert (SCALER_DIR / "preprocessor.joblib").exists()


def test_single_patient_transform():
    preprocessor = load_preprocessor()
    sample = {
        "age": 45.0, "bp": 80.0, "sg": 1.020, "al": 0.0, "su": 0.0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 110.0, "bu": 35.0, "sc": 1.0, "sod": 138.0, "pot": 4.2,
        "hemo": 14.5, "pcv": 42.0, "wbcc": 7500.0, "rbcc": 4.8,
        "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    }
    df_sample = pd.DataFrame([sample])
    trans, feat_names = transform_data(df_sample, preprocessor)
    assert trans.shape == (1, len(feat_names))
    assert not np.isnan(trans).any()
