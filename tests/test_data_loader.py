"""
Tests for Data Loader and Anomaly Cleaning
"""
import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_raw_data, get_feature_and_target_matrices
from src.config import RAW_DATA_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


def test_raw_data_exists():
    assert RAW_DATA_PATH.exists(), "Raw CKD dataset must exist on disk."


def test_load_raw_data():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 400, "Dataset must contain exactly 400 records."
    # Ensure all numerical features are numeric
    for col in NUMERICAL_FEATURES:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric type."


def test_get_feature_and_target_matrices():
    X, y = get_feature_and_target_matrices(save_cleaned=False)
    assert len(X) == 400
    assert len(y) == 400
    assert X.shape[1] == len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
    # Check binary mapping (ckd=1, notckd=0)
    assert set(y.unique()).issubset({0, 1})
    assert (y == 1).sum() == 250, "Expected 250 CKD positive cases."
    assert (y == 0).sum() == 150, "Expected 150 healthy cases."
