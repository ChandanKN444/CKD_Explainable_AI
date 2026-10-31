"""
Data Loading and Validation Module for CKD Explainable AI
"""
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd

try:
    from src.config import (
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN
    )
except ImportError:
    from config import (
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN
    )


def clean_text_field(val):
    """Normalize string fields by stripping whitespace, tabs, and handling missing tokens."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().replace("\t", "").lower()
    if val_str in ["?", "", "nan", "null"]:
        return np.nan
    return val_str


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw CKD dataset, clean common anomalies (whitespace, tabs, '?'),
    convert numerical string columns to floats, and standardize categorical values.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle alias column naming variations if present in any standard UCI version
    rename_map = {
        "classification": "class",
        "wc": "wbcc",
        "rc": "rbcc"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Clean object/string columns of tabs and '?'
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean_text_field)

    # Specific typographical corrections common to the UCI CKD dataset
    typo_corrections = {
        "dm": {"\tyes": "yes", " yes": "yes", "\tno": "no"},
        "cad": {"\tno": "no"},
        "class": {"ckd\t": "ckd", "notckd": "notckd"}
    }
    for col, mapping in typo_corrections.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # Convert numerical features that may have been parsed as string due to '?' or tabs
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_feature_and_target_matrices(
    filepath: Optional[Path] = None,
    save_cleaned: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw data, cleans it, separates features and target, and encodes target as binary:
    - ckd -> 1
    - notckd -> 0
    """
    df = load_raw_data(filepath)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    # Drop any row where target is missing
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Map target: ckd -> 1, notckd -> 0
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"ckd": 1, "notckd": 0})
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Invalid target class labels encountered during mapping.")

    # Ensure all required features are present
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")

    X = df[expected_cols].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    if save_cleaned:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df = pd.concat([X, y], axis=1)
        cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    return X, y


if __name__ == "__main__":
    X, y = get_feature_and_target_matrices()
    print(f"Data loaded successfully! Features shape: {X.shape}, Target distribution:\n{y.value_counts()}")
