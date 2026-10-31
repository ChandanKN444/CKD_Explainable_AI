"""
Data Preprocessing Pipeline Module for CKD Explainable AI
Builds a leakage-free Scikit-Learn ColumnTransformer pipeline.
"""
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    from src.config import (
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        SCALER_DIR
    )
except ImportError:
    from config import (
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        SCALER_DIR
    )


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    1. Imputes numerical missing values using median and scales using StandardScaler.
    2. Imputes categorical missing values with most_frequent and encodes using OneHotEncoder.
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract clean feature names from a fitted ColumnTransformer."""
    try:
        # scikit-learn >= 1.0 feature names retrieval
        raw_names = preprocessor.get_feature_names_out()
        cleaned = []
        for name in raw_names:
            # Strip transformer prefixes like 'num__' and 'cat__'
            if "__" in name:
                cleaned.append(name.split("__", 1)[1])
            else:
                cleaned.append(name)
        return cleaned
    except Exception:
        # Fallback manual reconstruction
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        return NUMERICAL_FEATURES + cat_names


def fit_and_save_preprocessor(
    X_train: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Tuple[ColumnTransformer, np.ndarray, List[str]]:
    """
    Fit preprocessor strictly on X_train to prevent data leakage,
    serialize the artifact to disk, and return transformed array + feature names.
    """
    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = get_feature_names(preprocessor)

    target_path = Path(save_path) if save_path else (SCALER_DIR / "preprocessor.joblib")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, target_path)

    return preprocessor, X_train_transformed, feature_names


def load_preprocessor(load_path: Optional[Path] = None) -> ColumnTransformer:
    """Load a previously serialized preprocessor pipeline."""
    target_path = Path(load_path) if load_path else (SCALER_DIR / "preprocessor.joblib")
    if not target_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at: {target_path}")
    return joblib.load(target_path)


def transform_data(
    X: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None
) -> Tuple[np.ndarray, List[str]]:
    """Transform input features using an existing or loaded preprocessor."""
    if preprocessor is None:
        preprocessor = load_preprocessor()
    transformed = preprocessor.transform(X)
    feature_names = get_feature_names(preprocessor)
    return transformed, feature_names
