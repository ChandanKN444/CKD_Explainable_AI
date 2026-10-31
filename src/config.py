"""
Global Configuration and Clinical Metadata for CKD Explainable AI
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "chronic_kidney_disease.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_ckd.csv"
SAMPLE_PATIENTS_DIR = DATA_DIR / "sample_patients"

MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
SCALER_DIR = MODELS_DIR / "scaler"
METRICS_DIR = MODELS_DIR / "metrics"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated_reports"

# Feature Names & Types
NUMERICAL_FEATURES = [
    "age",    # Age (years)
    "bp",     # Blood Pressure (mm/Hg)
    "sg",     # Specific Gravity (1.005, 1.010, 1.015, 1.020, 1.025)
    "al",     # Albumin (0, 1, 2, 3, 4, 5)
    "su",     # Sugar (0, 1, 2, 3, 4, 5)
    "bgr",    # Blood Glucose Random (mgs/dl)
    "bu",     # Blood Urea (mgs/dl)
    "sc",     # Serum Creatinine (mgs/dl)
    "sod",    # Sodium (mEq/L)
    "pot",    # Potassium (mEq/L)
    "hemo",   # Hemoglobin (gms)
    "pcv",    # Packed Cell Volume (%)
    "wbcc",   # White Blood Cell Count (cells/cumm)
    "rbcc"    # Red Blood Cell Count (millions/cmm)
]

CATEGORICAL_FEATURES = [
    "rbc",    # Red Blood Cells: normal, abnormal
    "pc",     # Pus Cell: normal, abnormal
    "pcc",    # Pus Cell Clumps: present, notpresent
    "ba",     # Bacteria: present, notpresent
    "htn",    # Hypertension: yes, no
    "dm",     # Diabetes Mellitus: yes, no
    "cad",    # Coronary Artery Disease: yes, no
    "appet",  # Appetite: good, poor
    "pe",     # Pedal Edema: yes, no
    "ane"     # Anemia: yes, no
]

TARGET_COLUMN = "class"

# Human-Readable Descriptions
FEATURE_METADATA = {
    "age": {"name": "Age", "unit": "years", "normal_range": "18 - 80", "min": 1.0, "max": 100.0, "default": 55.0},
    "bp": {"name": "Blood Pressure", "unit": "mm Hg", "normal_range": "60 - 80", "min": 50.0, "max": 180.0, "default": 80.0},
    "sg": {"name": "Specific Gravity", "unit": "ratio", "normal_range": "1.010 - 1.025", "min": 1.005, "max": 1.025, "default": 1.020},
    "al": {"name": "Albumin", "unit": "grade (0-5)", "normal_range": "0", "min": 0.0, "max": 5.0, "default": 0.0},
    "su": {"name": "Sugar", "unit": "grade (0-5)", "normal_range": "0", "min": 0.0, "max": 5.0, "default": 0.0},
    "rbc": {"name": "Red Blood Cells", "options": ["normal", "abnormal"], "default": "normal"},
    "pc": {"name": "Pus Cell", "options": ["normal", "abnormal"], "default": "normal"},
    "pcc": {"name": "Pus Cell Clumps", "options": ["notpresent", "present"], "default": "notpresent"},
    "ba": {"name": "Bacteria", "options": ["notpresent", "present"], "default": "notpresent"},
    "bgr": {"name": "Blood Glucose Random", "unit": "mg/dL", "normal_range": "70 - 140", "min": 50.0, "max": 500.0, "default": 110.0},
    "bu": {"name": "Blood Urea", "unit": "mg/dL", "normal_range": "15 - 45", "min": 10.0, "max": 400.0, "default": 35.0},
    "sc": {"name": "Serum Creatinine", "unit": "mg/dL", "normal_range": "0.6 - 1.2", "min": 0.4, "max": 20.0, "default": 1.0},
    "sod": {"name": "Serum Sodium", "unit": "mEq/L", "normal_range": "135 - 145", "min": 100.0, "max": 170.0, "default": 138.0},
    "pot": {"name": "Serum Potassium", "unit": "mEq/L", "normal_range": "3.5 - 5.0", "min": 2.0, "max": 10.0, "default": 4.2},
    "hemo": {"name": "Hemoglobin", "unit": "g/dL", "normal_range": "12.0 - 16.5", "min": 3.0, "max": 20.0, "default": 14.5},
    "pcv": {"name": "Packed Cell Volume (PCV)", "unit": "%", "normal_range": "36 - 50", "min": 10.0, "max": 60.0, "default": 42.0},
    "wbcc": {"name": "White Blood Cell Count", "unit": "cells/μL", "normal_range": "4,500 - 11,000", "min": 2000.0, "max": 30000.0, "default": 7500.0},
    "rbcc": {"name": "Red Blood Cell Count", "unit": "million/μL", "normal_range": "4.2 - 5.9", "min": 2.0, "max": 8.0, "default": 4.8},
    "htn": {"name": "Hypertension", "options": ["no", "yes"], "default": "no"},
    "dm": {"name": "Diabetes Mellitus", "options": ["no", "yes"], "default": "no"},
    "cad": {"name": "Coronary Artery Disease", "options": ["no", "yes"], "default": "no"},
    "appet": {"name": "Appetite", "options": ["good", "poor"], "default": "good"},
    "pe": {"name": "Pedal Edema", "options": ["no", "yes"], "default": "no"},
    "ane": {"name": "Anemia", "options": ["no", "yes"], "default": "no"}
}

# Model Candidates
MODELS = {
    "Random Forest": {
        "class_name": "RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 3,
            "random_state": 42
        }
    },
    "Logistic Regression": {
        "class_name": "LogisticRegression",
        "params": {
            "max_iter": 1000,
            "C": 1.0,
            "random_state": 42
        }
    },
    "Decision Tree": {
        "class_name": "DecisionTreeClassifier",
        "params": {
            "max_depth": 6,
            "min_samples_split": 4,
            "random_state": 42
        }
    },
    "XGBoost": {
        "class_name": "XGBClassifier",
        "params": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "eval_metric": "logloss",
            "random_state": 42
        }
    }
}

# Random Seed for Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20
