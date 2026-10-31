# CKD Explainable AI: Chronic Kidney Disease Prediction using Explainable Artificial Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-15%20Passed-success.svg)]()

A comprehensive, production-grade clinical decision support system that predicts **Chronic Kidney Disease (CKD)** and provides **interpretable, feature-level clinical explanations** using **SHAP (SHapley Additive exPlanations)**.

Built from scratch as an advanced college capstone project, improving upon baseline academic implementations through **leakage-free preprocessing**, **multi-model benchmarking**, **interactive web dashboards**, **digital lab document OCR with clinician verification**, and **automated PDF clinical report generation**.

---

## 📌 Table of Contents
1. [Overview & Clinical Motivation](#-overview--clinical-motivation)
2. [Project Architecture](#-project-architecture)
3. [Dataset & Clinical Biomarkers](#-dataset--clinical-biomarkers)
4. [Machine Learning Methodology](#-machine-learning-methodology)
5. [Model Performance Benchmarks](#-model-performance-benchmarks)
6. [Explainable AI (SHAP) Integration](#-explainable-ai-shap-integration)
7. [Streamlit Dashboard Features](#-streamlit-dashboard-features)
8. [Lab Report OCR with Human-in-the-Loop](#-lab-report-ocr-with-human-in-the-loop)
9. [Automated PDF Diagnostic Report Generation](#-automated-pdf-diagnostic-report-generation)
10. [Folder Structure](#-folder-structure)
11. [Installation & Setup](#-installation--setup)
12. [Running the Application](#-running-the-application)
13. [Running Automated Tests](#-running-automated-tests)
14. [Limitations & Disclaimer](#-limitations--disclaimer)

---

## 💡 Overview & Clinical Motivation

**Chronic Kidney Disease (CKD)** is a progressive loss in renal function affecting over 10% of the world's population. In early stages, patients are largely asymptomatic, while late stages lead to end-stage renal disease (ESRD) requiring dialysis or kidney transplantation.

Traditional machine learning classifiers frequently operate as "black boxes"—rendering opaque predictions without explaining the underlying biological rationale. In clinical medicine, opacity creates mistrust: clinicians must know **which specific laboratory abnormalities** drove an algorithm's risk score.

This system solves the opacity problem by coupling multi-model supervised classification (**Random Forest, XGBoost, Decision Tree, Logistic Regression**) with **SHAP**, offering:
- **Global Population Insights:** Explaining dataset-wide biomarker risk dynamics.
- **Local Patient Attribution:** Breaking down exactly how each biomarker pushed an individual patient's score relative to baseline risk.

---

## 🏗️ Project Architecture

```
[Raw CKD Dataset (400 cases, 24 features)]
                      │
                      ▼
       [Data Cleaning & Anomaly Correction]
        (Removes tabs, mixed types, handles '?')
                      │
                      ▼
       [Stratified Train-Test Split (80/20)]
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   [Training Set (320)]     [Test Set (80)]
         │                         │
         ▼                         │
   [ColumnTransformer]             │
   • Median Imputer + Scaler       │
   • Mode Imputer + OneHot         │
         │                         │
         ▼                         ▼
  [Model Training Suite] ──> [Model Evaluation]
  • Random Forest            • Accuracy, F1, ROC-AUC
  • Logistic Regression      • Confusion Matrices
  • Decision Tree            • Precision, Sensitivity
  • XGBoost
         │
         ▼
  [SHAP Explainability Engine]
  • TreeExplainer / LinearExplainer
  • Summary Beeswarm Plot
  • Patient Waterfall & Force Plots
         │
         ▼
┌──────────────────────────────────────────────┐
│          Interactive Application Layer       │
├──────────────────────┬───────────────────────┤
│ Streamlit Dashboard  │ ReportLab PDF Engine  │
│ • Interactive EDA    │ • Patient Metrics     │
│ • Model Scorecard    │ • Reference Ranges    │
│ • Live Patient XAI   │ • Embedded Waterfall  │
│ • Batch Screening    │ • Physician Notes     │
│ • Lab OCR + Verify   │                       │
└──────────────────────┴───────────────────────┘
```

---

## 🔬 Dataset & Clinical Biomarkers

The system utilizes the standard UCI Chronic Kidney Disease benchmark dataset containing 400 patient records across 24 clinical parameters and 1 target label (`class`: `ckd` vs `notckd`).

### Monitored Biomarkers:
| Parameter | Full Name | Medical Importance | Normal Interval |
| :--- | :--- | :--- | :--- |
| `sc` | Serum Creatinine | Primary index of glomerular filtration | 0.6 - 1.2 mg/dL |
| `bu` | Blood Urea | Nitrogen waste clearance efficiency | 15 - 45 mg/dL |
| `hemo` | Hemoglobin | Detects renal erythropoietin deficiency | 12.0 - 16.5 g/dL |
| `al` | Albumin (Urine) | Indicates glomerular capillary damage | Grade 0 (Negative) |
| `sg` | Specific Gravity | Urine concentration capability of tubules | 1.010 - 1.025 |
| `bgr` | Blood Glucose | Identifies diabetic nephropathy risk | 70 - 140 mg/dL |
| `pcv` | Packed Cell Volume | Hematocrit percentage | 36 - 50 % |
| `sod` | Serum Sodium | Tubule fluid balance & electrolyte stability | 135 - 145 mEq/L |
| `pot` | Serum Potassium | Cardiac risk from hyperkalemia | 3.5 - 5.0 mEq/L |
| `wbcc` | White Blood Cells | Renal/urinary tract inflammation marker | 4,500 - 11,000 /μL |
| `rbcc` | Red Blood Cells | Erythrocyte count | 4.2 - 5.9 M/μL |
| `bp` | Blood Pressure | Systemic hypertension & vascular damage | 60 - 80 mm Hg |

---

## 🧪 Machine Learning Methodology

### 1. Leakage-Free Preprocessing Pipeline
Unlike naive implementations that fit scalers before splitting, this project enforces strict scikit-learn pipelines:
- **Numerical Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
- **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- All imputers and scalers are fitted **strictly on the 80% training split**.

### 2. Multi-Model Suite
Four distinct classifier paradigms are trained and benchmarked:
1. **Random Forest Classifier:** Ensemble bagging of decorrelated decision trees.
2. **XGBoost Classifier:** Gradient-boosted decision trees optimizing binary logloss.
3. **Decision Tree Classifier:** Single interpretable decision tree with bounded depth.
4. **Logistic Regression:** L2-regularized linear baseline with probability estimation.

---

## 📊 Model Performance Benchmarks

Evaluated on a strictly held-out 20% test split (80 samples: 50 CKD positive, 30 healthy):

| Algorithm | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | **98.75%** | **1.0000** | **0.9800** | **1.0000** | **0.9899** | **1.0000** |
| **Decision Tree** | **97.50%** | **0.9800** | **0.9800** | **0.9667** | **0.9800** | **0.9893** |

*Note: High classification scores are characteristic of the UCI CKD dataset due to the high diagnostic separation provided by serum creatinine and albumin in combination with hemoglobin.*

---

## 🧠 Explainable AI (SHAP) Integration

The project integrates `shap.TreeExplainer` and `shap.LinearExplainer` to provide mathematically grounded feature attribution based on cooperative game theory (Shapley values).

### 1. Global Explainability
- **SHAP Summary Beeswarm Plot:** Visualizes how high vs low values of each biomarker shift risk across the cohort.
- **Mean Absolute SHAP Ranking:** Top global determinants identified:
  1. `Serum Creatinine (sc)`
  2. `Hemoglobin (hemo)`
  3. `Albumin in Urine (al)`
  4. `Packed Cell Volume (pcv)`
  5. `Specific Gravity (sg)`

### 2. Local Patient-Specific Attribution
- **Waterfall Plots:** Shows step-by-step how an individual patient's risk deviates from baseline expected log-odds to the final prediction.
- **Risk Direction:** Colors indicate whether a factor increased risk (red) or decreased risk (green).

---

## 🖥️ Streamlit Dashboard Features

The web interface is structured into 5 dedicated pages:

- **🏠 Main Landing (`app/app.py`):** Executive overview, clinical problem statement, technology stack, and navigation guide.
- **📊 1. Exploratory Data Analysis (`pages/1_📊_Exploratory_Data_Analysis.py`):** Interactive data tables, missing rate graphs, biomarker distribution boxplots, and correlation heatmaps.
- **🧪 2. Model Comparison (`pages/2_🧪_Model_Comparison.py`):** Side-by-side performance scorecard, multi-metric bar chart, overlaid ROC curves, confusion matrices, and dynamic retraining.
- **🩺 3. Patient Diagnosis & XAI (`pages/3_🩺_Patient_Diagnosis_&_XAI.py`):** Patient entry form, preset sample cases, Plotly risk gauge, interactive SHAP waterfall chart, top risk table, and 1-click clinical PDF export.
- **📄 4. Batch Screening (`pages/4_📄_Batch_Prediction_&_Reports.py`):** CSV upload for screening cohorts of hundreds of patients, risk tier categorization (High/Moderate/Low), and annotated CSV download.
- **📑 5. Lab Report OCR Scanner (`pages/5_📑_Lab_Report_OCR_Scanner.py`):** Digital PDF / image extraction, automated biomarker matching, and human-in-the-loop verification form.

---

## 📑 Lab Report OCR with Human-in-the-Loop

Medical AI systems should never execute automated decisions on unverified text extractions. This project implements **Human-in-the-Loop OCR**:
1. Ingests electronic laboratory reports (`.pdf`) or scans (`.png`, `.jpg`).
2. Applies clinical regular expressions to extract parameters (Creatinine, Urea, Hemoglobin, Glucose, Albumin, etc.).
3. Populates an interactive verification form where the clinician reviews, corrects any misread values, and inputs clinical history (Hypertension, Diabetes).
4. Upon confirmation ("Verify Values & Run AI Diagnosis"), the validated data is passed to the ML and SHAP engine.

---

## 📄 Automated PDF Diagnostic Report Generation

Built with **ReportLab**, the application generates downloadable, print-ready diagnostic summaries:
- **Header:** Clinical facility metadata, patient ID, date, time, and algorithm used.
- **Diagnostic Box:** Color-coded status (High Risk vs Low Risk) and exact probability percentage.
- **Laboratory Panel Table:** Two-column table of 24 biomarkers comparing observed patient values against established reference ranges.
- **SHAP Attribution Breakdown:** Ranked table of top factors influencing the model's recommendation.
- **Embedded Visualizations:** Embedded SHAP waterfall figure.
- **Medical Disclaimer & Signature Block:** Reviewing nephrologist signature line.

---

## 📂 Folder Structure

```
CKD_Explainable_AI/
├── app/
│   ├── app.py                             # Streamlit main entry point
│   ├── components/
│   │   ├── inputs.py                      # Patient clinical input form
│   │   ├── visualizers.py                 # Plotly gauges and metric charts
│   │   └── shap_plots.py                  # Streamlit SHAP plot wrappers
│   └── pages/
│       ├── 1_📊_Exploratory_Data_Analysis.py
│       ├── 2_🧪_Model_Comparison.py
│       ├── 3_🩺_Patient_Diagnosis_&_XAI.py
│       ├── 4_📄_Batch_Prediction_&_Reports.py
│       └── 5_📑_Lab_Report_OCR_Scanner.py
│
├── assets/
│   └── css/custom.css                     # Medical dashboard CSS styling
│
├── data/
│   ├── raw/chronic_kidney_disease.csv     # Cleaned UCI CKD raw dataset
│   ├── processed/cleaned_ckd.csv          # Preprocessed data artifact
│   └── sample_patients/                   # Presets & test files
│       ├── healthy_patient.json
│       ├── early_stage_ckd_patient.json
│       ├── severe_ckd_patient.json
│       ├── batch_test_patients.csv
│       └── sample_lab_report.pdf
│
├── models/
│   ├── saved_models/                      # Serialized .joblib model weights
│   ├── scaler/preprocessor.joblib         # Fitted ColumnTransformer pipeline
│   └── metrics/model_comparison.json      # Benchmark evaluation scores
│
├── notebooks/
│   ├── 01_eda_and_data_analysis.ipynb
│   ├── 02_model_training_and_comparison.ipynb
│   └── 03_shap_explainability_experiments.ipynb
│
├── reports/
│   ├── figures/                           # Saved ROC & SHAP figures
│   └── generated_reports/                 # Generated clinical PDF reports
│
├── src/
│   ├── config.py                          # Global paths, features & ranges
│   ├── data_loader.py                     # Data loading & anomaly cleaning
│   ├── preprocessor.py                    # Scikit-learn ColumnTransformer
│   ├── train.py                           # Training & serialization pipeline
│   ├── evaluate.py                        # Metrics & ROC curve calculation
│   ├── explainability.py                  # SHAP TreeExplainer & waterfall
│   ├── ocr_extractor.py                   # PDF & Image regex OCR parser
│   ├── pdf_generator.py                   # ReportLab PDF generation engine
│   └── generate_shap_assets.py            # Asset pre-computation script
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_models.py
│   ├── test_explainability.py
│   └── test_pdf_and_ocr.py
│
├── requirements.txt                       # Explicit dependencies
└── README.md                              # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- Git

### 1. Clone or Open Workspace
```bash
cd CKD_Explainable_AI
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit interactive dashboard with:

```bash
streamlit run app/app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:

```bash
pytest tests/ -v
```

All 15 tests covering data loading, preprocessing, model inference, SHAP calculations, and PDF generation will run and validate the codebase.

---

## ⚠️ Limitations & Clinical Disclaimer

1. **Decision Support Only:** This software is an academic engineering project designed strictly for educational and research demonstration purposes. It **does not constitute medical advice**, clinical diagnosis, or treatment recommendations.
2. **Dataset Demographics:** The models are trained on the public UCI Chronic Kidney Disease dataset (400 cases). External validation across diverse clinical populations, pediatric cohorts, and multicenter trials is required before real-world clinical application.
3. **OCR Constraints:** Scanned image OCR relies on document image quality, lighting, and resolution. Manual verification of extracted laboratory parameters by a licensed clinician remains mandatory.
