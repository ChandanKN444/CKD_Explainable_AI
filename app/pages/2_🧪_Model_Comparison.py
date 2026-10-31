"""
Model Comparison and Evaluation Benchmarking Page
Displays comparative metrics, ROC-AUC curves, and confusion matrices across all ML models.
"""
import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.config import METRICS_DIR, FIGURES_DIR
from src.train import train_and_evaluate_all_models
from app.components.visualizers import create_model_comparison_bar_chart

st.set_page_config(page_title="Model Benchmarks | CKD Explainable AI", page_icon="🧪", layout="wide")

st.title("🧪 Machine Learning Model Comparison & Performance Benchmarks")
st.markdown("Compare predictive accuracy, sensitivity (recall), specificity, F1-score, and ROC-AUC across candidate algorithms.")

metrics_file = METRICS_DIR / "model_comparison.json"

col_header, col_retrain = st.columns([4, 1])
with col_header:
    st.markdown("All models are evaluated on a held-out test split (20%) using **leakage-free preprocessing**.")
with col_retrain:
    if st.button("🔄 Retrain All Models", use_container_width=True):
        with st.spinner("Retraining Random Forest, Logistic Regression, Decision Tree, and XGBoost..."):
            try:
                benchmarks, best = train_and_evaluate_all_models()
                st.success(f"Models retrained! Best model: **{best}**")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

if not metrics_file.exists():
    st.warning("No pre-trained model benchmark file found. Retraining now...")
    with st.spinner("Training models for first time..."):
        try:
            train_and_evaluate_all_models()
            st.rerun()
        except Exception as e:
            st.error(f"Initial training error: {e}")
            st.stop()

with open(metrics_file, "r") as f:
    benchmarks = json.load(f)

# Top KPI Highlights
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
best_acc = max(benchmarks.items(), key=lambda x: x[1]["accuracy"])
best_f1 = max(benchmarks.items(), key=lambda x: x[1]["f1_score"])
best_rec = max(benchmarks.items(), key=lambda x: x[1]["recall"])
best_auc = max(benchmarks.items(), key=lambda x: x[1]["roc_auc"] or 0)

kpi1.metric("Highest Accuracy", f"{best_acc[1]['accuracy']*100:.1f}%", f"{best_acc[0]}")
kpi2.metric("Highest F1-Score", f"{best_f1[1]['f1_score']:.4f}", f"{best_f1[0]}")
kpi3.metric("Highest Sensitivity (Recall)", f"{best_rec[1]['recall']*100:.1f}%", f"{best_rec[0]}")
kpi4.metric("Highest ROC-AUC", f"{best_auc[1]['roc_auc']:.3f}", f"{best_auc[0]}")

st.divider()

# Comparative Table
st.subheader("📊 Performance Scorecard")
table_data = []
for model_name, m in benchmarks.items():
    table_data.append({
        "Model": model_name,
        "Accuracy": f"{m['accuracy']*100:.2f}%",
        "Precision": f"{m['precision']:.4f}",
        "Recall (Sensitivity)": f"{m['recall']:.4f}",
        "Specificity": f"{m['specificity']:.4f}",
        "F1-Score": f"{m['f1_score']:.4f}",
        "ROC-AUC": f"{m['roc_auc']:.4f}" if m['roc_auc'] else "N/A"
    })

scorecard_df = pd.DataFrame(table_data)
st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

# Comparative Bar Chart
st.subheader("📈 Multi-Metric Graphical Comparison")
fig_bar = create_model_comparison_bar_chart(benchmarks)
st.plotly_chart(fig_bar, use_container_width=True)

# Visualizations: ROC Curves & Confusion Matrices
st.divider()
col_roc, col_cm = st.columns([1, 1])

with col_roc:
    st.subheader("🎯 Overlaid ROC Curves")
    roc_img = FIGURES_DIR / "roc_curves_comparison.png"
    if roc_img.exists():
        st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
    else:
        st.info("ROC Curve image will be displayed after model training.")

with col_cm:
    st.subheader("🔲 Confusion Matrices")
    cm_model = st.selectbox("Select Model to View Confusion Matrix:", options=list(benchmarks.keys()))
    cm_slug = cm_model.lower().replace(" ", "_")
    cm_img = FIGURES_DIR / f"confusion_matrix_{cm_slug}.png"
    if cm_img.exists():
        st.image(str(cm_img), caption=f"Confusion Matrix: {cm_model}", use_container_width=True)
    else:
        st.info(f"Confusion matrix for {cm_model} not found.")

st.caption("Clinical Insight: In screening, High Sensitivity (Recall) is paramount to minimize False Negatives (missing a CKD patient).")
