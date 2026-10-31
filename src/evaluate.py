"""
Model Evaluation and Performance Benchmarking Module
Computes Accuracy, Precision, Recall, Specificity, F1, and ROC-AUC scores.
Generates confusion matrix and ROC curve visualizations.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

try:
    from src.config import METRICS_DIR, FIGURES_DIR
except ImportError:
    from config import METRICS_DIR, FIGURES_DIR


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes standard binary classification metrics including specificity and ROC-AUC.
    """
    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = None
    fpr_list, tpr_list, thresholds_list = [], [], []
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
            fpr, tpr, thresh = roc_curve(y_true, y_proba)
            fpr_list = [float(x) for x in fpr]
            tpr_list = [float(x) for x in tpr]
            thresholds_list = [1.0 if np.isinf(x) else float(x) for x in thresh]
        except Exception:
            roc_auc = None

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "roc_curve": {
            "fpr": fpr_list,
            "tpr": tpr_list,
            "thresholds": thresholds_list
        }
    }

    return metrics


def save_benchmark_metrics(
    benchmarks: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None
) -> Path:
    """Save multi-model benchmark results to a formatted JSON file."""
    path = Path(output_path) if output_path else (METRICS_DIR / "model_comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=4)
    return path


def plot_and_save_roc_curves(
    benchmarks: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None
) -> Path:
    """Plot overlaid ROC curves for all models and save to figures directory."""
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance (AUC = 0.50)")

    for name, data in benchmarks.items():
        roc = data.get("roc_curve", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auc = data.get("roc_auc", None)
        if fpr and tpr:
            auc_str = f"{auc:.3f}" if auc is not None else "N/A"
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_str})", linewidth=2)

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curves Comparison - CKD Prediction", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = Path(save_path) if save_path else (FIGURES_DIR / "roc_curves_comparison.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_and_save_confusion_matrices(
    benchmarks: Dict[str, Dict[str, Any]],
    save_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Generate and save confusion matrix heatmaps for each model."""
    out_dir = Path(save_dir) if save_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    for name, data in benchmarks.items():
        cm_dict = data.get("confusion_matrix", {})
        cm = np.array([
            [cm_dict.get("tn", 0), cm_dict.get("fp", 0)],
            [cm_dict.get("fn", 0), cm_dict.get("tp", 0)]
        ])

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted NotCKD", "Predicted CKD"],
            yticklabels=["Actual NotCKD", "Actual CKD"]
        )
        plt.title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        clean_name = name.lower().replace(" ", "_")
        file_path = out_dir / f"confusion_matrix_{clean_name}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()
        saved_paths[name] = file_path

    return saved_paths
