"""
Interactive Visualizations Component using Plotly
Provides clinical risk gauge meter, biomarker radar charts, and comparative metric visualizations.
"""
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_risk_gauge(probability: float) -> go.Figure:
    """
    Renders an interactive semicircular clinical gauge meter indicating CKD risk probability.
    """
    pct = probability * 100.0

    if pct < 35.0:
        bar_color = "#2E7D32"   # Green (Low Risk)
        status_label = "LOW RISK"
    elif pct < 65.0:
        bar_color = "#F57C00"   # Orange (Borderline / Moderate Risk)
        status_label = "MODERATE RISK"
    else:
        bar_color = "#D32F2F"   # Red (High Risk)
        status_label = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": bar_color}},
        title={"text": f"<b>CKD Probability: {status_label}</b>", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#718096"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 35], "color": "#E8F5E9"},
                {"range": [35, 65], "color": "#FFF3E0"},
                {"range": [65, 100], "color": "#FFEBEE"}
            ],
            "threshold": {
                "line": {"color": "#B71C1C", "width": 4},
                "thickness": 0.8,
                "value": 50.0
            }
        }
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"}
    )
    return fig


def create_model_comparison_bar_chart(benchmarks: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Renders a grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC across models.
    """
    records = []
    metrics_to_plot = ["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"]

    for model_name, metrics in benchmarks.items():
        for m in metrics_to_plot:
            val = metrics.get(m)
            if val is not None:
                records.append({
                    "Model": model_name,
                    "Metric": m.replace("_", " ").title(),
                    "Score": val
                })

    df = pd.DataFrame(records)
    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".2%",
        title="Cross-Model Performance Benchmarks",
        color_discrete_sequence=["#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]
    )
    fig.update_layout(
        yaxis_range=[0.70, 1.02],
        xaxis_title="",
        yaxis_title="Score",
        legend_title="Model",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_shap_horizontal_bar(top_contributions: List[Dict[str, Any]], max_display: int = 8) -> go.Figure:
    """
    Renders an interactive horizontal bar chart of top SHAP feature impacts for an individual patient.
    """
    items = top_contributions[:max_display][::-1]  # Reverse for ascending order in horizontal bar

    features = [it["feature"] for it in items]
    shap_vals = [it["shap_value"] for it in items]
    colors_list = ["#E53E3E" if v > 0 else "#38A169" for v in shap_vals]
    text_labels = [f"{v:+.3f}" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker_color=colors_list,
        text=text_labels,
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>Local SHAP Feature Contributions (Impact on Risk)</b>",
        xaxis_title="SHAP Value (Positive = Risk Increase, Negative = Risk Decrease)",
        yaxis_title="",
        height=320,
        margin=dict(l=40, r=40, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
