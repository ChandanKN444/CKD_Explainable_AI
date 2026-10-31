"""
Streamlit SHAP Visualization Helper Component
Cleanly renders SHAP waterfall, summary, and force plots in Streamlit.
"""
import streamlit as st
import matplotlib.pyplot as plt
import shap


def render_shap_waterfall_plot(waterfall_figure):
    """Safely display a pre-generated SHAP waterfall figure in Streamlit."""
    if waterfall_figure is not None:
        st.pyplot(waterfall_figure, use_container_width=True)
        plt.close(waterfall_figure)


def render_shap_summary_plot(shap_values, X_sample, feature_names):
    """Dynamically computes and renders a SHAP summary beeswarm plot in Streamlit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
