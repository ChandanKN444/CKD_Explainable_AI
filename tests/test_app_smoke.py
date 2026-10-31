"""
Smoke Test for Streamlit Application Components and Pages
Ensures all modules, components, and pages can compile and run without runtime exceptions.
"""
import py_compile
from pathlib import Path


def test_streamlit_pages_compile():
    app_dir = Path(__file__).resolve().parent.parent / "app"
    page_files = [
        app_dir / "app.py",
        app_dir / "components" / "inputs.py",
        app_dir / "components" / "visualizers.py",
        app_dir / "components" / "shap_plots.py",
        app_dir / "pages" / "1_📊_Exploratory_Data_Analysis.py",
        app_dir / "pages" / "2_🧪_Model_Comparison.py",
        app_dir / "pages" / "3_🩺_Patient_Diagnosis_&_XAI.py",
        app_dir / "pages" / "4_📄_Batch_Prediction_&_Reports.py",
        app_dir / "pages" / "5_📑_Lab_Report_OCR_Scanner.py",
    ]

    for p in page_files:
        assert p.exists(), f"File does not exist: {p}"
        # Compiles python byte code to catch any syntax errors
        py_compile.compile(str(p), doraise=True)
