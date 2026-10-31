"""
Tests for Clinical PDF Generation and Lab Report OCR Parser
"""
import pytest
from pathlib import Path
from src.config import GENERATED_REPORTS_DIR, SAMPLE_PATIENTS_DIR
from src.pdf_generator import generate_patient_pdf_report
from src.ocr_extractor import process_lab_document, parse_clinical_values_from_text


def test_pdf_report_generation(tmp_path):
    sample_data = {
        "age": 55, "bp": 85, "sg": 1.015, "al": 2, "su": 0,
        "rbc": "normal", "pc": "abnormal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 135, "bu": 50, "sc": 2.1, "sod": 136, "pot": 4.6,
        "hemo": 10.4, "pcv": 32, "wbcc": 8200, "rbcc": 3.7,
        "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    }

    dummy_pred = {
        "is_ckd": True,
        "prediction_probability": 0.88,
        "top_contributions": [
            {"feature": "sc", "raw_value": 2.1, "shap_value": 0.35, "impact": "Increases CKD Risk"},
            {"feature": "hemo", "raw_value": 10.4, "shap_value": -0.20, "impact": "Increases CKD Risk"},
        ],
        "waterfall_figure": None
    }

    out_file = tmp_path / "test_report.pdf"
    res_path = generate_patient_pdf_report(
        patient_data=sample_data,
        prediction_result=dummy_pred,
        model_name="Random Forest",
        output_path=out_file
    )

    assert res_path.exists()
    assert res_path.stat().st_size > 1000, "Generated PDF must be non-trivial size."
    with open(res_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-", "File must have valid PDF magic bytes."


def test_ocr_text_parsing():
    sample_text = """
    Metropolitan Clinical Laboratories
    Patient: Test Subject
    Age: 62   Blood Pressure: 130/85
    Serum Creatinine: 1.8 mg/dL
    Blood Urea: 56 mg/dL
    Hemoglobin: 11.2 g/dL
    Random Blood Sugar: 160 mg/dL
    Specific Gravity: 1.015
    Albumin: 2
    """
    extracted = parse_clinical_values_from_text(sample_text)
    assert extracted.get("sc") == 1.8
    assert extracted.get("bu") == 56.0
    assert extracted.get("hemo") == 11.2
    assert extracted.get("bgr") == 160.0
    assert extracted.get("sg") == 1.015
    assert extracted.get("al") == 2.0
    assert extracted.get("age") == 62.0


def test_sample_lab_pdf_ocr():
    sample_pdf = SAMPLE_PATIENTS_DIR / "sample_lab_report.pdf"
    assert sample_pdf.exists(), "Sample lab report PDF must exist."
    with open(sample_pdf, "rb") as f:
        bytes_data = f.read()

    result = process_lab_document(bytes_data, "sample_lab_report.pdf")
    assert result["match_count"] >= 8, "Expected at least 8 parameters to be parsed from sample PDF."
    features = result["extracted_features"]
    assert features.get("sc") == 2.4
    assert features.get("hemo") == 9.8
    assert features.get("al") == 2.0
