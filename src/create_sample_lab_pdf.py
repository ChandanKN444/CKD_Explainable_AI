"""
Generate a sample laboratory blood test PDF for OCR testing.
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from src.config import SAMPLE_PATIENTS_DIR

def build_sample_lab_pdf():
    SAMPLE_PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = SAMPLE_PATIENTS_DIR / "sample_lab_report.pdf"

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>METROPOLITAN CLINICAL LABORATORIES</b>", styles["Heading1"]))
    story.append(Paragraph("Comprehensive Metabolic & Renal Function Panel", styles["Heading3"]))
    story.append(Spacer(1, 10))

    meta_text = "<b>Patient:</b> John Doe &nbsp;&nbsp;|&nbsp;&nbsp; <b>Age:</b> 54 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Blood Pressure:</b> 140/90 mmHg"
    story.append(Paragraph(meta_text, styles["Normal"]))
    story.append(Spacer(1, 15))

    data = [
        ["Test Description", "Observed Result", "Reference Units", "Reference Range"],
        ["Serum Creatinine", "2.4", "mg/dL", "0.6 - 1.2"],
        ["Blood Urea Nitrogen", "68", "mg/dL", "15 - 45"],
        ["Blood Glucose Random", "185", "mg/dL", "70 - 140"],
        ["Serum Sodium", "132", "mEq/L", "135 - 145"],
        ["Serum Potassium", "5.4", "mEq/L", "3.5 - 5.0"],
        ["Hemoglobin", "9.8", "g/dL", "12.0 - 16.5"],
        ["Packed Cell Volume (PCV)", "30", "%", "36 - 50"],
        ["Total Leukocyte Count (WBC)", "10,400", "cells/cumm", "4500 - 11000"],
        ["Red Blood Cell Count (RBC)", "3.6", "million/cmm", "4.2 - 5.9"],
        ["Urine Specific Gravity", "1.015", "ratio", "1.010 - 1.025"],
        ["Urine Albumin", "2", "scale 0-5", "0 (Negative)"]
    ]

    t = Table(data, colWidths=[200, 100, 100, 120])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    doc.build(story)
    print(f"Sample Lab Report PDF created at: {pdf_path}")

if __name__ == "__main__":
    build_sample_lab_pdf()
