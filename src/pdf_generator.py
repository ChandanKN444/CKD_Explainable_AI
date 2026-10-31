"""
Clinical PDF Diagnostic Report Generator Module
Creates an executive clinical report containing patient details, test results vs normal ranges,
AI risk score, SHAP explanation breakdown, and embedded waterfall visualization using ReportLab.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import io

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable
)
import matplotlib.pyplot as plt

try:
    from src.config import GENERATED_REPORTS_DIR, FEATURE_METADATA
except ImportError:
    from config import GENERATED_REPORTS_DIR, FEATURE_METADATA


def generate_patient_pdf_report(
    patient_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    model_name: str,
    patient_id: str = "PT-89421",
    patient_name: str = "Anonymous Patient",
    output_path: Optional[Path] = None
) -> Path:
    """
    Builds a clinical PDF summary report using ReportLab.
    """
    GENERATED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = GENERATED_REPORTS_DIR / f"CKD_Report_{patient_id}_{timestamp_slug}.pdf"
    else:
        output_path = Path(output_path)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=10
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )
    bold_style = ParagraphStyle(
        "ReportBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    # 1. Header & Title
    story.append(Paragraph("NEPHROLOGY CLINICAL AI ASSESSMENT REPORT", title_style))
    story.append(Paragraph("Chronic Kidney Disease (CKD) Diagnostic Screening & Explainable AI (SHAP)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # 2. Metadata Grid
    report_time = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    meta_table_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style),
            Paragraph(f"<b>Patient Name:</b> {patient_name}", body_style),
            Paragraph(f"<b>Date:</b> {report_time}", body_style)
        ],
        [
            Paragraph(f"<b>Model Used:</b> {model_name}", body_style),
            Paragraph("<b>Diagnostic Tool:</b> CKD Explainable AI v1.0", body_style),
            Paragraph("<b>Facility:</b> Renal Care Diagnostic Center", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[180, 180, 180])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Diagnostic Conclusion Box
    is_ckd = prediction_result.get("is_ckd", False)
    probability = prediction_result.get("prediction_probability", 0.0)
    risk_pct = probability * 100

    if is_ckd:
        status_text = "CHRONIC KIDNEY DISEASE (CKD) DETECTED - HIGH RISK"
        box_bg = colors.HexColor("#FFF5F5")
        border_color = colors.HexColor("#E53E3E")
        text_color = "#C53030"
    else:
        status_text = "NO EVIDENCE OF CKD - LOW RISK / NORMAL RENAL PROFILE"
        box_bg = colors.HexColor("#F0FFF4")
        border_color = colors.HexColor("#38A169")
        text_color = "#276749"

    diag_style = ParagraphStyle(
        "DiagStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(text_color)
    )
    diag_data = [
        [
            Paragraph(f"DIAGNOSTIC STATUS: {status_text}", diag_style),
            Paragraph(f"Calculated Risk Probability: <b>{risk_pct:.1f}%</b>", diag_style)
        ]
    ]
    diag_table = Table(diag_data, colWidths=[380, 160])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), box_bg),
        ("BOX", (0, 0), (-1, -1), 1.5, border_color),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 10))

    # 4. Biomarkers Table
    story.append(Paragraph("Patient Clinical Laboratory Biomarkers", section_style))
    lab_rows = [
        [
            Paragraph("<b>Parameter</b>", bold_style),
            Paragraph("<b>Observed Value</b>", bold_style),
            Paragraph("<b>Reference Range</b>", bold_style),
            Paragraph("<b>Parameter</b>", bold_style),
            Paragraph("<b>Observed Value</b>", bold_style),
            Paragraph("<b>Reference Range</b>", bold_style)
        ]
    ]

    keys = list(patient_data.keys())
    half = (len(keys) + 1) // 2
    for i in range(half):
        k1 = keys[i]
        meta1 = FEATURE_METADATA.get(k1, {})
        val1 = str(patient_data.get(k1, "N/A"))
        unit1 = meta1.get("unit", "")
        range1 = meta1.get("normal_range", "N/A")
        name1 = meta1.get("name", k1)
        disp1 = f"{val1} {unit1}".strip()

        if i + half < len(keys):
            k2 = keys[i + half]
            meta2 = FEATURE_METADATA.get(k2, {})
            val2 = str(patient_data.get(k2, "N/A"))
            unit2 = meta2.get("unit", "")
            range2 = meta2.get("normal_range", "N/A")
            name2 = meta2.get("name", k2)
            disp2 = f"{val2} {unit2}".strip()
        else:
            name2, disp2, range2 = "", "", ""

        lab_rows.append([
            Paragraph(name1, body_style),
            Paragraph(disp1, bold_style),
            Paragraph(range1, body_style),
            Paragraph(name2, body_style),
            Paragraph(disp2, bold_style),
            Paragraph(range2, body_style),
        ])

    lab_table = Table(lab_rows, colWidths=[110, 80, 80, 110, 80, 80])
    lab_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(lab_table)
    story.append(Spacer(1, 10))

    # 5. Explainable AI (SHAP) Attribution Table
    story.append(Paragraph("Explainable AI (SHAP) Biomarker Attribution Analysis", section_style))
    story.append(Paragraph(
        "SHAP values quantify how much each specific biomarker pushed the model prediction toward or away from CKD risk relative to baseline expectations.",
        body_style
    ))
    story.append(Spacer(1, 6))

    top_shap = prediction_result.get("top_contributions", [])[:6]
    shap_rows = [
        [
            Paragraph("<b>Biomarker Feature</b>", bold_style),
            Paragraph("<b>Patient Value</b>", bold_style),
            Paragraph("<b>SHAP Value</b>", bold_style),
            Paragraph("<b>Risk Direction</b>", bold_style)
        ]
    ]

    for item in top_shap:
        f_key = item["feature"]
        raw_v = item["raw_value"]
        s_val = item["shap_value"]
        direction = item["impact"]
        f_meta = FEATURE_METADATA.get(f_key, {})
        f_name = f_meta.get("name", f_key)
        dir_color = "#C53030" if "Increases" in direction else "#276749"

        shap_rows.append([
            Paragraph(f_name, body_style),
            Paragraph(str(raw_v), body_style),
            Paragraph(f"{s_val:+.4f}", bold_style),
            Paragraph(f"<font color='{dir_color}'><b>{direction}</b></font>", body_style)
        ])

    shap_table = Table(shap_rows, colWidths=[170, 110, 110, 150])
    shap_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(shap_table)

    # 6. Embedded Waterfall Plot if available
    waterfall_fig = prediction_result.get("waterfall_figure")
    if waterfall_fig is not None:
        buf = io.BytesIO()
        waterfall_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        story.append(Spacer(1, 8))
        story.append(Paragraph("Patient SHAP Waterfall Chart", section_style))
        story.append(Image(buf, width=480, height=220))
        plt.close(waterfall_fig)

    # 7. Clinical Disclaimer & Sign-off
    story.append(Spacer(1, 10))
    disclaimer_text = (
        "<b>Medical Disclaimer:</b> This diagnostic report was generated by an automated machine learning decision "
        "support system. The results are statistical predictions and must not replace professional judgment by a licensed "
        "nephrologist. Further diagnostic confirmatory tests (eGFR, Renal Ultrasound, Serum Creatinine Clearance) are recommended."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle("Disc", parent=body_style, fontSize=7, leading=10, textColor=colors.HexColor("#718096"))))
    story.append(Spacer(1, 15))

    sig_data = [
        [
            Paragraph("<b>Reviewing Nephrologist:</b> ___________________________", body_style),
            Paragraph("<b>Signature:</b> ___________________________", body_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([("PADDING", (0, 0), (-1, -1), 2)]))
    story.append(sig_table)

    # Build Document
    doc.build(story)
    return output_path
