"""
Streamlit Input Form Component
Provides structured, validated clinical parameter input fields with medical reference tooltips.
"""
from typing import Dict, Any, Optional
import streamlit as st

try:
    from src.config import FEATURE_METADATA
except ImportError:
    from config import FEATURE_METADATA


def render_patient_input_form(initial_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Renders categorized patient data entry fields with normal range tooltips.
    Returns a dictionary of clinical parameters ready for model inference.
    """
    init = initial_values or {}

    col_cat1, col_cat2 = st.columns(2)

    with col_cat1:
        st.subheader("🩸 Blood Chemistry & Renal Biomarkers")

        sc = st.number_input(
            "Serum Creatinine (sc) [Normal: 0.6 - 1.2 mg/dL]",
            min_value=0.4,
            max_value=20.0,
            value=float(init.get("sc", 1.0)),
            step=0.1,
            help="High creatinine indicates reduced glomerular filtration."
        )

        bu = st.number_input(
            "Blood Urea (bu) [Normal: 15 - 45 mg/dL]",
            min_value=5.0,
            max_value=400.0,
            value=float(init.get("bu", 35.0)),
            step=1.0,
            help="Elevated levels occur when kidneys fail to filter urea nitrogen."
        )

        bgr = st.number_input(
            "Blood Glucose Random (bgr) [Normal: 70 - 140 mg/dL]",
            min_value=40.0,
            max_value=500.0,
            value=float(init.get("bgr", 110.0)),
            step=5.0,
            help="High blood glucose correlates strongly with diabetic nephropathy."
        )

        sod = st.number_input(
            "Serum Sodium (sod) [Normal: 135 - 145 mEq/L]",
            min_value=90.0,
            max_value=170.0,
            value=float(init.get("sod", 138.0)),
            step=1.0,
            help="Electrolyte balance regulated by renal tubules."
        )

        pot = st.number_input(
            "Serum Potassium (pot) [Normal: 3.5 - 5.0 mEq/L]",
            min_value=1.5,
            max_value=12.0,
            value=float(init.get("pot", 4.2)),
            step=0.1,
            help="Hyperkalemia can be life-threatening in severe kidney disease."
        )

        st.subheader("🔬 Hematology Panel")

        hemo = st.number_input(
            "Hemoglobin (hemo) [Normal: 12.0 - 16.5 g/dL]",
            min_value=3.0,
            max_value=20.0,
            value=float(init.get("hemo", 14.5)),
            step=0.1,
            help="Kidneys produce erythropoietin; kidney disease often induces anemia."
        )

        pcv = st.number_input(
            "Packed Cell Volume (pcv) [Normal: 36 - 50 %]",
            min_value=10.0,
            max_value=60.0,
            value=float(init.get("pcv", 42.0)),
            step=1.0,
            help="Hematocrit percentage of blood volume occupied by red cells."
        )

        rbcc = st.number_input(
            "Red Blood Cell Count (rbcc) [Normal: 4.2 - 5.9 million/μL]",
            min_value=1.5,
            max_value=8.0,
            value=float(init.get("rbcc", 4.8)),
            step=0.1
        )

        wbcc = st.number_input(
            "White Blood Cell Count (wbcc) [Normal: 4,500 - 11,000 cells/μL]",
            min_value=1500.0,
            max_value=35000.0,
            value=float(init.get("wbcc", 7500.0)),
            step=100.0
        )

    with col_cat2:
        st.subheader("🧪 Urinalysis & Physical Vitals")

        age = st.number_input(
            "Age (years)",
            min_value=1.0,
            max_value=105.0,
            value=float(init.get("age", 52.0)),
            step=1.0
        )

        bp = st.number_input(
            "Blood Pressure (bp) [Normal: 60 - 80 mm Hg diastolic/mean]",
            min_value=50.0,
            max_value=180.0,
            value=float(init.get("bp", 80.0)),
            step=5.0
        )

        sg_options = [1.005, 1.010, 1.015, 1.020, 1.025]
        sg_idx = sg_options.index(float(init.get("sg", 1.020))) if float(init.get("sg", 1.020)) in sg_options else 3
        sg = st.selectbox(
            "Specific Gravity (sg) [Normal: 1.010 - 1.025]",
            options=sg_options,
            index=sg_idx,
            help="Inability to concentrate urine is an early hallmark of CKD."
        )

        al = st.selectbox(
            "Albumin in Urine (al) [0 = Normal, 1-5 = Proteinuria]",
            options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            index=int(init.get("al", 0.0)),
            help="High albumin excretion indicates glomerular permeability damage."
        )

        su = st.selectbox(
            "Sugar in Urine (su) [0 = None, 1-5 = Glucosuria]",
            options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            index=int(init.get("su", 0.0))
        )

        rbc = st.selectbox(
            "Red Blood Cells in Urine (rbc)",
            options=["normal", "abnormal"],
            index=0 if init.get("rbc", "normal") == "normal" else 1
        )

        pc = st.selectbox(
            "Pus Cells in Urine (pc)",
            options=["normal", "abnormal"],
            index=0 if init.get("pc", "normal") == "normal" else 1
        )

        pcc = st.selectbox(
            "Pus Cell Clumps (pcc)",
            options=["notpresent", "present"],
            index=0 if init.get("pcc", "notpresent") == "notpresent" else 1
        )

        ba = st.selectbox(
            "Bacteria in Urine (ba)",
            options=["notpresent", "present"],
            index=0 if init.get("ba", "notpresent") == "notpresent" else 1
        )

    st.subheader("📋 Comorbidities & Clinical Symptoms")
    col_c1, col_c2, col_c3, col_c4, col_c5, col_c6 = st.columns(6)

    with col_c1:
        htn = st.selectbox("Hypertension", ["no", "yes"], index=0 if init.get("htn", "no") == "no" else 1)
    with col_c2:
        dm = st.selectbox("Diabetes", ["no", "yes"], index=0 if init.get("dm", "no") == "no" else 1)
    with col_c3:
        cad = st.selectbox("CAD History", ["no", "yes"], index=0 if init.get("cad", "no") == "no" else 1)
    with col_c4:
        appet = st.selectbox("Appetite", ["good", "poor"], index=0 if init.get("appet", "good") == "good" else 1)
    with col_c5:
        pe = st.selectbox("Pedal Edema", ["no", "yes"], index=0 if init.get("pe", "no") == "no" else 1)
    with col_c6:
        ane = st.selectbox("Anemia", ["no", "yes"], index=0 if init.get("ane", "no") == "no" else 1)

    patient_payload = {
        "age": age,
        "bp": bp,
        "sg": sg,
        "al": al,
        "su": su,
        "rbc": rbc,
        "pc": pc,
        "pcc": pcc,
        "ba": ba,
        "bgr": bgr,
        "bu": bu,
        "sc": sc,
        "sod": sod,
        "pot": pot,
        "hemo": hemo,
        "pcv": pcv,
        "wbcc": wbcc,
        "rbcc": rbcc,
        "htn": htn,
        "dm": dm,
        "cad": cad,
        "appet": appet,
        "pe": pe,
        "ane": ane
    }

    return patient_payload
