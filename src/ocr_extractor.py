"""
Medical Document & Lab Report OCR Extraction Module
Extracts clinical biomarkers from digital PDFs and scanned lab report images,
paired with manual clinician verification.
"""
import re
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import io

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    from src.config import FEATURE_METADATA
except ImportError:
    from config import FEATURE_METADATA


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from an electronic PDF lab report."""
    if pypdf is None:
        raise ImportError("pypdf is required to extract text from PDF files.")

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    extracted_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)
    return "\n".join(extracted_text)


def extract_text_from_image(file_bytes: bytes) -> Tuple[str, Optional[str]]:
    """
    Attempt OCR on an image file using pytesseract.
    Returns: (extracted_text, error_message_if_any)
    """
    if pytesseract is None or Image is None:
        return "", "Pillow or pytesseract is not installed."

    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text, None
    except Exception as e:
        # Graceful handling if Tesseract OCR engine binary is not installed on Windows
        return "", f"OCR Engine Notice: {str(e)}"


def parse_clinical_values_from_text(text: str) -> Dict[str, Any]:
    """
    Applies medical regex patterns to parse clinical laboratory biomarkers
    from noisy clinical report text.
    """
    patterns = {
        "sc": r"(?:serum\s+creatinine|creatinine|creat|s\.cr|sc)[\s:=_-]*([0-9]+\.?[0-9]*)",
        "bu": r"(?:blood\s+urea\s+nitrogen|blood\s+urea|urea|bun)[\s:=_-]*([0-9]+\.?[0-9]*)",
        "hemo": r"(?:hemoglobin|haemoglobin|hemo|hgb|hb)[\s:=_-]*([0-9]+\.?[0-9]*)",
        "bgr": r"(?:blood\s+glucose(?:\s+random)?|random\s+blood\s+sugar|glucose|bgr|rbs)[\s:=_-]*([0-9]+\.?[0-9]*)",
        "sod": r"(?:serum\s+sodium|sodium|sod|na\+?)[\s:=_-]*([0-9]+\.?[0-9]*)",
        "pot": r"(?:serum\s+potassium|potassium|pot|k\+?)[\s:=_-]*([0-9]+\.?[0-9]*)",
        "pcv": r"(?:packed\s+cell\s+volume(?:\s*\(pcv\))?|hematocrit|pcv|hct)[\s:=_-]*([0-9]+\.?[0-9]*)",
        "wbcc": r"(?:total\s+leukocyte\s+count(?:\s*\(wbc\))?|white\s+blood\s+cells?|wbcc|wbc|tlc)[\s:=_-]*([0-9,]+)",
        "rbcc": r"(?:total\s+red\s+cells?|red\s+blood\s+cell(?:\s+count)?(?:\s*\(rbc\))?|rbcc|rbc|trc)[\s:=_-]*([0-9]+\.?[0-9]*)",
        "sg": r"(?:urine\s+specific\s+gravity|specific\s+gravity|sg)[\s:=_-]*(1\.0[0-3][0-9])",
        "al": r"(?:urine\s+albumin|albumin|proteinuria|al)[\s:=_-]*(nil|trace|present|[0-5]|\++)",
        "bp": r"(?:blood\s+pressure|b\.p|bp)[\s:=_-]*([0-9]{2,3})\s*(?:/\s*([0-9]{2,3}))?",
        "age": r"(?:age|patient\s+age)[\s:=_-]*([0-9]{1,3})"
    }

    results = {}
    lower_text = text.lower()

    for feature, pattern in patterns.items():
        match = re.search(pattern, lower_text)
        if match:
            raw_val = match.group(1).strip().replace(",", "")
            try:
                if feature == "al":
                    # Map text albumin to 0-5 scale
                    if raw_val in ["nil", "negative", "trace"]:
                        results[feature] = 0.0
                    elif raw_val in ["+", "present"]:
                        results[feature] = 1.0
                    elif raw_val == "++":
                        results[feature] = 2.0
                    elif raw_val == "+++":
                        results[feature] = 3.0
                    else:
                        results[feature] = float(min(5.0, max(0.0, float(raw_val))))
                elif feature == "bp":
                    # If systolic/diastolic matched, take diastolic / mean as bp feature
                    if match.group(2):
                        results[feature] = float(match.group(2))
                    else:
                        results[feature] = float(raw_val)
                elif feature == "sg":
                    # Closest standard specific gravity category
                    val = float(raw_val)
                    options = [1.005, 1.010, 1.015, 1.020, 1.025]
                    closest = min(options, key=lambda x: abs(x - val))
                    results[feature] = closest
                else:
                    results[feature] = float(raw_val)
            except ValueError:
                pass

    return results


def process_lab_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Main entry point for processing an uploaded clinical PDF or image lab report.
    Returns extracted features and extraction status metadata.
    """
    ext = Path(filename).suffix.lower()
    error_msg = None
    extracted_text = ""

    if ext == ".pdf":
        try:
            extracted_text = extract_text_from_pdf(file_bytes)
        except Exception as e:
            error_msg = f"Failed to read PDF: {e}"
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        extracted_text, error_msg = extract_text_from_image(file_bytes)
    else:
        error_msg = f"Unsupported file format: {ext}"

    extracted_features = parse_clinical_values_from_text(extracted_text) if extracted_text else {}

    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "extracted_features": extracted_features,
        "match_count": len(extracted_features),
        "error_notice": error_msg
    }
