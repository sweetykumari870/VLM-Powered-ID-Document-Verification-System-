import re
from typing import Dict, Optional

FIELDS = ("name", "dob", "id_number", "expiry_date")
_DATE = r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4})"
ID_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z]|\d{4}\s?\d{4}\s?\d{4})\b")


def run_ocr(img_binary) -> str:
    """Tesseract OCR. Requires the tesseract binary (e.g. `sudo apt install tesseract-ocr`)."""
    import pytesseract

    return pytesseract.image_to_string(img_binary, config="--psm 6")


def parse_fields(text: str) -> Dict[str, Optional[str]]:
    """Rule-based field extraction from OCR text (baseline to compare against the VLM)."""
    def find(pattern, flags=re.IGNORECASE | re.MULTILINE):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    name = find(r"^\s*name\s*[:\-]\s*(.+?)\s*$")
    dob = find(r"(?:dob|date\s*of\s*birth)\s*[:\-]?\s*" + _DATE)
    expiry = find(r"(?:expiry(?:\s*date)?|valid\s*(?:until|upto|till)|exp\.?)\s*[:\-]?\s*" + _DATE)
    m = ID_PATTERN.search(text.upper())
    return {"name": name, "dob": dob, "id_number": m.group(1) if m else None, "expiry_date": expiry}
