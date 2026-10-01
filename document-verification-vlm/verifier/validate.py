import re
from datetime import date, datetime
from typing import Dict, List, Optional

from rapidfuzz import fuzz

from .ocr import FIELDS, ID_PATTERN

DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")


def normalize_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def normalize(field: str, value: Optional[str]) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    v = str(value).strip()
    if field == "name":
        return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z ]", "", v)).upper().strip()
    if field == "id_number":
        return re.sub(r"\s+", "", v).upper()
    if field in ("dob", "expiry_date"):
        d = normalize_date(v)
        return d.isoformat() if d else v
    return v


def validate_fields(fields: Dict[str, Optional[str]], today: Optional[date] = None) -> Dict[str, List[str]]:
    """Per-field issues: format, plausibility, expiry."""
    today = today or date.today()
    issues: Dict[str, List[str]] = {f: [] for f in FIELDS}
    for f in FIELDS:
        if not fields.get(f):
            issues[f].append("missing")
    idn = normalize("id_number", fields.get("id_number"))
    if idn and not ID_PATTERN.fullmatch(idn):
        issues["id_number"].append("invalid_format")
    if fields.get("dob"):
        d = normalize_date(fields["dob"])
        if d is None:
            issues["dob"].append("invalid_format")
        elif d > today or (today.year - d.year) > 120:
            issues["dob"].append("implausible")
    if fields.get("expiry_date"):
        e = normalize_date(fields["expiry_date"])
        if e is None:
            issues["expiry_date"].append("invalid_format")
        elif e < today:
            issues["expiry_date"].append("expired")
    return {k: v for k, v in issues.items() if v}


def cross_check(a: Dict[str, Optional[str]], b: Dict[str, Optional[str]], name_threshold: int = 90
                ) -> Dict[str, str]:
    """Compare two extractions (e.g. OCR vs VLM): match | mismatch | missing."""
    out = {}
    for f in FIELDS:
        va, vb = normalize(f, a.get(f)), normalize(f, b.get(f))
        if va is None or vb is None:
            out[f] = "missing"
        elif f == "name":
            out[f] = "match" if fuzz.token_sort_ratio(va, vb) >= name_threshold else "mismatch"
        else:
            out[f] = "match" if va == vb else "mismatch"
    return out


def decide(issues: Dict[str, List[str]], checks: Dict[str, str]) -> str:
    """REJECTED if expired; REVIEW if any issue/mismatch/missing; else VERIFIED."""
    if any("expired" in v for v in issues.values()):
        return "REJECTED"
    if issues or any(s != "match" for s in checks.values()):
        return "REVIEW"
    return "VERIFIED"


def mask(value: Optional[str], visible: int = 4) -> Optional[str]:
    if not value:
        return value
    return "*" * max(0, len(value) - visible) + value[-visible:]


def mask_fields(fields: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Mask sensitive values for API output (id fully hidden except last 4, DOB year only)."""
    out = dict(fields)
    out["id_number"] = mask(fields.get("id_number"))
    d = normalize_date(fields.get("dob"))
    out["dob"] = f"**/**/{d.year}" if d else mask(fields.get("dob"), 0)
    return out
