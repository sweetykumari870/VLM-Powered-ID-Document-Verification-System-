from typing import Optional

from .ocr import FIELDS, parse_fields, run_ocr
from .preprocess import decode_image, preprocess_for_ocr
from .validate import cross_check, decide, mask_fields, validate_fields
from .vlm import extract_with_vlm


def ocr_extract(image_bytes: bytes) -> dict:
    return parse_fields(run_ocr(preprocess_for_ocr(decode_image(image_bytes))))


def verify_document(image_bytes: bytes, prompt_version: str = "v2_schema_strict", client=None,
                    ocr_fn=ocr_extract) -> dict:
    """OCR + VLM extraction -> validation -> cross-check -> decision. Image is never persisted.
    Output values are masked."""
    vlm_fields = extract_with_vlm(image_bytes, prompt_version, client=client)
    try:
        ocr_fields = ocr_fn(image_bytes)
        ocr_note = None
    except Exception as e:  # e.g. tesseract binary missing: degrade gracefully to VLM-only
        ocr_fields, ocr_note = {f: None for f in FIELDS}, f"OCR unavailable: {type(e).__name__}"
    issues = validate_fields(vlm_fields)
    checks = cross_check(ocr_fields, vlm_fields)
    decision = decide(issues, checks) if not ocr_note else (
        "REJECTED" if any("expired" in v for v in issues.values()) else "REVIEW")
    return {
        "decision": decision,
        "fields": mask_fields(vlm_fields),
        "validation_issues": issues,
        "ocr_vs_vlm": checks,
        "notes": ocr_note,
        "prompt_version": prompt_version,
    }
