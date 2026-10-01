import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier.metrics import evaluate, field_counts
from verifier.ocr import parse_fields
from verifier.pipeline import verify_document
from verifier.vlm import PROMPTS, extract_with_vlm, parse_vlm_json, sniff_media_type

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20
TRUTH = {"name": "Aarav Sharma", "dob": "05/04/1999", "id_number": "ABCDE1234F", "expiry_date": "01/01/2030"}


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)]))


def test_parse_fields_from_ocr_text():
    text = "SAMPLE ID\nName: Aarav Sharma\nDOB: 05/04/1999\nID Number: ABCDE1234F\nExpiry: 01/01/2030\n"
    assert parse_fields(text) == TRUTH


def test_parse_fields_missing():
    out = parse_fields("nothing useful")
    assert all(v is None for v in out.values())


def test_sniff_media_type():
    assert sniff_media_type(PNG) == "image/png"
    assert sniff_media_type(b"\xff\xd8\xff\xe0abc") == "image/jpeg"
    with pytest.raises(ValueError):
        sniff_media_type(b"GIF89a....")


def test_parse_vlm_json_handles_fences_and_null():
    out = parse_vlm_json('```json\n{"name": "A B", "dob": null, "id_number": "X", "expiry_date": ""}\n```')
    assert out["name"] == "A B" and out["dob"] is None and out["expiry_date"] is None


def test_extract_with_vlm_and_unknown_version():
    out = extract_with_vlm(PNG, "v2_schema_strict", client=FakeClient(json.dumps(TRUTH)))
    assert out == TRUTH
    with pytest.raises(ValueError):
        extract_with_vlm(PNG, "nope", client=FakeClient("{}"))
    assert set(PROMPTS) == {"v1_basic", "v2_schema_strict"}


def test_verify_document_end_to_end_with_fakes():
    res = verify_document(PNG, client=FakeClient(json.dumps({**TRUTH, "expiry_date": "01/01/2099"})),
                          ocr_fn=lambda b: {**TRUTH, "expiry_date": "01/01/2099"})
    assert res["decision"] == "VERIFIED"
    assert res["fields"]["id_number"].endswith("234F") and "ABCDE" not in res["fields"]["id_number"]


def test_verify_document_rejects_expired_and_degrades_without_ocr():
    def boom(_):
        raise RuntimeError("tesseract missing")
    res = verify_document(PNG, client=FakeClient(json.dumps({**TRUTH, "expiry_date": "01/01/2020"})), ocr_fn=boom)
    assert res["decision"] == "REJECTED" and "OCR unavailable" in res["notes"]
    res2 = verify_document(PNG, client=FakeClient(json.dumps({**TRUTH, "expiry_date": "01/01/2099"})), ocr_fn=boom)
    assert res2["decision"] == "REVIEW"


def test_metrics_counts_and_evaluate():
    pred = {**TRUTH, "name": "Wrong Name", "dob": None}
    tp, fp, fn, correct = field_counts(pred, TRUTH)
    assert (tp, fp, fn) == (2, 1, 2) and correct["name"] is False
    res = evaluate([TRUTH, pred], [TRUTH, TRUTH])
    assert res["exact_match_doc"] == 0.5
    assert res["field_accuracy"]["id_number"] == 1.0
    assert {e["field"] for e in res["errors"]} == {"name", "dob"}
    assert 0 < res["f1"] < 1
