import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier.validate import (cross_check, decide, mask, mask_fields, normalize,
                               normalize_date, validate_fields)

TODAY = date(2026, 10, 1)
GOOD = {"name": "Aarav Sharma", "dob": "05/04/1999", "id_number": "ABCDE1234F", "expiry_date": "01/01/2030"}


def test_normalize_date_formats():
    assert normalize_date("05/04/1999") == date(1999, 4, 5)
    assert normalize_date("05-04-1999") == date(1999, 4, 5)
    assert normalize_date("31/02/2000") is None
    assert normalize_date(None) is None


def test_normalize_fields():
    assert normalize("name", "  aarav   sharma. ") == "AARAV SHARMA"
    assert normalize("id_number", "1234 5678 9012") == "123456789012"
    assert normalize("dob", "05/04/1999") == "1999-04-05"
    assert normalize("name", "  ") is None


def test_validate_good_document():
    assert validate_fields(GOOD, today=TODAY) == {}


def test_validate_detects_expired_bad_id_and_future_dob():
    bad = {**GOOD, "expiry_date": "01/01/2020", "id_number": "XYZ", "dob": "01/01/2030"}
    issues = validate_fields(bad, today=TODAY)
    assert "expired" in issues["expiry_date"]
    assert "invalid_format" in issues["id_number"]
    assert "implausible" in issues["dob"]


def test_cross_check_fuzzy_name_and_mismatch():
    other = {**GOOD, "name": "SHARMA AARAV", "id_number": "ABCDE1234X"}
    out = cross_check(GOOD, other)
    assert out["name"] == "match"          # token-order tolerant
    assert out["id_number"] == "mismatch"
    assert cross_check(GOOD, {**GOOD, "dob": None})["dob"] == "missing"


def test_decide():
    ok = {f: "match" for f in GOOD}
    assert decide({}, ok) == "VERIFIED"
    assert decide({"expiry_date": ["expired"]}, ok) == "REJECTED"
    assert decide({}, {**ok, "dob": "mismatch"}) == "REVIEW"
    assert decide({"id_number": ["invalid_format"]}, ok) == "REVIEW"


def test_masking():
    assert mask("ABCDE1234F") == "******234F"
    m = mask_fields(GOOD)
    assert m["id_number"] == "******234F" and m["dob"] == "**/**/1999"
    assert m["name"] == GOOD["name"]
