import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cv2 = pytest.importorskip("cv2")

from verifier.preprocess import decode_image, estimate_skew_angle, preprocess_for_ocr, rotate


def text_image():
    img = np.full((400, 900, 3), 255, np.uint8)
    for i, y in enumerate((80, 160, 240, 320)):
        cv2.putText(img, f"Line {i} Name: Test Person ID ABCDE1234F", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    return img


def test_skew_estimation_recovers_rotation():
    straight = cv2.cvtColor(text_image(), cv2.COLOR_BGR2GRAY)
    skewed = rotate(straight, 5.0)
    angle = estimate_skew_angle(skewed)
    assert abs(angle + 5.0) <= 1.0           # correction ~ -5 degrees
    assert abs(estimate_skew_angle(straight)) <= 1.0


def test_preprocess_returns_binary_image():
    out = preprocess_for_ocr(text_image())
    assert out.ndim == 2 and set(np.unique(out)) <= {0, 255}


def test_decode_image_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    ok, buf = cv2.imencode(".png", text_image())
    assert decode_image(buf.tobytes()).shape[2] == 3
