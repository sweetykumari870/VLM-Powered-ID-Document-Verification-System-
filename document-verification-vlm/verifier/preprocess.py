import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 10.0, step: float = 0.5) -> float:
    """Return the rotation (degrees) to apply to straighten text, by maximising the
    variance of horizontal projection profiles (text lines align -> sharper profile)."""
    scale = 800.0 / max(gray.shape) if max(gray.shape) > 800 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    h, w = bw.shape
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale -> upscale small images -> deskew -> denoise -> adaptive threshold."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = rotate(gray, estimate_skew_angle(gray))
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)
