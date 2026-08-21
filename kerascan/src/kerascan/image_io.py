"""Audit-safe input and output helpers."""
from __future__ import annotations
from pathlib import Path
import hashlib
import cv2
import numpy as np

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported image type: {path.suffix}")
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Preserve an RGBA source verbatim for audit. Analysis helpers intentionally
    # ignore alpha rather than compositing/resizing the original acquisition.
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)


def save_png(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save {path}")
    return path


def image_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
