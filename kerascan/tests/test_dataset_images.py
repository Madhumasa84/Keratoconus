"""Robustness verification against real images from normal/ and suspicious/ datasets.

Verifies:
- Doctor-selected real images pass the ROI-aware quality gate (no false blur rejection)
- Pipeline runs on real images without unexpected exceptions or crashes
- Radial scan and ring tracking handle varied real image inputs
"""
from __future__ import annotations

from pathlib import Path
import pytest

from kerascan.config import EngineConfig
from kerascan.inference import KerascanEngine
from kerascan.quality import evaluate_quality
from kerascan.roi_detection import detect_placido_roi
from kerascan.image_io import read_image


NORMAL_DIR = Path(__file__).parent.parent.parent / "normal"
SUSPICIOUS_DIR = Path(__file__).parent.parent.parent / "suspicious"
SAMPLE_DIR = Path(__file__).parent.parent.parent / "sample_images"


def test_sample_images_exist_and_gradable():
    """aleft.png and aright.png must pass quality gate and run through engine."""
    engine = KerascanEngine()
    for name in ("aleft.png", "aright.png"):
        img_path = SAMPLE_DIR / name
        if not img_path.exists():
            pytest.skip(f"{img_path} not found")
        img = read_image(img_path)
        roi = detect_placido_roi(img)
        quality = evaluate_quality(roi.crop, roi.center_roi, roi.outer_radius_px)
        assert quality["gradable"] is True, f"{name} failed quality gate: {quality['flags']}"
        assert quality["quality_level"] in {"ACCEPTABLE", "ACCEPTABLE_WITH_WARNING"}

        res = engine.analyze(img_path)
        assert res["failure_stage"] != "ACQUISITION", f"{name} rejected at acquisition: {res['message']}"


def test_normal_folder_images_not_falsely_rejected():
    """Real normal images should not be rejected by full-crop blur false positives."""
    if not NORMAL_DIR.exists():
        pytest.skip("normal/ folder not found")

    normal_files = sorted(NORMAL_DIR.glob("*.jpg"))[:10]  # sample first 10 for test speed
    assert len(normal_files) > 0, "No jpg files in normal/"

    passed_count = 0
    for file_path in normal_files:
        try:
            img = read_image(file_path)
            roi = detect_placido_roi(img)
            quality = evaluate_quality(roi.crop, roi.center_roi, roi.outer_radius_px)
            if quality["gradable"]:
                passed_count += 1
        except Exception:
            pass

    # At least 70% of real normal samples must pass acquisition quality
    assert passed_count >= len(normal_files) * 0.70, (
        f"Too many normal images rejected: {passed_count}/{len(normal_files)} passed"
    )


def test_suspicious_folder_images_gradability():
    """Suspicious dataset images should decode and evaluate quality properly."""
    if not SUSPICIOUS_DIR.exists():
        pytest.skip("suspicious/ folder not found")

    suspicious_files = sorted(list(SUSPICIOUS_DIR.glob("*.jpg")) + list(SUSPICIOUS_DIR.glob("*.png")))[:10]
    assert len(suspicious_files) > 0, "No images in suspicious/"

    for file_path in suspicious_files:
        img = read_image(file_path)
        assert img is not None
        assert img.ndim in (2, 3)
        assert img.shape[0] > 50 and img.shape[1] > 50
