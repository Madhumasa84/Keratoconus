"""Phase 1 engine integration tests (no real images required)."""
import sys
import numpy as np
import pytest


def test_phase1_engine_importable():
    from kerascan import EngineConfig, KerascanEngine
    assert KerascanEngine is not None
    assert EngineConfig is not None



def test_engine_result_has_required_keys(kerascan_engine, sample_normal_image):
    result = kerascan_engine.analyze(sample_normal_image)
    required_keys = {"screening_result", "quality", "roi", "pipeline_version"}
    assert required_keys.issubset(set(result.keys()))


def test_blank_image_returns_ungradable(kerascan_engine, blank_image):
    result = kerascan_engine.analyze(blank_image)
    assert result["screening_result"] == "UNGRADABLE"
    assert result["quality"]["gradable"] is False


def test_screening_result_is_valid_label(kerascan_engine, sample_normal_image):
    result = kerascan_engine.analyze(sample_normal_image)
    assert result["screening_result"] in ("SUSPICIOUS", "NORMAL-LIKE", "UNGRADABLE", "NOT_CALIBRATED", "ANALYSIS_BLOCKED")


def test_engine_analysis_failure_is_an_explicit_image_rejection(tmp_path):
    """A missing local file cannot be reinterpreted as a normal image."""
    from app.services.screening_service import ScreeningService
    svc = ScreeningService()
    result = svc._analyse_image("/nonexistent/path/image.png")
    assert result["screening_result"] == "UNGRADABLE"
    assert result["image_status"] == "IMAGE_REJECTED"
    assert "rejected" in result.get("message", "").lower()


def test_incorrect_image_format_is_rejected_without_crash(kerascan_engine, tmp_path):
    bad_file = tmp_path / "bad.png"
    bad_file.write_text("this is not a valid image file")
    from app.services.screening_service import ScreeningService
    svc = ScreeningService()
    result = svc._analyse_image(str(bad_file))
    assert result["screening_result"] == "UNGRADABLE"
    assert result["image_status"] == "IMAGE_REJECTED"


def test_model_and_pipeline_version_captured(kerascan_engine, sample_normal_image):
    result = kerascan_engine.analyze(sample_normal_image)
    assert result.get("pipeline_version") is not None
    assert "phase1" in result.get("pipeline_version", "")


def test_ungradable_result_skips_classifier(kerascan_engine, blank_image):
    result = kerascan_engine.analyze(blank_image)
    assert result.get("classification_skipped") is True
    assert "prototype_score" not in result


def test_numpy_array_input_accepted(kerascan_engine):
    """Engine accepts numpy array directly."""
    import numpy as np
    from PIL import Image, ImageDraw
    # draw simple concentric rings inline
    img_pil = Image.new("RGB", (480, 480), (0, 0, 0))
    draw = ImageDraw.Draw(img_pil)
    cx, cy = 240, 240
    for i in range(1, 9):
        r = i * 25
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(180, 180, 180), width=3)
    img = np.array(img_pil)
    result = kerascan_engine.analyze(img)
    assert "screening_result" in result
