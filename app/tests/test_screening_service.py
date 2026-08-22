"""Image-gate and active-input tests for ScreeningService."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.services.screening_service import ScreeningService


def form(identifier="SCREEN-001"):
    return {
        "screening_id": identifier, "age": 14, "sex": "Female", "site": "Synthetic School",
        "screening_date": "2026-08-21", "operator_id": "OP1", "device_id": "KERASCAN-SYNTH",
        "consent_recorded": True,
    }


def measurement(**changes):
    values = {"k1_d": 43.0, "k2_d": 44.0, "pachymetry_um": 530.0, "cylinder_d": 0.50}
    values.update(changes)
    return values


def raw_result(label="NORMAL-LIKE", *, quality="ACCEPTABLE", segmentation="PASS", tracking="PASS",
               failure="NONE", classified=True, geometry="PASS", roi=True, quality_flags=None):
    quality_flags = quality_flags or []
    return {
        "screening_result": label,
        "classification_performed": classified,
        "classification_skipped": not classified,
        "acquisition_quality": {"status": quality, "score": 80.0, "flags": quality_flags, "metrics": {"ring_tracking_confidence": 0.9}},
        "quality": {"status": quality, "gradable": quality == "ACCEPTABLE", "quality_score": 80.0, "flags": quality_flags, "metrics": {"ring_tracking_confidence": 0.9}},
        "roi": {"box_xyxy": [10, 10, 110, 110], "center_full": [60, 60], "outer_radius_px": 50, "confidence": .9, "method": "hough"} if roi else {},
        "segmentation": {"status": segmentation, "confidence": .9, "flags": []},
        "tracking": {"status": tracking, "confidence": .9, "flags": []},
        "geometry_validation": {"status": geometry},
        "failure_stage": failure,
        "pipeline_version": "synthetic-pipeline-1",
        "model": {"model_hash": "synthetic-model-hash"},
        "features": {"mean_ring_spacing": 0.1},
    }


class FakeEngine:
    def __init__(self, by_name):
        self.by_name = by_name

    def analyze(self, source, output_dir=None):
        result = deepcopy(self.by_name.get(Path(source).stem, self.by_name.get("default")))
        if output_dir:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            for name, colour in (
                ("cropped_roi.png", "#4477aa"),
                ("cropped_roi_centres.png", "#aa7744"),
                ("tracked_rings_cartesian.png", "#44aa77"),
                ("directional_spacing.png", "#aa4477"),
            ):
                image = Image.new("RGB", (120, 80), colour)
                ImageDraw.Draw(image).text((5, 5), name, fill="white")
                image.save(output / name)
        return result


@pytest.fixture
def image_files(tmp_path):
    paths = {}
    for eye in ("od", "os", "same"):
        path = tmp_path / f"{eye}.png"
        image = Image.new("RGB", (120, 120), "black")
        draw = ImageDraw.Draw(image)
        draw.ellipse((15, 15, 105, 105), outline="white", width=4)
        draw.text((4, 4), eye, fill="white")
        image.save(path)
        paths[eye] = path
    return paths


def service_with(engine_results):
    service = ScreeningService()
    service._image_engine = FakeEngine(engine_results)
    return service


def test_active_measurements_require_exactly_the_four_inputs():
    service = ScreeningService()
    assert service.validate_measurements(measurement())[0] is True
    for key in ("k1_d", "k2_d", "pachymetry_um", "cylinder_d"):
        candidate = measurement()
        candidate[key] = None
        valid, errors = service.validate_measurements(candidate)
        assert valid is False
        assert errors


def test_measurement_ranges_and_k1_order_are_validated():
    service = ScreeningService()
    assert service.validate_measurements(measurement(k1_d=47, k2_d=46))[0] is False
    assert service.validate_measurements(measurement(k2_d=75))[0] is False
    assert service.validate_measurements(measurement(pachymetry_um=900))[0] is False
    assert service.validate_measurements(measurement(cylinder_d=-13))[0] is False


@pytest.mark.parametrize("eye", ["OD", "OS"])
def test_missing_mandatory_eye_image_is_incomplete(image_files, eye):
    service = service_with({"od": raw_result(), "os": raw_result()})
    data = {
        "form": form(), "od_image_path": str(image_files["od"]) if eye != "OD" else "",
        "os_image_path": str(image_files["os"]) if eye != "OS" else "",
        "od_measurements": measurement(), "os_measurements": measurement(),
    }
    result = service.conduct_screening(data)
    verification = result.od_image_verification if eye == "OD" else result.os_image_verification
    assert verification.image_status == "MISSING"
    assert result.child_result.decision == "INCOMPLETE_SCREENING"


def test_both_images_missing_is_incomplete():
    service = service_with({"default": raw_result()})
    result = service.conduct_screening({"form": form(), "od_measurements": measurement(), "os_measurements": measurement()})
    assert result.od_image_verification.image_status == "MISSING"
    assert result.os_image_verification.image_status == "MISSING"
    assert result.child_result.decision == "INCOMPLETE_SCREENING"


def test_unsupported_image_is_rejected(tmp_path):
    path = tmp_path / "unsupported.bmp"
    Image.new("RGB", (100, 100), "black").save(path)
    verification = service_with({"default": raw_result()}).verify_image(path, "OD")
    assert verification.image_status == "IMAGE_REJECTED"
    assert verification.failure_stage == "UNSUPPORTED"


def test_blurred_or_poor_quality_image_is_rejected(image_files):
    service = service_with({"od": raw_result(quality="FAIL", classified=False, failure="ACQUISITION", quality_flags=["blur"])})
    verification = service.verify_image(image_files["od"], "OD")
    assert verification.image_status == "IMAGE_REJECTED"
    assert verification.failure_stage == "ACQUISITION"
    assert "blur" in verification.message
    assert verification.engine_result != "NORMAL-LIKE" or verification.image_status != "NORMAL_LIKE"


def test_roi_failure_is_image_rejected(image_files):
    service = service_with({"od": raw_result(roi=False, classified=False)})
    verification = service.verify_image(image_files["od"], "OD")
    assert verification.image_status == "IMAGE_REJECTED"
    assert verification.failure_stage == "ROI"


def test_segmentation_and_tracking_failures_have_distinct_states(image_files):
    segmentation = service_with({"od": raw_result(segmentation="FAIL", classified=False, failure="SEGMENTATION")})
    tracking = service_with({"od": raw_result(tracking="FAIL", classified=False, failure="TRACKING")})
    assert segmentation.verify_image(image_files["od"], "OD").image_status == "SEGMENTATION_FAILED"
    assert tracking.verify_image(image_files["od"], "OD").image_status == "TRACKING_FAILED"


def test_hardware_ring_count_unavailable_is_analysis_blocked(image_files):
    service = service_with({"od": raw_result(label="UNGRADABLE", classified=False, failure="CONFIGURATION")})
    verification = service.verify_image(image_files["od"], "OD")
    assert verification.image_status == "ANALYSIS_BLOCKED"
    assert verification.raw_result["classification_performed"] is False


def test_not_calibrated_geometry_cannot_become_an_image_negative(image_files):
    service = service_with({"od": raw_result(label="NOT_CALIBRATED", classified=False, failure="NONE")})
    verification = service.verify_image(image_files["od"], "OD")
    assert verification.image_status == "ANALYSIS_BLOCKED"
    assert verification.engine_result == "NOT_CALIBRATED"


def test_geometry_failure_prevents_classification_state(image_files):
    service = service_with({"od": raw_result(geometry="FAIL", classified=True)})
    verification = service.verify_image(image_files["od"], "OD")
    assert verification.image_status == "TRACKING_FAILED"
    assert verification.geometry_validation_status == "FAIL"


def test_same_upload_cannot_be_used_for_both_eyes(image_files):
    service = service_with({"same": raw_result()})
    result = service.conduct_screening({
        "form": form(), "od_image_path": str(image_files["same"]), "os_image_path": str(image_files["same"]),
        "od_measurements": measurement(), "os_measurements": measurement(),
    })
    assert result.od_image_verification.image_status == "IMAGE_REJECTED"
    assert result.os_image_verification.image_status == "IMAGE_REJECTED"
    assert result.child_result.decision == "INCOMPLETE_SCREENING"


def test_good_quality_bilateral_images_can_produce_completed_negative(image_files, tmp_path):
    service = service_with({"od": raw_result(), "os": raw_result()})
    result = service.conduct_screening({
        "form": form(), "od_image_path": str(image_files["od"]), "os_image_path": str(image_files["os"]),
        "analysis_output_dir": str(tmp_path / "analysis"),
        "od_measurements": measurement(), "os_measurements": measurement(),
    })
    assert result.od_image_verification.image_status == "NORMAL_LIKE"
    assert result.os_image_verification.image_status == "NORMAL_LIKE"
    assert result.child_result.decision == "SCREEN_NEGATIVE"
    assert result.child_result.action == "NO_IMMEDIATE_REFERRAL"


def test_analysis_blocked_image_never_produces_screen_negative(image_files):
    service = service_with({
        "od": raw_result(label="UNGRADABLE", classified=False, failure="CONFIGURATION"),
        "os": raw_result(),
    })
    result = service.conduct_screening({
        "form": form(), "od_image_path": str(image_files["od"]), "os_image_path": str(image_files["os"]),
        "od_measurements": measurement(), "os_measurements": measurement(),
    })
    assert result.od_image_verification.image_status == "ANALYSIS_BLOCKED"
    assert result.child_result.decision == "INCOMPLETE_SCREENING"
    assert result.child_result.decision != "SCREEN_NEGATIVE"


def test_measurements_with_rejected_image_remain_incomplete(image_files):
    service = service_with({"od": raw_result(quality="FAIL", classified=False), "os": raw_result()})
    result = service.conduct_screening({
        "form": form(), "od_image_path": str(image_files["od"]), "os_image_path": str(image_files["os"]),
        "od_measurements": measurement(k2_d=48), "os_measurements": measurement(),
    })
    assert result.child_result.decision == "INCOMPLETE_SCREENING"


def test_missing_measurement_with_valid_images_remains_incomplete(image_files):
    service = service_with({"od": raw_result(), "os": raw_result()})
    result = service.conduct_screening({
        "form": form(), "od_image_path": str(image_files["od"]), "os_image_path": str(image_files["os"]),
        "od_measurements": measurement(k1_d=None), "os_measurements": measurement(),
    })
    assert result.od_eye_result.decision == "INCOMPLETE_SCREENING"
    assert result.child_result.decision == "INCOMPLETE_SCREENING"


def test_new_records_persist_canonical_fields_and_image_provenance(db_session, image_files, tmp_path):
    service = ScreeningService(db_session=db_session)
    service._image_engine = FakeEngine({"od": raw_result(), "os": raw_result()})
    result = service.conduct_screening({
        "form": form("PERSIST-001"), "od_image_path": str(image_files["od"]), "os_image_path": str(image_files["os"]),
        "analysis_output_dir": str(tmp_path / "analysis"), "od_measurements": measurement(), "os_measurements": measurement(),
    })
    assert result.success
    history = service.get_screening_history("PERSIST-001")
    assert history["overall_action"] == "NO_IMMEDIATE_REFERRAL"
    for eye in history["eyes"]:
        assert eye["image_status"] == "NORMAL_LIKE"
        assert eye["image_hash"]
        assert eye["processed_output_hashes"]
        assert eye["analysis_provenance_hash"]
        assert eye["image_analyses"][0]["original_image_hash"] == eye["image_hash"]
        assert eye["measurements"][0]["pachymetry_measurement_type"] == "device_reported"


def test_referral_pdf_is_recorded_by_hash_only_for_refer_action(db_session, image_files, tmp_path):
    service = ScreeningService(db_session=db_session)
    service._image_engine = FakeEngine({"od": raw_result(label="SUSPICIOUS"), "os": raw_result()})
    result = service.conduct_screening({
        "form": form("PDF-REFER-001"), "od_image_path": str(image_files["od"]), "os_image_path": str(image_files["os"]),
        "analysis_output_dir": str(tmp_path / "analysis"), "od_measurements": measurement(), "os_measurements": measurement(),
    })
    assert result.child_result.action == "REFER"
    path = service.generate_referral_pdf(result, tmp_path / "referral.pdf")
    assert Path(path).exists()
    record = service.get_screening_history("PDF-REFER-001")
    assert record["pdf_generated"] is True
    assert record["pdf_sha256"]
    assert record["pdf_path"] is None


def test_simplified_ui_does_not_expose_deferred_inputs():
    source = (Path(__file__).parent.parent / "pages" / "03_measurements.py").read_text()
    for removed_label in (
        "K1 axis", "Kmax", "K2 axis", "Mean K", "Reading 2", "Central pachymetry",
        "Thinnest pachymetry", "Refraction type", "Sphere", "Cyl axis", "VA (logMAR)",
        "Measurement quality", "Clinical signs",
    ):
        assert removed_label not in source
    for active_label in ("K1 flat (D)", "K2 steep (D)", "Thickness (µm)", "Cylinder (D)"):
        assert active_label in source
