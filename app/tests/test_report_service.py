"""Unit tests for ReportService (PDF, JSON, Excel)."""
import json
import pytest
from pathlib import Path
from app.services.report_service import ReportService, DISCLAIMER


@pytest.fixture
def svc():
    return ReportService()


@pytest.fixture
def minimal_screening_data():
    return {
        "screening_id": "RPT-001",
        "age": 15,
        "sex": "Male",
        "site": "Test School",
        "screening_date": "2026-08-21",
        "operator_id": "OP01",
        "device_id": "DEV01",
        "consent_recorded": True,
        "overall_result": "SCREEN_NEGATIVE",
        "referral_priority": "NONE",
        "protocol_version": "2.0.0",
        "eyes": [
            {
                "laterality": "OD",
                "eye_result": "NORMAL-LIKE",
                "quality_gradable": True,
                "quality_score": 72.0,
                "quality_flags": [],
                "roi_method": "hough",
                "roi_confidence": 0.91,
                "roi_radius": 190.0,
                "image_path": None,
                "pipeline_version": "phase1-0.1.0",
                "measurements": [{"reading_number": 1, "k2_d": 44.0, "pachymetry_um": 530.0, "pachymetry_type": "central", "cylinder_d": 0.5}],
                "image_analyses": [{"model_hash": "abc123", "prototype_score": 0.2, "classification_skipped": False, "pipeline_version": "phase1-0.1.0"}],
                "decisions": [{"decision_level": "eye", "final_result": "SCREEN_NEGATIVE", "is_overridden": False}],
            },
            {
                "laterality": "OS",
                "eye_result": "NORMAL-LIKE",
                "quality_gradable": True,
                "quality_score": 68.0,
                "quality_flags": [],
                "roi_method": "hough",
                "roi_confidence": 0.87,
                "roi_radius": 185.0,
                "image_path": None,
                "pipeline_version": "phase1-0.1.0",
                "measurements": [{"reading_number": 1, "k2_d": 43.5, "pachymetry_um": 545.0, "pachymetry_type": "central", "cylinder_d": 0.25}],
                "image_analyses": [{"model_hash": "def456", "prototype_score": 0.15, "classification_skipped": False, "pipeline_version": "phase1-0.1.0"}],
                "decisions": [{"decision_level": "eye", "final_result": "SCREEN_NEGATIVE", "is_overridden": False}],
            },
        ],
        "decisions": [{"decision_level": "child", "final_result": "SCREEN_NEGATIVE", "is_overridden": False}],
        "referrals": [],
        "pentacam_followups": [],
    }


def test_pdf_generated(svc, minimal_screening_data, tmp_path):
    path = svc.generate_pdf(minimal_screening_data, str(tmp_path / "test.pdf"))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000  # non-trivial file


def test_pdf_created_without_images(svc, minimal_screening_data, tmp_path):
    """PDF must not crash when image files are missing."""
    data = dict(minimal_screening_data)
    data["eyes"][0]["image_path"] = "/nonexistent/path/od.png"
    path = svc.generate_pdf(data, str(tmp_path / "test_noimages.pdf"))
    assert Path(path).exists()


def test_json_valid(svc, minimal_screening_data, tmp_path):
    path = svc.generate_json(minimal_screening_data, str(tmp_path / "test.json"))
    with open(path) as f:
        data = json.load(f)
    assert "screening" in data
    assert "disclaimer" in data
    assert data["disclaimer"] == DISCLAIMER


def test_excel_has_correct_sheets(svc, minimal_screening_data, tmp_path):
    import openpyxl
    path = svc.generate_excel(minimal_screening_data, str(tmp_path / "test.xlsx"))
    wb = openpyxl.load_workbook(path)
    assert "Summary" in wb.sheetnames
    assert "Measurements" in wb.sheetnames
    assert "Audit Trail" in wb.sheetnames


def test_generate_all_exports(svc, minimal_screening_data, tmp_path):
    results = svc.generate_all_exports(minimal_screening_data, str(tmp_path))
    assert "pdf" in results
    assert "json" in results
    assert "excel" in results
    for fmt, path in results.items():
        if fmt != "errors":
            assert Path(path).exists(), f"{fmt} export not found at {path}"


def test_disclaimer_in_json(svc, minimal_screening_data, tmp_path):
    path = svc.generate_json(minimal_screening_data, str(tmp_path / "d.json"))
    content = Path(path).read_text()
    assert "not a confirmed diagnosis" in content
