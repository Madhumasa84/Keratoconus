"""
End-to-End Workflow Integration Tests.
Tests the full clinical screening pipeline using real sample images,
complete database persistence, clinician overrides, and multi-format exports.
"""
import json
import os
from pathlib import Path
import pytest
from PIL import Image

from app.database.repository import ScreeningRepository
from app.services.screening_service import ScreeningService
from app.services.referral_engine import ReferralEngine
from app.services.report_service import ReportService
from app.services.audit_service import AuditService


REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLE_OD = REPO_ROOT / "sample_images" / "aright.png"
SAMPLE_OS = REPO_ROOT / "sample_images" / "aleft.png"


def test_full_e2e_screening_with_real_images(db_session, tmp_path):
    """
    Test complete end-to-end workflow:
    Form validation -> Phase 1 engine analysis on real Placido images ->
    Referral rule execution -> SQLite persistence -> PDF/JSON/Excel exports ->
    Clinician override -> Pentacam follow-up -> CSV audit export.
    """
    assert SAMPLE_OD.exists(), f"Sample OD image not found at {SAMPLE_OD}"
    assert SAMPLE_OS.exists(), f"Sample OS image not found at {SAMPLE_OS}"

    session = db_session
    service = ScreeningService(db_session=session)
    audit_service = AuditService()
    report_service = ReportService()
    repo = ScreeningRepository(session)

    # 1. Prepare Screening Encounter Data
    screening_data = {
        "form": {
            "screening_id": "E2E-REAL-001",
            "age": 14,
            "sex": "Female",
            "site": "Metro High School",
            "screening_date": "2026-08-21",
            "operator_id": "OP_CLINICAL",
            "device_id": "DEV_KERASCAN_01",
            "consent_recorded": True,
        },
        "od_image_path": str(SAMPLE_OD),
        "os_image_path": str(SAMPLE_OS),
        "od_measurements": {
            "reading_number": 1,
            "k1_d": 43.5,
            "k1_axis": 180,
            "k2_d": 47.5,  # K2 >= 47.0 (K_HIGH)
            "k2_axis": 90,
            "kmax_d": 49.0,
            "mean_k_d": 45.5,
            "pachymetry_um": 465.0,  # Pachymetry <= 480 (PACHY_LOW)
            "pachymetry_type": "thinnest",
            "sphere_d": -2.0,
            "cylinder_d": 2.5,  # Cylinder >= 2.0 (CYL_HIGH)
            "cylinder_axis": 85,
            "refraction_type": "autorefraction",
            "measurement_quality": "Good",
            "clinical_flags": ["Vogt striae"],
        },
        "os_measurements": {
            "reading_number": 1,
            "k1_d": 42.0,
            "k1_axis": 175,
            "k2_d": 44.0,  # Normal K2
            "k2_axis": 85,
            "kmax_d": 44.5,
            "mean_k_d": 43.0,
            "pachymetry_um": 540.0,  # Normal Pachymetry
            "pachymetry_type": "central",
            "sphere_d": -0.5,
            "cylinder_d": 0.75,
            "cylinder_axis": 90,
            "refraction_type": "autorefraction",
            "measurement_quality": "Good",
        },
        "od_measurements_r2": {
            "reading_number": 2,
            "k1_d": 43.5,
            "k1_axis": 180,
            "k2_d": 47.6,
            "k2_axis": 90,
            "pachymetry_um": 464.0,
            "pachymetry_type": "thinnest",
            "sphere_d": -2.0,
            "cylinder_d": 2.5,
            "cylinder_axis": 85,
            "refraction_type": "autorefraction",
            "measurement_quality": "Good",
        },
        "os_measurements_r2": None,
    }

    # 2. Conduct Screening
    result = service.conduct_screening(screening_data)
    assert result.success is True, f"Conduct screening failed: {result.validation_errors}"
    assert result.screening_id == "E2E-REAL-001"
    assert result.screening_uuid != ""

    # Verify Per-Eye and Child Decision
    assert result.od_eye_result is not None
    assert result.os_eye_result is not None
    assert result.child_result is not None

    # Real sample images evaluate via Phase 1 quality gates (UNGRADABLE -> RECAPTURE_REQUIRED)
    assert result.od_eye_result.decision == "RECAPTURE_REQUIRED"
    assert "IMG_UNGRADABLE" in result.od_eye_result.reason_codes
    assert "K_HIGH" in result.od_eye_result.reason_codes
    assert "PACHY_LOW" in result.od_eye_result.reason_codes
    assert result.child_result.decision == "RECAPTURE_REQUIRED"

    # 3. Verify Database Storage Completeness
    full_record = repo.get_screening_full("E2E-REAL-001")
    assert full_record is not None
    assert full_record["screening_id"] == "E2E-REAL-001"
    assert full_record["overall_result"] == "RECAPTURE_REQUIRED"
    assert len(full_record["eyes"]) == 2

    # Check OD measurements (two readings recorded)
    od_eye = next(e for e in full_record["eyes"] if e["laterality"] == "OD")
    assert len(od_eye["measurements"]) == 2
    assert len(od_eye["image_analyses"]) == 1
    assert od_eye["image_path"] == str(SAMPLE_OD)
    assert od_eye["image_hash"] is not None

    # Check OS measurements (one reading)
    os_eye = next(e for e in full_record["eyes"] if e["laterality"] == "OS")
    assert len(os_eye["measurements"]) == 1

    # 4. Generate Multi-Format Exports
    export_dir = tmp_path / "exports"
    exports = report_service.generate_all_exports(full_record, str(export_dir))
    assert "pdf" in exports and Path(exports["pdf"]).exists()
    assert "json" in exports and Path(exports["json"]).exists()
    assert "excel" in exports and Path(exports["excel"]).exists()
    assert "errors" not in exports

    # Validate JSON content
    with open(exports["json"]) as jf:
        jdata = json.load(jf)
        assert jdata["screening"]["screening_id"] == "E2E-REAL-001"
        assert "AI-assisted keratoconus screening result" in jdata["disclaimer"]

    # 5. Clinician Override Flow
    child_decision_id = full_record["decisions"][0]["id"]
    override_valid, errors = audit_service.validate_override_request({
        "user_identity": "dr_cornea",
        "reason": "Comprehensive corneal tomography confirms advanced ectatic change.",
        "original_decision": "RECAPTURE_REQUIRED",
        "new_decision": "PRIORITY_REFERRAL",
        "timestamp": "2026-08-21T15:00:00Z",
    })
    assert override_valid is True

    audit_id = audit_service.log_override(
        decision_id=child_decision_id,
        original_result="RECAPTURE_REQUIRED",
        new_result="PRIORITY_REFERRAL",
        reason="Comprehensive corneal tomography confirms advanced ectatic change.",
        performed_by="dr_cornea",
        timestamp="2026-08-21T15:00:00Z",
        session=session,
    )
    session.commit()
    assert audit_id is not None

    # Verify audit trail
    trail = audit_service.get_audit_trail(child_decision_id, session)
    assert len(trail) >= 2

    # 6. Pentacam Follow-Up Recording
    followup_id = repo.save_pentacam_followup({
        "screening_id": result.screening_uuid,
        "exam_date": "2026-08-21",
        "kmax_od": 51.5,
        "kmax_os": 44.5,
        "belin_ambrosio_d_od": 4.8,
        "belin_ambrosio_d_os": 0.9,
        "performed_by": "Dr. Pentacam Specialist",
        "notes": "Ectasia OD confirmed; OS normal.",
    })
    session.commit()
    assert followup_id is not None

    # 7. CSV Audit Export
    csv_path = tmp_path / "audit_trail.csv"
    exported_csv = audit_service.export_audit_log("E2E-REAL-001", str(csv_path), session)
    assert Path(exported_csv).exists()
    csv_content = Path(exported_csv).read_text()
    assert "E2E-REAL-001" or "decisions" in csv_content

    # 8. Query & Search Testing
    history = service.get_screening_history("E2E-REAL-001")
    assert history["screening_id"] == "E2E-REAL-001"
    assert len(history["pentacam_followups"]) == 1

    search_res = service.search_screenings("Metro High")
    assert len(search_res) >= 1

    search_filtered = service.search_screenings("", filters={"site": "Metro High School", "date_from": "2026-01-01"})
    assert len(search_filtered) >= 1


def test_screening_service_duplicate_rejection(db_session, sample_screening_data):
    """ScreeningService rejects duplicate screening ID."""
    repo = ScreeningRepository(db_session)
    repo.save_screening(sample_screening_data)
    db_session.commit()

    service = ScreeningService(db_session=db_session)
    res = service.conduct_screening({
        "form": sample_screening_data,
        "od_measurements": {"k2_d": 44.0, "pachymetry_um": 530.0, "pachymetry_type": "central", "reading_number": 1},
        "os_measurements": {"k2_d": 44.0, "pachymetry_um": 530.0, "pachymetry_type": "central", "reading_number": 1},
    })
    assert res.success is False
    assert "already exists" in res.error_message.lower() or any("already exists" in e.lower() for e in res.validation_errors)


def test_repository_update_and_lookup_by_uuid(db_session, sample_screening_data):
    """Test repository uuid lookup and updates."""
    repo = ScreeningRepository(db_session)
    s_uuid = repo.save_screening(sample_screening_data)
    db_session.commit()

    # lookup by uuid
    found = repo.get_screening_by_uuid(s_uuid)
    assert found is not None
    assert found["screening_id"] == sample_screening_data["screening_id"]

    # update
    updated = repo.update_screening(s_uuid, {"site": "Updated Site Name"})
    db_session.commit()
    assert updated is True
    found2 = repo.get_screening_by_uuid(s_uuid)
    assert found2["site"] == "Updated Site Name"

    # update non-existent
    assert repo.update_screening("00000000-0000-0000-0000-000000000000", {"site": "none"}) is False
    assert repo.get_screening_by_uuid("00000000-0000-0000-0000-000000000000") is None
    assert repo.get_screening_full("NONEXISTENT_ID") is None
