"""End-to-end persistence checks using only bundled non-confidential samples."""
from __future__ import annotations

from pathlib import Path

from app.database.repository import ScreeningRepository
from app.services.report_service import ReportService
from app.services.screening_service import ScreeningService


REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLE_OD = REPO_ROOT / "sample_images" / "aright.png"
SAMPLE_OS = REPO_ROOT / "sample_images" / "aleft.png"


def _form(identifier):
    return {
        "screening_id": identifier, "age": 14, "sex": "Female", "site": "Synthetic Test Site",
        "screening_date": "2026-08-21", "operator_id": "OP_TEST", "device_id": "DEV_KERASCAN_01",
        "consent_recorded": True,
    }


def _measurements(k1=43.0, k2=44.0, pachy=530.0, cylinder=0.5):
    return {"k1_d": k1, "k2_d": k2, "pachymetry_um": pachy, "cylinder_d": cylinder}


def test_clean_pattern_and_normal_measurements_complete_as_screen_negative(db_session, tmp_path):
    """The full happy path: a clean ring pattern plus normal measurements.

    The bundled sample images are synthetic concentric patterns (no patient
    imagery is kept in this repository), so both eyes analyse cleanly. With all
    measurements within threshold the encounter must COMPLETE as a screen
    negative -- and still produce no referral PDF, since that is reserved for
    REFER outcomes.

    The converse invariant -- that a borderline or ungradable image can never be
    reported as a clean negative -- is covered in test_referral_engine
    (test_noncompleted_image_can_never_be_normal).
    """
    service = ScreeningService(db_session=db_session)
    result = service.conduct_screening({
        "form": _form("E2E-REAL-001"),
        "od_image_path": str(SAMPLE_OD), "os_image_path": str(SAMPLE_OS),
        "analysis_output_dir": str(tmp_path / "analysis"),
        "od_measurements": _measurements(),
        "os_measurements": _measurements(),
    })
    assert result.success
    assert result.od_image_verification.image_status == "NORMAL_LIKE"
    assert result.os_image_verification.image_status == "NORMAL_LIKE"
    assert result.child_result.decision == "SCREEN_NEGATIVE"
    assert result.child_result.action == "NO_IMMEDIATE_REFERRAL"

    record = ScreeningRepository(db_session).get_screening_full("E2E-REAL-001")
    assert record["overall_result"] == "SCREEN_NEGATIVE"
    assert len(record["eyes"]) == 2
    assert all(eye["measurements"][0]["pachymetry_measurement_type"] == "device_reported" for eye in record["eyes"])

    exports = ReportService().generate_all_exports(record, str(tmp_path / "exports"))
    assert "pdf" not in exports, "a referral PDF belongs only to a REFER outcome"
    assert Path(exports["json"]).exists()
    assert Path(exports["excel"]).exists()


def test_acquisition_quality_is_advisory_not_blocking(db_session, tmp_path):
    """A capture the quality metric dislikes must still be analysed.

    The workflow assumes a clinician reviewed the photo before uploading, so a
    low acquisition-quality score is recorded but never rejects the image. Only
    a genuine pipeline failure (unreadable file, no ring pattern found) stops it.
    """
    service = ScreeningService(db_session=db_session)
    assert service._enforce_quality_gate is False

    verification = service.verify_image(SAMPLE_OD, "OD", tmp_path / "quality")
    assert verification.image_status != "IMAGE_REJECTED"
    # The measurement itself is still produced and retained for review.
    acquisition = verification.raw_result.get("acquisition_quality") or {}
    assert acquisition.get("status")


def test_referral_pdf_is_produced_by_the_real_engine_end_to_end(db_session, tmp_path):
    """A REFER outcome must reach a genuine generated PDF through the real engine.

    This runs the actual image pipeline (not FakeEngine) on the bundled synthetic
    patterns, and reaches REFER via the discordant-measurement route: a
    normal-looking ring pattern with two abnormal quantitative domains. That is
    the case your criteria call "screen-positive due to discordant quantitative
    abnormalities; KeraScan image normal".
    """
    service = ScreeningService(db_session=db_session)
    result = service.conduct_screening({
        "form": _form("E2E-REAL-003"),
        "od_image_path": str(SAMPLE_OD), "os_image_path": str(SAMPLE_OS),
        "analysis_output_dir": str(tmp_path / "analysis3"),
        "od_measurements": _measurements(k2=47.5, pachy=465.0),
        "os_measurements": _measurements(),
    })
    assert result.success
    assert result.od_image_verification.image_status == "NORMAL_LIKE"
    assert result.od_eye_result.decision == "DISCORDANT_SCREEN_POSITIVE"
    assert result.od_eye_result.action == "REFER"
    assert result.child_result.action == "REFER"

    pdf_path = service.generate_referral_pdf(result, tmp_path / "referral.pdf")
    assert Path(pdf_path).exists()
    assert Path(pdf_path).stat().st_size > 0


def test_bare_unconfigured_engine_still_stays_blocked(db_session, tmp_path):
    """A caller who does NOT opt into the provisional demo configuration gets the
    original conservative behavior: the shipped kerascan library default never
    classifies anything on its own."""
    from kerascan import EngineConfig

    service = ScreeningService(db_session=db_session, engine_config=EngineConfig())
    result = service.conduct_screening({
        "form": _form("E2E-REAL-002"),
        "od_image_path": str(SAMPLE_OD), "os_image_path": str(SAMPLE_OS),
        "analysis_output_dir": str(tmp_path / "analysis2"),
        "od_measurements": _measurements(), "os_measurements": _measurements(),
    })
    assert result.success
    assert result.od_image_verification.image_status == "ANALYSIS_BLOCKED"
    assert result.os_image_verification.image_status == "ANALYSIS_BLOCKED"
    assert result.child_result.decision == "INCOMPLETE_SCREENING"


def test_screening_service_duplicate_rejection(db_session):
    repo = ScreeningRepository(db_session)
    repo.save_screening(_form("DUPLICATE-001"))
    db_session.commit()
    result = ScreeningService(db_session=db_session).conduct_screening({
        "form": _form("DUPLICATE-001"), "od_measurements": _measurements(), "os_measurements": _measurements(),
    })
    assert result.success is False
    assert "duplicate" in result.error_message.lower()


def test_repository_update_and_lookup_by_uuid(db_session):
    repo = ScreeningRepository(db_session)
    identifier = "LOOKUP-001"
    screening_uuid = repo.save_screening(_form(identifier))
    db_session.commit()
    assert repo.get_screening_by_uuid(screening_uuid)["screening_id"] == identifier
    assert repo.update_screening(screening_uuid, {"site": "Updated Local Site"})
    db_session.commit()
    assert repo.get_screening_by_uuid(screening_uuid)["site"] == "Updated Local Site"
    assert repo.get_screening_full("NONEXISTENT") is None
