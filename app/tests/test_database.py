"""Unit tests for the database layer."""
import os
import pytest


def test_init_db_creates_tables(tmp_db):
    """init_db creates all expected tables."""
    import sqlite3
    conn = sqlite3.connect(tmp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    required = {"screenings", "eyes", "measurements", "image_analysis",
                "decisions", "referrals", "pentacam_followup", "audit_log"}
    assert required.issubset(tables), f"Missing tables: {required - tables}"


def test_save_and_retrieve_screening(db_session, sample_screening_data):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    uuid = repo.save_screening(sample_screening_data)
    db_session.commit()
    assert uuid
    row = repo.get_screening(sample_screening_data["screening_id"])
    assert row is not None
    assert row["screening_id"] == sample_screening_data["screening_id"]
    assert row["age"] == sample_screening_data["age"]


def test_save_and_retrieve_eye(db_session, sample_screening_data):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    screening_uuid = repo.save_screening(sample_screening_data)
    eye_uuid = repo.save_eye({
        "screening_id": screening_uuid,
        "laterality": "OD",
        "eye_result": "NORMAL-LIKE",
        "quality_gradable": True,
        "quality_score": 72.0,
    })
    db_session.commit()
    assert eye_uuid


def test_save_and_retrieve_measurements(db_session, sample_screening_data, sample_measurements_od):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    s_uuid = repo.save_screening(sample_screening_data)
    e_uuid = repo.save_eye({"screening_id": s_uuid, "laterality": "OD"})
    meas_ids = repo.save_measurements([dict(sample_measurements_od, eye_id=e_uuid)])
    db_session.commit()
    assert len(meas_ids) == 1


def test_data_persists_after_session_restart(tmp_db, sample_screening_data):
    """Data saved in one session is readable in a new session."""
    import importlib
    import app.database as db_module
    importlib.reload(db_module)
    db_module.init_db()

    from app.database import SessionLocal
    with SessionLocal() as s1:
        from app.database.repository import ScreeningRepository
        repo = ScreeningRepository(s1)
        repo.save_screening(sample_screening_data)
        s1.commit()

    with SessionLocal() as s2:
        repo2 = ScreeningRepository(s2)
        row = repo2.get_screening(sample_screening_data["screening_id"])
        assert row is not None
        assert row["screening_id"] == sample_screening_data["screening_id"]


def test_duplicate_screening_id_detected(db_session, sample_screening_data):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    repo.save_screening(sample_screening_data)
    db_session.commit()
    assert repo.screening_id_exists(sample_screening_data["screening_id"]) is True


def test_audit_log_entry_created_on_save(db_session, sample_screening_data):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    uuid = repo.save_screening(sample_screening_data)
    db_session.commit()
    audit = repo.get_audit_log(uuid)
    assert len(audit) >= 1
    assert audit[0]["action"] == "INSERT"


def test_override_preserves_original_decision(db_session, sample_screening_data):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    s_uuid = repo.save_screening(sample_screening_data)
    decision_id = repo.save_decision({
        "screening_id": s_uuid,
        "decision_level": "child",
        "automated_result": "SCREEN_NEGATIVE",
        "final_result": "SCREEN_NEGATIVE",
    })
    db_session.commit()

    repo.update_decision_override(decision_id, {
        "override_new": "STANDARD_REFERRAL",
        "override_by": "clinician01",
        "override_reason": "Clinician observed corneal ectasia signs not captured by algorithm.",
    })
    db_session.commit()

    # Verify override stored and original preserved
    from sqlalchemy import select
    from app.database.models import Decision
    stmt = select(Decision).where(Decision.id == decision_id)
    row = db_session.scalars(stmt).first()
    assert row.is_overridden is True
    assert row.override_original == "SCREEN_NEGATIVE"   # original preserved
    assert row.final_result == "STANDARD_REFERRAL"       # updated
    assert row.override_by == "clinician01"


def test_search_screenings_by_id(db_session, sample_screening_data):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    repo.save_screening(sample_screening_data)
    db_session.commit()
    results = repo.search_screenings("TEST-001")
    assert any(r["screening_id"] == "TEST-001" for r in results)


def test_search_screenings_by_site(db_session, sample_screening_data):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    repo.save_screening(sample_screening_data)
    db_session.commit()
    results = repo.search_screenings("Test School")
    assert len(results) >= 1
