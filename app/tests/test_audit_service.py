"""Unit tests for AuditService."""
import pytest
from datetime import datetime, timezone
from app.services.audit_service import AuditService


@pytest.fixture
def audit_svc():
    return AuditService()


def test_override_requires_user_identity(audit_svc):
    valid, errors = audit_svc.validate_override_request({
        "user_identity": "",
        "reason": "Clinical examination showed clear ectasia pattern not captured.",
        "original_decision": "SCREEN_NEGATIVE",
        "new_decision": "STANDARD_REFERRAL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert valid is False
    assert any("user_identity" in e.lower() or "identity" in e.lower() for e in errors)


def test_override_requires_mandatory_reason(audit_svc):
    valid, errors = audit_svc.validate_override_request({
        "user_identity": "dr_smith",
        "reason": "Too short",
        "original_decision": "SCREEN_NEGATIVE",
        "new_decision": "STANDARD_REFERRAL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert valid is False
    assert any("reason" in e.lower() or "character" in e.lower() for e in errors)


def test_override_decision_must_differ(audit_svc):
    valid, errors = audit_svc.validate_override_request({
        "user_identity": "dr_smith",
        "reason": "Clinical signs confirmed on slit-lamp examination today.",
        "original_decision": "SCREEN_NEGATIVE",
        "new_decision": "SCREEN_NEGATIVE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert valid is False
    assert any("differ" in e.lower() for e in errors)


def test_override_requires_timestamp(audit_svc):
    valid, errors = audit_svc.validate_override_request({
        "user_identity": "dr_smith",
        "reason": "Clinical signs confirmed on slit-lamp examination today.",
        "original_decision": "SCREEN_NEGATIVE",
        "new_decision": "STANDARD_REFERRAL",
        "timestamp": None,
    })
    assert valid is False
    assert any("timestamp" in e.lower() for e in errors)


def test_valid_override_passes(audit_svc):
    valid, errors = audit_svc.validate_override_request({
        "user_identity": "dr_smith",
        "reason": "Vogt striae and Fleischer ring observed on slit-lamp. Clinical KC suspected.",
        "original_decision": "SCREEN_NEGATIVE",
        "new_decision": "STANDARD_REFERRAL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert valid is True
    assert errors == []


def test_override_recorded_in_db(db_session, sample_screening_data, audit_svc):
    """Override creates audit entry; original automated result preserved."""
    from app.database.repository import ScreeningRepository
    from sqlalchemy import select
    from app.database.models import Decision

    repo = ScreeningRepository(db_session)
    s_uuid = repo.save_screening(sample_screening_data)
    decision_id = repo.save_decision({
        "screening_id": s_uuid,
        "decision_level": "child",
        "automated_result": "SCREEN_NEGATIVE",
        "final_result": "SCREEN_NEGATIVE",
    })
    db_session.commit()

    audit_svc.log_override(
        decision_id=decision_id,
        original_result="SCREEN_NEGATIVE",
        new_result="STANDARD_REFERRAL",
        reason="Vogt striae and Fleischer ring observed on slit-lamp examination.",
        performed_by="dr_jones",
        timestamp=datetime.now(timezone.utc),
        session=db_session,
    )
    db_session.commit()

    # Original automated decision preserved
    stmt = select(Decision).where(Decision.id == decision_id)
    dec = db_session.scalars(stmt).first()
    assert dec.automated_result == "SCREEN_NEGATIVE"
    assert dec.override_original == "SCREEN_NEGATIVE"
    assert dec.final_result == "STANDARD_REFERRAL"
    assert dec.is_overridden is True

    # Audit trail exists
    audit = repo.get_audit_log(decision_id)
    assert len(audit) >= 2  # INSERT + UPDATE
    actions = {e["action"] for e in audit}
    assert "UPDATE" in actions


def test_audit_log_contains_timestamp(db_session, sample_screening_data):
    from app.database.repository import ScreeningRepository
    repo = ScreeningRepository(db_session)
    uuid = repo.save_screening(sample_screening_data)
    db_session.commit()
    audit = repo.get_audit_log(uuid)
    assert len(audit) >= 1
    for entry in audit:
        assert entry.get("performed_at") is not None
