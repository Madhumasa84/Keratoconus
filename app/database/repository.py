"""
KERASCAN Phase 2 — ScreeningRepository.

Thread-safe SQLAlchemy 2.0 repository for all screening workflow persistence.
All public methods accept plain dicts and return plain dicts.

Usage::

    from app.database import SessionLocal, init_db
    from app.database.repository import ScreeningRepository

    init_db()
    with SessionLocal() as session:
        repo = ScreeningRepository(session)
        sid = repo.save_screening({...})
        session.commit()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from .models import (
    AuditLog, Decision, Eye, ImageAnalysis,
    Measurement, PentacamFollowup, Referral, Screening,
)

log = logging.getLogger(__name__)


def _row_to_dict(obj: Any) -> dict:
    result: dict = {}
    for col in obj.__table__.columns:
        value = getattr(obj, col.name)
        if isinstance(value, datetime):
            value = value.isoformat()
        result[col.name] = value
    return result


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ScreeningRepository:
    """All database access for the KERASCAN screening workflow."""

    def __init__(self, session: Session) -> None:
        self._s = session

    # ------------------------------------------------------------------
    # Screening
    # ------------------------------------------------------------------

    def save_screening(self, screening_data: dict) -> str:
        """Persist a new screening record. Returns internal UUID id."""
        data = {k: v for k, v in screening_data.items()
                if k not in ("id", "created_at", "updated_at") and hasattr(Screening, k)}
        row = Screening(**data)
        self._s.add(row)
        self._s.flush()
        self.log_audit("screenings", row.id, "INSERT", None, _row_to_dict(row), "system")
        log.debug("save_screening: id=%s screening_id=%s", row.id, row.screening_id)
        return row.id

    def update_screening(self, screening_uuid: str, update_data: dict) -> bool:
        """Update an existing screening record. Returns True if found."""
        stmt = select(Screening).where(Screening.id == screening_uuid)
        row = self._s.scalars(stmt).first()
        if row is None:
            return False
        old = _row_to_dict(row)
        for k, v in update_data.items():
            if hasattr(row, k) and k not in ("id", "created_at"):
                setattr(row, k, v)
        self._s.flush()
        self.log_audit("screenings", row.id, "UPDATE", old, _row_to_dict(row), "system")
        return True

    # ------------------------------------------------------------------
    # Eye
    # ------------------------------------------------------------------

    def save_eye(self, eye_data: dict) -> str:
        """Persist a per-eye record. Returns internal UUID id."""
        data = {k: v for k, v in eye_data.items()
                if k not in ("id", "created_at") and hasattr(Eye, k)}
        row = Eye(**data)
        self._s.add(row)
        self._s.flush()
        self.log_audit("eyes", row.id, "INSERT", None, _row_to_dict(row), "system")
        log.debug("save_eye: id=%s laterality=%s", row.id, row.laterality)
        return row.id

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    def save_measurements(self, measurement_list: list[dict]) -> list[str]:
        """Persist one or more measurement readings. Returns list of new ids."""
        ids: list[str] = []
        for mdata in measurement_list:
            data = {k: v for k, v in mdata.items()
                    if k not in ("id", "created_at") and hasattr(Measurement, k)}
            row = Measurement(**data)
            self._s.add(row)
            self._s.flush()
            self.log_audit("measurements", row.id, "INSERT", None, _row_to_dict(row), "system")
            ids.append(row.id)
        log.debug("save_measurements: %d rows", len(ids))
        return ids

    # ------------------------------------------------------------------
    # ImageAnalysis
    # ------------------------------------------------------------------

    def save_image_analysis(self, analysis_data: dict) -> str:
        """Store the full engine output for one eye. Returns new id."""
        data = dict(analysis_data)
        data.pop("id", None)
        data.pop("created_at", None)

        engine_result: dict = data.get("engine_result") or {}
        # Strip non-serialisable numpy artifacts
        if "_artifacts" in engine_result:
            engine_result = {k: v for k, v in engine_result.items() if k != "_artifacts"}
            data["engine_result"] = engine_result

        # Backfill convenience columns from engine_result if not provided
        defaults = {
            "screening_result": engine_result.get("screening_result"),
            "prototype_score": engine_result.get("prototype_score"),
            "classification_skipped": engine_result.get("classification_skipped"),
            "pipeline_version": engine_result.get("pipeline_version"),
            "model_hash": (engine_result.get("model") or {}).get("model_hash"),
        }
        for k, v in defaults.items():
            data.setdefault(k, v)

        filtered = {k: v for k, v in data.items() if hasattr(ImageAnalysis, k)}
        row = ImageAnalysis(**filtered)
        self._s.add(row)
        self._s.flush()
        self.log_audit("image_analysis", row.id, "INSERT", None, _row_to_dict(row), "system")
        log.debug("save_image_analysis: id=%s result=%s", row.id, row.screening_result)
        return row.id

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def save_decision(self, decision_data: dict) -> str:
        """Persist an automated decision record. Returns new id."""
        data = {k: v for k, v in decision_data.items()
                if k not in ("id", "created_at") and hasattr(Decision, k)}
        row = Decision(**data)
        self._s.add(row)
        self._s.flush()
        self.log_audit("decisions", row.id, "INSERT", None, _row_to_dict(row), "system")
        log.debug("save_decision: id=%s level=%s result=%s", row.id, row.decision_level, row.final_result)
        return row.id

    def update_decision_override(self, decision_id: str, override_data: dict) -> bool:
        """Apply a clinician override. Original decision is preserved. Returns True if found."""
        stmt = select(Decision).where(Decision.id == decision_id)
        row = self._s.scalars(stmt).first()
        if row is None:
            log.warning("update_decision_override: decision_id=%s not found", decision_id)
            return False
        old = _row_to_dict(row)
        row.is_overridden = True
        row.override_original = row.final_result
        row.override_new = override_data.get("override_new")
        row.override_by = override_data.get("override_by")
        row.override_at = _now_utc()
        row.override_reason = override_data.get("override_reason")
        row.final_result = row.override_new
        self._s.flush()
        self.log_audit("decisions", row.id, "UPDATE", old, _row_to_dict(row), override_data.get("override_by", "unknown"))
        log.debug("update_decision_override: %s -> %s", decision_id, row.override_new)
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_screening(self, screening_id: str) -> dict | None:
        """Get screening by human screening_id. Returns None if not found."""
        stmt = select(Screening).where(Screening.screening_id == screening_id)
        row = self._s.scalars(stmt).first()
        return _row_to_dict(row) if row else None

    def get_screening_by_uuid(self, uuid: str) -> dict | None:
        """Get screening by internal UUID. Returns None if not found."""
        stmt = select(Screening).where(Screening.id == uuid)
        row = self._s.scalars(stmt).first()
        return _row_to_dict(row) if row else None

    def get_screening_full(self, screening_id: str) -> dict | None:
        """Return screening with all related eyes, measurements, analyses, decisions."""
        stmt = select(Screening).where(Screening.screening_id == screening_id)
        screening_row = self._s.scalars(stmt).first()
        if screening_row is None:
            return None

        result = _row_to_dict(screening_row)
        eyes_out: list[dict] = []
        for eye in screening_row.eyes:
            eye_dict = _row_to_dict(eye)
            eye_dict["measurements"] = [_row_to_dict(m) for m in eye.measurements]
            eye_dict["image_analyses"] = [_row_to_dict(a) for a in eye.image_analyses]
            eye_dict["decisions"] = [_row_to_dict(d) for d in eye.decisions if d.decision_level == "eye"]
            eyes_out.append(eye_dict)

        result["eyes"] = eyes_out
        result["decisions"] = [_row_to_dict(d) for d in screening_row.decisions if d.decision_level == "child"]
        result["referrals"] = [_row_to_dict(r) for r in screening_row.referrals]
        result["pentacam_followups"] = [_row_to_dict(p) for p in screening_row.pentacam_followups]
        return result

    def list_screenings(self, limit: int = 50, offset: int = 0, site: str | None = None,
                        date_from: str | None = None, date_to: str | None = None) -> list[dict]:
        """Paginated screening list, optionally filtered by site and date."""
        stmt = select(Screening)
        filters = []
        if site:
            filters.append(Screening.site == site)
        if date_from:
            filters.append(Screening.screening_date >= date_from)
        if date_to:
            filters.append(Screening.screening_date <= date_to)
        if filters:
            stmt = stmt.where(and_(*filters))
        stmt = stmt.order_by(Screening.created_at.desc()).offset(offset).limit(limit)
        return [_row_to_dict(r) for r in self._s.scalars(stmt).all()]

    def search_screenings(self, query: str) -> list[dict]:
        """Search across screening_id, site, operator_id using LIKE."""
        pattern = f"%{query}%"
        stmt = (
            select(Screening)
            .where(or_(
                Screening.screening_id.like(pattern),
                Screening.site.like(pattern),
                Screening.operator_id.like(pattern),
            ))
            .order_by(Screening.created_at.desc())
            .limit(200)
        )
        return [_row_to_dict(r) for r in self._s.scalars(stmt).all()]

    def screening_id_exists(self, screening_id: str) -> bool:
        """Check if a human screening_id already exists."""
        stmt = select(Screening.id).where(Screening.screening_id == screening_id)
        return self._s.scalars(stmt).first() is not None

    # ------------------------------------------------------------------
    # Referral
    # ------------------------------------------------------------------

    def save_referral(self, referral_data: dict) -> str:
        """Persist a referral record. Returns new id."""
        data = {k: v for k, v in referral_data.items()
                if k not in ("id", "created_at") and hasattr(Referral, k)}
        row = Referral(**data)
        self._s.add(row)
        self._s.flush()
        self.log_audit("referrals", row.id, "INSERT", None, _row_to_dict(row), "system")
        log.debug("save_referral: id=%s", row.id)
        return row.id

    # ------------------------------------------------------------------
    # Pentacam follow-up
    # ------------------------------------------------------------------

    def save_pentacam_followup(self, followup_data: dict) -> str:
        """Persist a Pentacam follow-up record. Returns new id."""
        data = {k: v for k, v in followup_data.items()
                if k not in ("id", "created_at") and hasattr(PentacamFollowup, k)}
        row = PentacamFollowup(**data)
        self._s.add(row)
        self._s.flush()
        self.log_audit("pentacam_followup", row.id, "INSERT", None, _row_to_dict(row), "system")
        log.debug("save_pentacam_followup: id=%s", row.id)
        return row.id

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def log_audit(self, table_name: str, record_id: str, action: str,
                  old_value: dict | None, new_value: dict | None, performed_by: str | None) -> None:
        """Write one immutable audit log entry."""
        row = AuditLog(
            table_name=table_name,
            record_id=record_id,
            action=action,
            old_value=old_value,
            new_value=new_value,
            performed_by=performed_by,
        )
        self._s.add(row)

    def get_audit_log(self, record_id: str) -> list[dict]:
        """Return all audit entries for a record, oldest first."""
        stmt = select(AuditLog).where(AuditLog.record_id == record_id).order_by(AuditLog.performed_at.asc())
        return [_row_to_dict(r) for r in self._s.scalars(stmt).all()]
