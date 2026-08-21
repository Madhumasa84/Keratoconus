"""
audit_service.py — Immutable audit trail for all decisions and overrides.

Original automated decisions are NEVER deleted or modified.
Every override creates a new audit entry while preserving the original.
"""
from __future__ import annotations

import csv
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_OVERRIDE_FIELDS = ("user_identity", "reason", "original_decision", "new_decision")
MIN_REASON_LENGTH = 20


class AuditService:
    """Provides audit logging and override validation for KERASCAN screenings."""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        table_name: str,
        record_id: str,
        action: str,
        old_value: dict | None,
        new_value: dict | None,
        performed_by: str,
        session,
    ) -> str:
        """
        Log an audit event to the audit_log table.

        Parameters
        ----------
        table_name   : database table being audited
        record_id    : UUID of the affected record
        action       : 'INSERT' | 'UPDATE' | 'DELETE'
        old_value    : previous state dict (None for INSERT)
        new_value    : new state dict (None for DELETE)
        performed_by : operator or system identifier
        session      : SQLAlchemy session

        Returns
        -------
        str : new audit log entry id
        """
        from app.database.models import AuditLog

        entry = AuditLog(
            table_name=table_name,
            record_id=record_id,
            action=action,
            old_value=old_value,
            new_value=new_value,
            performed_by=performed_by,
        )
        session.add(entry)
        session.flush()
        log.debug("audit: %s on %s.%s by %s", action, table_name, record_id, performed_by)
        return entry.id

    def get_audit_trail(self, record_id: str, session) -> list[dict]:
        """
        Return full audit trail for a record, ordered chronologically.
        """
        from sqlalchemy import select
        from app.database.models import AuditLog

        stmt = (
            select(AuditLog)
            .where(AuditLog.record_id == record_id)
            .order_by(AuditLog.performed_at.asc())
        )
        rows = session.scalars(stmt).all()
        result = []
        for row in rows:
            d = {}
            for col in row.__table__.columns:
                val = getattr(row, col.name)
                if isinstance(val, datetime):
                    val = val.isoformat()
                d[col.name] = val
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Override
    # ------------------------------------------------------------------

    def validate_override_request(self, override_data: dict) -> tuple[bool, list[str]]:
        """
        Validate a decision override request.

        Required fields:
          - user_identity   (non-empty)
          - reason          (non-empty, >= 20 chars)
          - original_decision
          - new_decision    (must differ from original)
          - timestamp

        Returns (is_valid, list_of_errors).
        """
        errors: list[str] = []

        user = override_data.get("user_identity", "")
        if not str(user).strip():
            errors.append("user_identity is required.")

        reason = override_data.get("reason", "")
        if not str(reason).strip():
            errors.append("A mandatory override reason is required.")
        elif len(str(reason).strip()) < MIN_REASON_LENGTH:
            errors.append(f"Override reason must be at least {MIN_REASON_LENGTH} characters.")

        original = override_data.get("original_decision")
        new = override_data.get("new_decision")

        if not original:
            errors.append("original_decision is required.")
        if not new:
            errors.append("new_decision is required.")
        if original and new and original == new:
            errors.append("new_decision must differ from original_decision.")

        if not override_data.get("timestamp"):
            errors.append("timestamp is required.")

        return len(errors) == 0, errors

    def log_override(
        self,
        decision_id: str,
        original_result: str,
        new_result: str,
        reason: str,
        performed_by: str,
        timestamp: datetime,
        session,
    ) -> str:
        """
        Log a decision override. Original automated decision is preserved.
        Updates the Decision row and creates an audit entry.

        Returns the audit log entry id.
        """
        from app.database.repository import ScreeningRepository

        repo = ScreeningRepository(session)
        repo.update_decision_override(decision_id, {
            "override_new": new_result,
            "override_by": performed_by,
            "override_reason": reason,
        })

        audit_id = self.log_event(
            table_name="decisions",
            record_id=decision_id,
            action="UPDATE",
            old_value={"final_result": original_result},
            new_value={
                "final_result": new_result,
                "override_by": performed_by,
                "override_at": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                "override_reason": reason,
                "original_automated_result_preserved": original_result,
            },
            performed_by=performed_by,
            session=session,
        )
        log.info(
            "log_override: decision_id=%s %s -> %s by %s",
            decision_id, original_result, new_result, performed_by
        )
        return audit_id

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_audit_log(self, screening_id: str, output_path: str, session) -> str:
        """
        Export the full audit log for a screening as a CSV file.

        Parameters
        ----------
        screening_id : the human-readable screening ID
        output_path  : target file path (will be created / overwritten)
        session      : SQLAlchemy session

        Returns
        -------
        str : absolute path to the generated CSV file
        """
        from sqlalchemy import select
        from app.database.models import AuditLog, Screening, Eye

        # Resolve screening UUID
        stmt = select(Screening).where(Screening.screening_id == screening_id)
        screening_row = session.scalars(stmt).first()

        if screening_row is None:
            raise ValueError(f"Screening '{screening_id}' not found.")

        # Collect all record IDs for this screening (screening + eyes)
        record_ids = {screening_row.id}
        for eye in screening_row.eyes:
            record_ids.add(eye.id)
            for m in eye.measurements:
                record_ids.add(m.id)
            for ia in eye.image_analyses:
                record_ids.add(ia.id)
            for d in eye.decisions:
                record_ids.add(d.id)

        stmt = (
            select(AuditLog)
            .where(AuditLog.record_id.in_(record_ids))
            .order_by(AuditLog.performed_at.asc())
        )
        rows = session.scalars(stmt).all()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["id", "table_name", "record_id", "action",
                      "performed_by", "performed_at", "old_value", "new_value"]

        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                import json
                writer.writerow({
                    "id": row.id,
                    "table_name": row.table_name,
                    "record_id": row.record_id,
                    "action": row.action,
                    "performed_by": row.performed_by or "",
                    "performed_at": row.performed_at.isoformat() if isinstance(row.performed_at, datetime) else str(row.performed_at),
                    "old_value": json.dumps(row.old_value) if row.old_value else "",
                    "new_value": json.dumps(row.new_value) if row.new_value else "",
                })

        log.info("export_audit_log: wrote %d entries to %s", len(rows), output)
        return str(output.resolve())
