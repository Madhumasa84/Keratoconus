"""
KERASCAN Phase 2 — SQLAlchemy ORM models (SQLite, SQLAlchemy 2.0 style).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint, types,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class JSONText(types.TypeDecorator):
    """Stores a Python dict/list as a JSON TEXT column."""
    impl = types.Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, default=float)

    def process_result_value(self, value: str | None, dialect: Any) -> Any:
        if value is None:
            return None
        return json.loads(value)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class OperatorAccount(Base):
    """Local role account; password hashes only, never a network identity provider."""
    __tablename__ = "operator_accounts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    operator_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="operator")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"
    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)


class Screening(Base):
    __tablename__ = "screenings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    screening_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(16), nullable=True)
    site: Mapped[str | None] = mapped_column(String(128), nullable=True)
    screening_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    operator_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    device_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    consent_recorded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    overall_result: Mapped[str | None] = mapped_column(String(32), nullable=True)
    overall_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    affected_eyes: Mapped[list | None] = mapped_column(JSONText, nullable=True)
    referral_priority: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    json_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    excel_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # ``pdf_path`` remains a local operational field for pre-existing records;
    # new reports are audited by hash and never exported with a filesystem path.
    pdf_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pdf_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_identifier: Mapped[str | None] = mapped_column(String(64), nullable=True)
    software_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    protocol_version: Mapped[str | None] = mapped_column(String(96), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc, onupdate=_now_utc)

    eyes: Mapped[list[Eye]] = relationship("Eye", back_populates="screening", cascade="all, delete-orphan")
    decisions: Mapped[list[Decision]] = relationship("Decision", back_populates="screening", cascade="all, delete-orphan")
    referrals: Mapped[list[Referral]] = relationship("Referral", back_populates="screening", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Screening screening_id={self.screening_id!r} result={self.overall_result!r}>"


class Eye(Base):
    __tablename__ = "eyes"
    __table_args__ = (
        CheckConstraint("laterality IN ('OD', 'OS')", name="ck_eye_laterality"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    screening_id: Mapped[str] = mapped_column(String(36), ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False, index=True)
    laterality: Mapped[str] = mapped_column(String(4), nullable=False)
    eye_result: Mapped[str | None] = mapped_column(String(32), nullable=True)
    kerascan_image_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_status: Mapped[str | None] = mapped_column(String(48), nullable=True)
    image_failure_stage: Mapped[str | None] = mapped_column(String(48), nullable=True)
    image_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_codes: Mapped[list | None] = mapped_column(JSONText, nullable=True)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processed_image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processed_output_hashes: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    analysis_artifacts: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    analysis_provenance_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    geometry_validation_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    roi_box: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    roi_center: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    roi_radius: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quality_gradable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_flags: Mapped[list | None] = mapped_column(JSONText, nullable=True)
    quality_metrics: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    features: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    protocol_version: Mapped[str | None] = mapped_column(String(96), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)

    screening: Mapped[Screening] = relationship("Screening", back_populates="eyes")
    measurements: Mapped[list[Measurement]] = relationship("Measurement", back_populates="eye", cascade="all, delete-orphan")
    image_analyses: Mapped[list[ImageAnalysis]] = relationship("ImageAnalysis", back_populates="eye", cascade="all, delete-orphan")
    decisions: Mapped[list[Decision]] = relationship("Decision", back_populates="eye", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Eye laterality={self.laterality!r} result={self.eye_result!r}>"


class Measurement(Base):
    __tablename__ = "measurements"
    __table_args__ = (
        CheckConstraint("pachymetry_type IN ('central', 'thinnest') OR pachymetry_type IS NULL", name="ck_measurement_pachymetry_type"),
        CheckConstraint("refraction_type IN ('autorefraction', 'subjective') OR refraction_type IS NULL", name="ck_measurement_refraction_type"),
        CheckConstraint("reading_number IN (1, 2, 3) OR reading_number IS NULL", name="ck_measurement_reading_number"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    eye_id: Mapped[str] = mapped_column(String(36), ForeignKey("eyes.id", ondelete="CASCADE"), nullable=False, index=True)
    k1_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    k1_axis: Mapped[float | None] = mapped_column(Float, nullable=True)
    k2_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    k2_axis: Mapped[float | None] = mapped_column(Float, nullable=True)
    kmax_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_k_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    pachymetry_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    # New screening rows use this protocol-level designation. The legacy
    # ``pachymetry_type`` remains untouched for historical records.
    pachymetry_measurement_type: Mapped[str | None] = mapped_column(String(24), nullable=True)
    pachymetry_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    sphere_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    cylinder_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    cylinder_axis: Mapped[float | None] = mapped_column(Float, nullable=True)
    va_logmar: Mapped[float | None] = mapped_column(Float, nullable=True)
    va_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    measurement_quality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reading_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    refraction_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    clinical_flags: Mapped[list | None] = mapped_column(JSONText, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)

    eye: Mapped[Eye] = relationship("Eye", back_populates="measurements")

    def __repr__(self) -> str:
        return f"<Measurement eye_id={self.eye_id!r} k2_d={self.k2_d} reading={self.reading_number}>"


class ImageAnalysis(Base):
    __tablename__ = "image_analysis"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    eye_id: Mapped[str] = mapped_column(String(36), ForeignKey("eyes.id", ondelete="CASCADE"), nullable=False, index=True)
    engine_result: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    screening_result: Mapped[str | None] = mapped_column(String(32), nullable=True)
    prototype_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    classification_skipped: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    model_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_status: Mapped[str | None] = mapped_column(String(48), nullable=True)
    failure_stage: Mapped[str | None] = mapped_column(String(48), nullable=True)
    geometry_validation_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    original_image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processed_output_hashes: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    artifact_manifest: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    provenance_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)

    eye: Mapped[Eye] = relationship("Eye", back_populates="image_analyses")

    def __repr__(self) -> str:
        return f"<ImageAnalysis screening_result={self.screening_result!r} score={self.prototype_score}>"


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = (
        CheckConstraint("decision_level IN ('eye', 'child')", name="ck_decision_level"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    eye_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("eyes.id", ondelete="SET NULL"), nullable=True, index=True)
    screening_id: Mapped[str] = mapped_column(String(36), ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False, index=True)
    decision_level: Mapped[str] = mapped_column(String(8), nullable=False)
    automated_result: Mapped[str | None] = mapped_column(String(32), nullable=True)
    automated_reason_codes: Mapped[list | None] = mapped_column(JSONText, nullable=True)
    final_result: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_overridden: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    override_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    override_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    override_original: Mapped[str | None] = mapped_column(String(32), nullable=True)
    override_new: Mapped[str | None] = mapped_column(String(32), nullable=True)
    override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    protocol_version: Mapped[str | None] = mapped_column(String(96), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)

    eye: Mapped[Eye | None] = relationship("Eye", back_populates="decisions")
    screening: Mapped[Screening] = relationship("Screening", back_populates="decisions")

    def __repr__(self) -> str:
        return f"<Decision level={self.decision_level!r} final={self.final_result!r} overridden={self.is_overridden}>"


class Referral(Base):
    __tablename__ = "referrals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    screening_id: Mapped[str] = mapped_column(String(36), ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False, index=True)
    referral_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    referral_destination: Mapped[str | None] = mapped_column(Text, nullable=True)
    referral_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    referral_priority: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)

    screening: Mapped[Screening] = relationship("Screening", back_populates="referrals")

    def __repr__(self) -> str:
        return f"<Referral screening_id={self.screening_id!r} priority={self.referral_priority!r}>"


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        CheckConstraint("action IN ('INSERT', 'UPDATE', 'DELETE')", name="ck_audit_action"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    table_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    record_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(8), nullable=False)
    old_value: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    new_value: Mapped[dict | None] = mapped_column(JSONText, nullable=True)
    performed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    performed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now_utc)

    def __repr__(self) -> str:
        return f"<AuditLog table={self.table_name!r} record={self.record_id!r} action={self.action!r}>"
