"""
screening_service.py — Orchestrates the complete screening workflow.

Responsibilities:
 1. Validate screening form and measurement inputs
 2. Call KerascanEngine on OD and OS images
 3. Apply ReferralEngine rules
 4. Persist all data via ScreeningRepository
 5. Return a complete ScreeningResult
"""
from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# Phase 1 engine — import via its public interface
_PHASE1_SRC = Path(__file__).parent.parent.parent / "kerascan" / "src"
if str(_PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(_PHASE1_SRC))

try:
    from kerascan import EngineConfig, KerascanEngine
    _ENGINE_AVAILABLE = True
except ImportError as e:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(e)

from .referral_engine import ReferralEngine, EyeReferralResult, ChildReferralResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

K_MIN, K_MAX = 30.0, 70.0          # dioptres
PACHY_MIN, PACHY_MAX = 200.0, 800.0  # µm
CYL_MIN, CYL_MAX = 0.0, 12.0       # dioptres (magnitude)
CYL_AXIS_MIN, CYL_AXIS_MAX = 0, 180  # degrees
SPHERE_MIN, SPHERE_MAX = -30.0, 20.0  # dioptres
AGE_MIN, AGE_MAX = 5, 25
K2_AGREEMENT_THRESHOLD = 0.5       # D — if readings disagree more than this, request 3rd


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScreeningResult:
    screening_id: str
    screening_uuid: str
    od_eye_result: EyeReferralResult | None
    os_eye_result: EyeReferralResult | None
    child_result: ChildReferralResult | None
    od_engine_raw: dict | None
    os_engine_raw: dict | None
    validation_errors: list[str] = field(default_factory=list)
    success: bool = True
    error_message: str = ""
    pdf_path: str = ""
    json_path: str = ""
    excel_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ScreeningService:
    """
    Orchestrates the full keratoconus screening workflow.

    Parameters
    ----------
    db_session  : open SQLAlchemy session (caller manages lifecycle)
    engine_config : optional EngineConfig override
    protocol_path : optional path to referral_protocol.yaml
    """

    def __init__(self, db_session=None, engine_config=None, protocol_path=None):
        self._session = db_session
        self._engine_config = engine_config or (EngineConfig() if _ENGINE_AVAILABLE else None)
        self._referral_engine = ReferralEngine(protocol_path)
        self._image_engine: Any = None  # lazy-initialised

    def _get_image_engine(self) -> Any:
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(f"Phase 1 engine not available: {_ENGINE_IMPORT_ERROR}")
        if self._image_engine is None:
            self._image_engine = KerascanEngine(self._engine_config)
        return self._image_engine

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_screening_form(self, form_data: dict) -> tuple[bool, list[str]]:
        """
        Validate top-level screening form fields.
        Returns (is_valid, list_of_error_messages).
        """
        errors: list[str] = []

        # Required text fields
        for field_name in ("screening_id", "operator_id", "device_id", "site"):
            val = form_data.get(field_name)
            if not val or not str(val).strip():
                errors.append(f"'{field_name}' is required.")

        # screening_id alphanumeric
        sid = str(form_data.get("screening_id", "")).strip()
        if sid and not sid.replace("-", "").replace("_", "").isalnum():
            errors.append("screening_id must be alphanumeric (hyphens and underscores allowed).")

        # Age
        age = form_data.get("age")
        if age is None:
            errors.append("'age' is required.")
        else:
            try:
                age_i = int(age)
                if not (AGE_MIN <= age_i <= AGE_MAX):
                    errors.append(f"Age must be between {AGE_MIN} and {AGE_MAX}.")
            except (TypeError, ValueError):
                errors.append("'age' must be an integer.")

        # Sex
        sex = form_data.get("sex")
        if sex not in ("Male", "Female", "Not recorded"):
            errors.append("'sex' must be one of: Male, Female, Not recorded.")

        # Consent
        if not form_data.get("consent_recorded"):
            errors.append("Consent must be recorded before proceeding.")

        # Screening date
        sd = form_data.get("screening_date")
        if not sd:
            errors.append("'screening_date' is required.")

        return len(errors) == 0, errors

    def validate_measurements(self, measurements: dict) -> tuple[bool, list[str]]:
        """
        Validate measurement numeric ranges and type constraints.
        Returns (is_valid, list_of_error_messages).
        """
        errors: list[str] = []

        def _check_range(name, val, lo, hi, unit=""):
            if val is not None:
                try:
                    v = float(val)
                    if not (lo <= v <= hi):
                        errors.append(f"{name} {v} is outside valid range [{lo}, {hi}]{' ' + unit if unit else ''}.")
                except (TypeError, ValueError):
                    errors.append(f"{name} must be a number.")

        # Keratometry — distinguish K2/steep K from Kmax and mean K
        _check_range("K1", measurements.get("k1_d"), K_MIN, K_MAX, "D")
        _check_range("K2 (steep K)", measurements.get("k2_d"), K_MIN, K_MAX, "D")
        _check_range("Kmax", measurements.get("kmax_d"), K_MIN, K_MAX, "D")
        _check_range("Mean K", measurements.get("mean_k_d"), K_MIN, K_MAX, "D")

        for axis_field, label in (("k1_axis", "K1 axis"), ("k2_axis", "K2 axis")):
            ax = measurements.get(axis_field)
            if ax is not None:
                _check_range(label, ax, 0, 180, "°")

        # Pachymetry — distinguish central from thinnest (stored separately)
        pachy = measurements.get("pachymetry_um")
        _check_range("Pachymetry", pachy, PACHY_MIN, PACHY_MAX, "µm")
        pachy_type = measurements.get("pachymetry_type")
        if pachy is not None and pachy_type not in ("central", "thinnest"):
            errors.append("pachymetry_type must be 'central' or 'thinnest'.")

        # Refraction — distinguish autorefraction from subjective
        ref_type = measurements.get("refraction_type")
        if ref_type is not None and ref_type not in ("autorefraction", "subjective"):
            errors.append("refraction_type must be 'autorefraction' or 'subjective'.")

        # Cylinder magnitude and axis
        cyl = measurements.get("cylinder_d")
        _check_range("Cylinder magnitude", cyl, CYL_MIN, CYL_MAX, "D")

        cyl_ax = measurements.get("cylinder_axis")
        if cyl_ax is not None:
            try:
                ca = float(cyl_ax)
                if not (CYL_AXIS_MIN <= ca <= CYL_AXIS_MAX):
                    errors.append(f"Cylinder axis {ca}° is outside valid range [0, 180]°.")
            except (TypeError, ValueError):
                errors.append("cylinder_axis must be a number (degrees).")

        _check_range("Sphere", measurements.get("sphere_d"), SPHERE_MIN, SPHERE_MAX, "D")

        return len(errors) == 0, errors

    def check_measurement_agreement(self, reading1: dict, reading2: dict) -> bool:
        """
        Returns True if readings agree (K2 difference <= threshold).
        Returns False if they disagree and a third reading is required.
        """
        k2_r1 = reading1.get("k2_d")
        k2_r2 = reading2.get("k2_d")
        if k2_r1 is not None and k2_r2 is not None:
            diff = abs(float(k2_r1) - float(k2_r2))
            if diff > K2_AGREEMENT_THRESHOLD:
                log.info(
                    "check_measurement_agreement: K2 readings disagree by %.2fD (threshold %.2fD) — third reading required",
                    diff, K2_AGREEMENT_THRESHOLD
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Image analysis
    # ------------------------------------------------------------------

    def _analyse_image(self, image_path: str | Path) -> dict:
        """
        Run Phase 1 engine on one eye image.
        Returns engine result dict. On failure returns UNGRADABLE-equivalent.
        """
        try:
            engine = self._get_image_engine()
            result = engine.analyze(image_path)
            # Strip numpy artifacts for serialisation
            result.pop("_artifacts", None)
            return result
        except Exception:
            log.exception("Phase 1 engine analysis failed for a local input (path redacted)")
            return {
                "screening_result": "UNGRADABLE",
                "message": "Engine error: local input could not be analysed.",
                "classification_skipped": True,
                "quality": {"gradable": False, "quality_score": 0, "flags": ["engine_error"], "metrics": {}},
                "roi": {},
                "features": {},
                "pipeline_version": "unknown",
            }

    @staticmethod
    def _image_sha256(path: str | Path) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except Exception:
            return "unavailable"
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def conduct_screening(self, screening_data: dict) -> ScreeningResult:
        """
        Run the full screening workflow.

        screening_data keys:
          form: dict (age, sex, site, etc.)
          od_image_path: str path to OD image file
          os_image_path: str path to OS image file
          od_measurements: dict (reading_number=1, k2_d, pachymetry_um, ...)
          os_measurements: dict
          od_measurements_r2: dict | None (second reading)
          os_measurements_r2: dict | None
          od_measurements_r3: dict | None (third reading, if required)
          os_measurements_r3: dict | None
        """
        from app.database.repository import ScreeningRepository
        from app.services.report_service import ReportService

        form = screening_data.get("form", {})
        od_image_path = screening_data.get("od_image_path", "")
        os_image_path = screening_data.get("os_image_path", "")
        od_meas_r1 = screening_data.get("od_measurements", {})
        os_meas_r1 = screening_data.get("os_measurements", {})
        od_meas_r2 = screening_data.get("od_measurements_r2")
        os_meas_r2 = screening_data.get("os_measurements_r2")

        # ── Form validation ──────────────────────────────────────────
        form_valid, form_errors = self.validate_screening_form(form)
        if not form_valid:
            return ScreeningResult(
                screening_id=form.get("screening_id", ""),
                screening_uuid="",
                od_eye_result=None, os_eye_result=None, child_result=None,
                od_engine_raw=None, os_engine_raw=None,
                validation_errors=form_errors,
                success=False,
                error_message="Form validation failed.",
            )

        # ── Duplicate check ──────────────────────────────────────────
        if self._session:
            repo = ScreeningRepository(self._session)
            if repo.screening_id_exists(form["screening_id"]):
                return ScreeningResult(
                    screening_id=form["screening_id"],
                    screening_uuid="",
                    od_eye_result=None, os_eye_result=None, child_result=None,
                    od_engine_raw=None, os_engine_raw=None,
                    validation_errors=[f"Screening ID '{form['screening_id']}' already exists."],
                    success=False,
                    error_message="Duplicate screening ID.",
                )
        else:
            repo = None

        # ── Measurement validation ───────────────────────────────────
        od_valid, od_errors = self.validate_measurements(od_meas_r1)
        os_valid, os_errors = self.validate_measurements(os_meas_r1)
        all_meas_errors = (
            [f"OD: {e}" for e in od_errors] +
            [f"OS: {e}" for e in os_errors]
        )
        if all_meas_errors:
            return ScreeningResult(
                screening_id=form["screening_id"],
                screening_uuid="",
                od_eye_result=None, os_eye_result=None, child_result=None,
                od_engine_raw=None, os_engine_raw=None,
                validation_errors=all_meas_errors,
                success=False,
                error_message="Measurement validation failed.",
            )

        # ── Measurement agreement / repeat logic ─────────────────────
        od_repeat_count = 1
        os_repeat_count = 1
        if od_meas_r2 and not self.check_measurement_agreement(od_meas_r1, od_meas_r2):
            log.info("OD readings disagree — third reading recommended")
            od_repeat_count = 2
        elif od_meas_r2:
            od_repeat_count = 2

        if os_meas_r2 and not self.check_measurement_agreement(os_meas_r1, os_meas_r2):
            log.info("OS readings disagree — third reading recommended")
            os_repeat_count = 2
        elif os_meas_r2:
            os_repeat_count = 2

        # Use latest (highest reading number) for threshold evaluation
        od_meas_final = od_meas_r2 if od_meas_r2 else od_meas_r1
        os_meas_final = os_meas_r2 if os_meas_r2 else os_meas_r1

        # ── Phase 1 image analysis ───────────────────────────────────
        log.info("Analysing OD image: <local path redacted>")
        od_engine_raw = self._analyse_image(od_image_path) if od_image_path else {
            "screening_result": "UNGRADABLE", "message": "No OD image provided",
            "quality": {"gradable": False, "quality_score": 0, "flags": [], "metrics": {}},
            "roi": {}, "features": {}, "pipeline_version": "unknown",
        }

        log.info("Analysing OS image: <local path redacted>")
        os_engine_raw = self._analyse_image(os_image_path) if os_image_path else {
            "screening_result": "UNGRADABLE", "message": "No OS image provided",
            "quality": {"gradable": False, "quality_score": 0, "flags": [], "metrics": {}},
            "roi": {}, "features": {}, "pipeline_version": "unknown",
        }

        # ── Referral rules ───────────────────────────────────────────
        od_eye_result = self._referral_engine.evaluate_eye(
            "OD", od_engine_raw.get("screening_result", "UNGRADABLE"),
            od_meas_final, od_repeat_count
        )
        os_eye_result = self._referral_engine.evaluate_eye(
            "OS", os_engine_raw.get("screening_result", "UNGRADABLE"),
            os_meas_final, os_repeat_count
        )
        child_result = self._referral_engine.evaluate_child(
            od_eye_result, os_eye_result, od_meas_final, os_meas_final
        )

        # ── Persist to database ──────────────────────────────────────
        screening_uuid = ""
        if repo is not None:
            try:
                screening_uuid = repo.save_screening({
                    "screening_id": form["screening_id"],
                    "age": form.get("age"),
                    "sex": form.get("sex"),
                    "site": form.get("site"),
                    "screening_date": str(form.get("screening_date", "")),
                    "operator_id": form.get("operator_id"),
                    "device_id": form.get("device_id"),
                    "consent_recorded": bool(form.get("consent_recorded")),
                    "overall_result": child_result.decision,
                    "referral_priority": child_result.referral_priority,
                    "protocol_version": self._referral_engine.get_protocol_version(),
                })

                for laterality, eye_result, engine_raw, meas_r1, meas_r2 in [
                    ("OD", od_eye_result, od_engine_raw, od_meas_r1, od_meas_r2),
                    ("OS", os_eye_result, os_engine_raw, os_meas_r1, os_meas_r2),
                ]:
                    image_path = od_image_path if laterality == "OD" else os_image_path
                    roi = engine_raw.get("roi", {})

                    eye_id = repo.save_eye({
                        "screening_id": screening_uuid,
                        "laterality": laterality,
                        "eye_result": engine_raw.get("screening_result"),
                        "reason_codes": eye_result.reason_codes,
                        "image_path": str(image_path) if image_path else None,
                        "image_hash": self._image_sha256(image_path) if image_path else None,
                        "roi_box": roi.get("box_xyxy"),
                        "roi_center": roi.get("center_full"),
                        "roi_radius": roi.get("outer_radius_px"),
                        "roi_confidence": roi.get("confidence"),
                        "roi_method": roi.get("method"),
                        "quality_gradable": engine_raw.get("quality", {}).get("gradable"),
                        "quality_score": engine_raw.get("quality", {}).get("quality_score"),
                        "quality_flags": engine_raw.get("quality", {}).get("flags", []),
                        "quality_metrics": engine_raw.get("quality", {}).get("metrics", {}),
                        "features": engine_raw.get("features", {}),
                        "pipeline_version": engine_raw.get("pipeline_version"),
                        "protocol_version": self._referral_engine.get_protocol_version(),
                    })

                    # Save all readings
                    readings = [(meas_r1, 1)]
                    if meas_r2:
                        readings.append((meas_r2, 2))
                    meas_rows = [dict(m, eye_id=eye_id, reading_number=rn) for m, rn in readings]
                    repo.save_measurements(meas_rows)

                    # Save image analysis
                    repo.save_image_analysis({"eye_id": eye_id, "engine_result": engine_raw})

                    # Save per-eye decision
                    repo.save_decision({
                        "eye_id": eye_id,
                        "screening_id": screening_uuid,
                        "decision_level": "eye",
                        "automated_result": eye_result.decision,
                        "automated_reason_codes": eye_result.reason_codes,
                        "final_result": eye_result.decision,
                        "protocol_version": self._referral_engine.get_protocol_version(),
                    })

                # Save child-level decision
                repo.save_decision({
                    "screening_id": screening_uuid,
                    "decision_level": "child",
                    "automated_result": child_result.decision,
                    "automated_reason_codes": child_result.reason_codes,
                    "final_result": child_result.decision,
                    "protocol_version": self._referral_engine.get_protocol_version(),
                })

                self._session.commit()
                log.info("conduct_screening: saved screening_id=%s uuid=%s", form["screening_id"], screening_uuid)

            except Exception as exc:
                log.exception("conduct_screening: database save failed: %s", exc)
                if self._session:
                    self._session.rollback()

        return ScreeningResult(
            screening_id=form["screening_id"],
            screening_uuid=screening_uuid,
            od_eye_result=od_eye_result,
            os_eye_result=os_eye_result,
            child_result=child_result,
            od_engine_raw=od_engine_raw,
            os_engine_raw=os_engine_raw,
            success=True,
        )

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def get_screening_history(self, screening_id: str) -> dict:
        """Return complete screening record including all related data."""
        if not self._session:
            return {}
        from app.database.repository import ScreeningRepository
        repo = ScreeningRepository(self._session)
        return repo.get_screening_full(screening_id) or {}

    def search_screenings(self, query: str, filters: dict | None = None) -> list[dict]:
        """Search screenings with optional site/date filters."""
        if not self._session:
            return []
        from app.database.repository import ScreeningRepository
        repo = ScreeningRepository(self._session)
        filters = filters or {}
        if query:
            return repo.search_screenings(query)
        return repo.list_screenings(
            site=filters.get("site"),
            date_from=filters.get("date_from"),
            date_to=filters.get("date_to"),
        )
