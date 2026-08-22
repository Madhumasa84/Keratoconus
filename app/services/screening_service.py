"""Local orchestration for the mandatory bilateral KeraScan workflow.

The service does not reinterpret Phase-1 failures.  It maps each engine run to
an explicit image state and only hands a completed image classification to the
screening matrix after all image gates have passed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PHASE1_SRC = Path(__file__).parent.parent.parent / "kerascan" / "src"
if str(_PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(_PHASE1_SRC))

try:
    from kerascan import EngineConfig, KerascanEngine
    from kerascan.config import (
        GeometryConfig,
        GeometryThresholds,
        QualityConfig,
        RadialConfig,
        TrackingConfig,
    )
    from kerascan.image_io import SUPPORTED_SUFFIXES, read_image

    _ENGINE_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - kept for local install diagnostics
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)
    SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Provisional, non-clinical demo configuration.
#
# The kerascan engine's own defaults stay conservative (unverified hardware
# ring count, no clinical thresholds -> ANALYSIS_BLOCKED/NOT_CALIBRATED for
# everyone). This app-layer opt-in exists only so the live demo can produce a
# result end to end; it is NOT a validated hardware or clinical configuration:
#
#   - The acquisition-quality gate is advisory here, not blocking: the workflow
#     assumes a clinician has already reviewed and selected the capture before
#     upload. Quality is still measured and shown alongside the result.
#   - The analysis basis is the set of rings actually reconstructed from each
#     capture (accept_detected_ring_count), NOT a verified hardware mire count,
#     which this project does not have. Real hand-held photos nearly always
#     carry partial eyelid/eyelash occlusion, so demanding a fixed ring count
#     with no missing sector refused ~85 of this project's own 112 sample
#     photos. Coverage requirements are correspondingly relaxed so a partial
#     capture is assessed on what was genuinely observed instead of refused.
#   - PROVISIONAL_GEOMETRY_THRESHOLDS are approximate 75th/90th-percentile
#     bounds on this project's own sample photos (pooled across normal/ and
#     suspicious/ folders -- folder membership was NOT used as a label; see
#     kerascan/ATTRIBUTION.md). They are an engineering outlier heuristic,
#     not a clinically validated cut-off, and must never be presented as one.
# ---------------------------------------------------------------------------
if _ENGINE_AVAILABLE:
    PROVISIONAL_GEOMETRY_THRESHOLDS = GeometryThresholds(
        version="provisional-non-clinical-outlier-heuristic-v1",
        suspicious_bounds={
            "SPACING_VARIATION": 0.28,
            "OPPOSITE_ASYMMETRY": 1.55,
            "LOCAL_COMPRESSION": 0.95,
            "LOCAL_EXPANSION": 1.38,
            "RING_SHAPE_IRREGULARITY": 0.147,
            "MULTIRING_AGREEMENT": 0.65,
        },
        indeterminate_bounds={
            "SPACING_VARIATION": 0.20,
            "OPPOSITE_ASYMMETRY": 1.30,
            "LOCAL_COMPRESSION": 0.85,
            "LOCAL_EXPANSION": 1.19,
            "RING_SHAPE_IRREGULARITY": 0.132,
            "MULTIRING_AGREEMENT": 0.51,
        },
    )

    def _default_engine_config() -> "EngineConfig":
        return EngineConfig(
            hardware_version="unverified-clinician-selected-capture-workflow",
            ring_count_source="detected_self_consistent_not_hardware_verified",
            quality=QualityConfig(enforce_gate=False),
            radial=RadialConfig(accept_detected_ring_count=True),
            tracking=TrackingConfig(min_direct_coverage=0.30, min_tracking_confidence=0.20),
            geometry=GeometryConfig(
                thresholds=PROVISIONAL_GEOMETRY_THRESHOLDS,
                max_missing_sector_fraction=0.60,
            ),
        )
else:  # pragma: no cover - kept for local install diagnostics
    def _default_engine_config():
        return None

from .protocol import ScreeningProtocol
from .referral_engine import (
    COMPLETE_IMAGE_STATES,
    ChildReferralResult,
    EyeReferralResult,
    EyeScreeningFlags,
    EyeScreeningInput,
    ReferralEngine,
)

log = logging.getLogger(__name__)

K_MIN, K_MAX = 30.0, 70.0
PACHY_MIN, PACHY_MAX = 200.0, 800.0
CYL_MIN, CYL_MAX = -12.0, 12.0
AGE_MIN, AGE_MAX = 5, 25

REPORT_IMAGE_FILENAMES = (
    "cropped_roi.png",
    "cropped_roi_centres.png",
    "tracked_rings_cartesian.png",
    "directional_spacing.png",
    # New reference-comparison images (full-stack pipeline)
    "observed_vs_concentric_reference.png",
    "observed_vs_smooth_reference.png",
    "clinician_comparison_panel.png",
)


@dataclass(frozen=True)
class ImageVerification:
    """Auditable result of the mandatory per-eye image gate."""

    eye: str
    image_status: str
    engine_result: str
    failure_stage: str
    message: str
    original_image_hash: str | None
    kerascan_image_id: str | None
    raw_result: dict[str, Any]
    geometry_validation_status: str
    processed_output_hashes: dict[str, str] = field(default_factory=dict)
    artifact_manifest: dict[str, dict[str, str]] = field(default_factory=dict)
    provenance_hash: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.image_status in COMPLETE_IMAGE_STATES


@dataclass
class ScreeningResult:
    screening_id: str
    screening_uuid: str
    od_eye_result: EyeReferralResult | None
    os_eye_result: EyeReferralResult | None
    child_result: ChildReferralResult | None
    od_engine_raw: dict | None
    os_engine_raw: dict | None
    od_image_verification: ImageVerification | None = None
    os_image_verification: ImageVerification | None = None
    validation_errors: list[str] = field(default_factory=list)
    success: bool = True
    error_message: str = ""
    pdf_path: str = ""
    json_path: str = ""
    excel_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ScreeningService:
    """Execute bilateral verification, measurement validation, and local storage."""

    def __init__(self, db_session=None, engine_config=None, protocol_path=None):
        self._session = db_session
        self._engine_config = engine_config or _default_engine_config()
        self._referral_engine = ReferralEngine(protocol_path)
        self._image_engine: Any = None

    @property
    def protocol(self) -> ScreeningProtocol:
        return self._referral_engine.protocol

    @property
    def _enforce_quality_gate(self) -> bool:
        """Whether a poor acquisition-quality score should reject the image."""
        quality = getattr(self._engine_config, "quality", None)
        return bool(getattr(quality, "enforce_gate", True))

    def _get_image_engine(self) -> Any:
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(f"Phase 1 engine not available: {_ENGINE_IMPORT_ERROR}")
        if self._image_engine is None:
            self._image_engine = KerascanEngine(self._engine_config)
        return self._image_engine

    # ------------------------------------------------------------------
    # Form and active-measurement validation
    # ------------------------------------------------------------------

    def validate_screening_form(self, form_data: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        for field_name in ("screening_id", "operator_id", "device_id", "site"):
            value = form_data.get(field_name)
            if not value or not str(value).strip():
                errors.append(f"'{field_name}' is required.")
        screening_id = str(form_data.get("screening_id", "")).strip()
        if screening_id and not screening_id.replace("-", "").replace("_", "").isalnum():
            errors.append("screening_id must be alphanumeric (hyphens and underscores allowed).")
        age = form_data.get("age")
        if age is None:
            errors.append("'age' is required.")
        else:
            try:
                numeric_age = int(age)
                if not AGE_MIN <= numeric_age <= AGE_MAX:
                    errors.append(f"Age must be between {AGE_MIN} and {AGE_MAX}.")
            except (TypeError, ValueError):
                errors.append("'age' must be an integer.")
        if form_data.get("sex") not in ("Male", "Female", "Not recorded"):
            errors.append("'sex' must be one of: Male, Female, Not recorded.")
        if not form_data.get("consent_recorded"):
            errors.append("Consent must be recorded before proceeding.")
        if not form_data.get("screening_date"):
            errors.append("'screening_date' is required.")
        return not errors, errors

    def validate_measurements(self, measurements: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate only the four active inputs; no inactive legacy field is read."""
        errors: list[str] = []

        def required_number(label: str, key: str, minimum: float, maximum: float) -> float | None:
            raw = measurements.get(key)
            if raw is None or raw == "":
                errors.append(f"{label} is required.")
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                errors.append(f"{label} must be a number.")
                return None
            if not minimum <= value <= maximum:
                errors.append(f"{label} {value:g} is outside valid range [{minimum:g}, {maximum:g}].")
            return value

        k1 = required_number("K1 flat (D)", "k1_d", K_MIN, K_MAX)
        k2 = required_number("K2 steep (D)", "k2_d", K_MIN, K_MAX)
        required_number("Pachymetry (µm)", "pachymetry_um", PACHY_MIN, PACHY_MAX)
        required_number("Cylinder (D)", "cylinder_d", CYL_MIN, CYL_MAX)
        if k1 is not None and k2 is not None and k1 > k2:
            errors.append("K1 flat (D) must be less than or equal to K2 steep (D).")
        return not errors, errors

    # ------------------------------------------------------------------
    # Image validation and provenance
    # ------------------------------------------------------------------

    @staticmethod
    def _file_sha256(path: str | Path) -> str | None:
        try:
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None

    # Public historical spelling retained for consumers; it is a file hash,
    # never a source path or remote identifier.
    _image_sha256 = _file_sha256

    @staticmethod
    def _serialisable_engine_result(result: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in result.items() if key != "_artifacts"}

    @staticmethod
    def _geometry_status(raw: dict[str, Any]) -> str:
        explicit = raw.get("geometry_validation")
        if isinstance(explicit, dict):
            value = explicit.get("status")
            if value in {"PASS", "FAIL"}:
                return value
            if explicit.get("valid") is True:
                return "PASS"
            if explicit.get("valid") is False:
                return "FAIL"
        # In the repaired engine geometry invariants are added to tracking flags;
        # a tracking PASS therefore proves geometry PASS.
        return "PASS" if raw.get("tracking", {}).get("status") == "PASS" else "FAIL"

    @staticmethod
    def _quality_acceptable(raw: dict[str, Any]) -> bool:
        acquisition = raw.get("acquisition_quality") or {}
        if acquisition:
            status = acquisition.get("status") or acquisition.get("quality_level") or ""
            return status in {"ACCEPTABLE", "ACCEPTABLE_WITH_WARNING"}
        quality = raw.get("quality") or {}
        # Compatibility for old local records only. New engine results carry the
        # explicit authoritative acquisition status.
        status = quality.get("status") or quality.get("quality_level") or ""
        return status in {"ACCEPTABLE", "ACCEPTABLE_WITH_WARNING"} or (
            "status" not in quality and "quality_level" not in quality and quality.get("gradable") is True
        )

    @staticmethod
    def _acquisition_rejection_message(raw: dict[str, Any]) -> str:
        quality = raw.get("acquisition_quality") or raw.get("quality") or {}
        flags = set(quality.get("flags") or [])
        if "blur" in flags:
            return "Image rejected: blur. Upload a new good-quality KeraScan image."
        if {"underexposed", "glare_or_saturation", "low_contrast"} & flags:
            return "Image rejected: poor exposure or contrast. Upload a new good-quality KeraScan image."
        if "pattern_off_centre" in flags:
            return "Image rejected: incorrect centring. Upload a new good-quality KeraScan image."
        if {"placido_pattern_too_small", "low_resolution", "possible_eyelid_or_eyelash_obstruction"} & flags:
            return "Image rejected: insufficient Placido coverage. Upload a new good-quality KeraScan image."
        return "Image rejected: acquisition quality did not pass; upload a new good-quality image."

    @staticmethod
    def _manifest_for_output(
        output_dir: Path | None,
        *,
        eye: str,
        source_hash: str | None,
        pipeline_version: str | None,
        model_version: str | None,
    ) -> tuple[dict[str, str], dict[str, dict[str, str]], str | None]:
        if output_dir is None or not output_dir.exists():
            return {}, {}, None
        hashes: dict[str, str] = {}
        manifest: dict[str, dict[str, str]] = {}
        for item in sorted(output_dir.iterdir()):
            if not item.is_file():
                continue
            digest = ScreeningService._file_sha256(item)
            if digest is None:
                continue
            hashes[item.name] = digest
            manifest[item.name] = {
                "path": str(item.resolve()),
                "sha256": digest,
                "eye": eye,
                "source_image_hash": source_hash or "",
            }
        provenance = {
            "eye": eye,
            "original_image_hash": source_hash,
            "processed_output_hashes": hashes,
            "pipeline_version": pipeline_version,
            "model_version": model_version,
        }
        run_hash = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()
        for record in manifest.values():
            record["provenance_hash"] = run_hash
        return hashes, manifest, run_hash

    def _verification_failure(
        self,
        eye: str,
        status: str,
        stage: str,
        message: str,
        *,
        image_hash: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> ImageVerification:
        result = raw or {
            "screening_result": "UNGRADABLE",
            "classification_performed": False,
            "classification_skipped": True,
            "quality": {"gradable": False, "quality_score": 0, "flags": [stage.lower()], "metrics": {}},
            "acquisition_quality": {"status": "FAIL", "score": 0, "flags": [stage.lower()], "metrics": {}},
            "segmentation": {"status": "NOT_RUN", "flags": []},
            "tracking": {"status": "NOT_RUN", "flags": []},
            "failure_stage": stage,
            "message": message,
            "pipeline_version": "unknown",
        }
        result = self._serialisable_engine_result(result)
        result["image_status"] = status
        result["geometry_validation"] = {"status": "FAIL"}
        return ImageVerification(
            eye=eye,
            image_status=status,
            engine_result=str(result.get("screening_result", "UNGRADABLE")),
            failure_stage=stage,
            message=message,
            original_image_hash=image_hash,
            kerascan_image_id=image_hash[:32] if image_hash else None,
            raw_result=result,
            geometry_validation_status="FAIL",
        )

    def verify_image(
        self,
        image_path: str | Path | None,
        eye: str,
        output_dir: str | Path | None = None,
    ) -> ImageVerification:
        """Decode and run the full image pipeline for exactly one labelled eye."""
        if eye not in {"OD", "OS"}:
            raise ValueError("eye must be OD or OS")
        if not image_path:
            return self._verification_failure(eye, "MISSING", "MISSING", f"No {eye} KeraScan image was provided.")
        path = Path(image_path)
        image_hash = self._file_sha256(path)
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return self._verification_failure(
                eye, "IMAGE_REJECTED", "UNSUPPORTED", "Image rejected: unsupported image format.", image_hash=image_hash
            )
        if image_hash is None:
            return self._verification_failure(
                eye, "IMAGE_REJECTED", "DECODE", "Image rejected: file cannot be decoded.", image_hash=image_hash
            )
        try:
            # Decode before invoking the engine so a malformed input cannot be
            # mistaken for a normal outcome.
            read_image(path)
        except Exception:
            return self._verification_failure(
                eye, "IMAGE_REJECTED", "DECODE", "Image rejected: file cannot be decoded.", image_hash=image_hash
            )

        actual_output = Path(output_dir) if output_dir else None
        try:
            raw = self._get_image_engine().analyze(path, actual_output)
        except Exception:
            log.exception("Phase 1 image analysis failed for local %s input", eye)
            return self._verification_failure(
                eye, "IMAGE_REJECTED", "ROI", "Image rejected: ROI detection failed.", image_hash=image_hash
            )
        raw = self._serialisable_engine_result(raw)
        raw["original_file_sha256"] = image_hash
        raw["geometry_validation"] = {"status": self._geometry_status(raw)}

        failure_stage = str(raw.get("failure_stage", "NONE"))
        segmentation_status = raw.get("segmentation", {}).get("status")
        tracking_status = raw.get("tracking", {}).get("status")
        geometry_status = raw["geometry_validation"]["status"]

        # The app mirrors the engine's quality policy: when the engine is
        # configured to treat acquisition quality as advisory (clinician-selected
        # captures), the app must not re-impose the rejection the engine skipped.
        if failure_stage == "ACQUISITION" or (self._enforce_quality_gate and not self._quality_acceptable(raw)):
            status, message = "IMAGE_REJECTED", self._acquisition_rejection_message(raw)
        elif not raw.get("roi", {}).get("box_xyxy") or raw.get("roi", {}).get("method") == "fallback_image_center":
            status, failure_stage, message = "IMAGE_REJECTED", "ROI", "Image rejected: incorrect centring or unsupported Placido coverage."
        elif segmentation_status != "PASS" or failure_stage == "SEGMENTATION":
            status, failure_stage, message = "SEGMENTATION_FAILED", "SEGMENTATION", "Image rejected: segmentation failure."
        elif tracking_status != "PASS" or geometry_status != "PASS" or failure_stage == "TRACKING":
            status, failure_stage, message = "TRACKING_FAILED", "TRACKING", "Image rejected: tracking failure."
        elif failure_stage == "CONFIGURATION":
            status, message = "ANALYSIS_BLOCKED", "Image analysis is blocked because the verified KeraScan hardware ring count has not been configured."
        elif raw.get("classification_performed") is not True:
            # This protects against a model call before the engine's explicit
            # geometry/model/hardware gates have permitted it.
            status, failure_stage, message = "ANALYSIS_BLOCKED", "MODEL_GATE", "Image analysis is blocked; completed gated image classification is unavailable."
        elif raw.get("screening_result") == "NORMAL-LIKE":
            status, message = "NORMAL_LIKE", (
                "Normal-like KeraScan image analysis completed (provisional, non-clinical geometry "
                "heuristic; not a validated diagnostic result)."
            )
        elif raw.get("screening_result") == "SUSPICIOUS":
            status, message = "SUSPICIOUS", (
                "Suspicious KeraScan image analysis completed (provisional, non-clinical geometry "
                "heuristic; not a validated diagnostic result)."
            )
        elif raw.get("screening_result") == "INDETERMINATE":
            status, message = "INDETERMINATE", (
                "Image analysis was indeterminate under the provisional geometry heuristic; "
                "repeat measurement or clinical review is required."
            )
        else:
            status, message = "IMAGE_REJECTED", "Image rejected: unsupported KeraScan analysis result."

        hashes, manifest, provenance_hash = self._manifest_for_output(
            actual_output,
            eye=eye,
            source_hash=image_hash,
            pipeline_version=raw.get("pipeline_version"),
            model_version=(raw.get("model") or {}).get("model_hash"),
        )
        raw.update({
            "image_status": status,
            "image_failure_stage": failure_stage,
            "image_message": message,
            "processed_output_hashes": hashes,
            "analysis_provenance_hash": provenance_hash,
        })
        return ImageVerification(
            eye=eye,
            image_status=status,
            engine_result=str(raw.get("screening_result", "UNGRADABLE")),
            failure_stage=failure_stage,
            message=message,
            original_image_hash=image_hash,
            kerascan_image_id=image_hash[:32],
            raw_result=raw,
            geometry_validation_status=geometry_status,
            processed_output_hashes=hashes,
            artifact_manifest=manifest,
            provenance_hash=provenance_hash,
        )

    def _analyse_image(self, image_path: str | Path) -> dict[str, Any]:
        """Backward-compatible raw-result helper used by older local callers."""
        verification = self.verify_image(image_path, "OD")
        return verification.raw_result

    # ------------------------------------------------------------------
    # Matrix and persistence
    # ------------------------------------------------------------------

    def _evaluate_eye(
        self,
        eye: str,
        verification: ImageVerification,
        measurements: dict[str, Any],
    ) -> EyeReferralResult:
        valid, validation_errors = self.validate_measurements(measurements)
        result = self._referral_engine.evaluate_eye(
            eye,
            verification.engine_result,
            measurements,
            image_status=verification.image_status,
            kerascan_image_id=verification.kerascan_image_id,
        )
        if valid:
            return result

        # If image verification already failed its state remains the primary
        # blocker, but we retain measurement errors as warnings in the record.
        flags = result.flags
        if flags is None:
            flags = EyeScreeningFlags("INVALID", "INVALID", "INVALID", "INVALID", 0)
        is_missing = any("is required" in error for error in validation_errors)
        codes = list(result.reason_codes)
        code = "MEASUREMENT_MISSING" if is_missing else "MEASUREMENT_INVALID"
        if code not in codes:
            codes.append(code)
        return EyeReferralResult(
            laterality=eye,
            decision="INCOMPLETE_SCREENING",
            action="INCOMPLETE",
            reason_codes=codes,
            engine_result=verification.engine_result,
            image_status=verification.image_status,
            flags=flags,
            explanation="Complete and correct the required screening measurements before finalizing.",
            missing_or_invalid_fields=validation_errors,
            protocol_version=self._referral_engine.get_protocol_version(),
        )

    def _canonical_measurements(self, data: dict[str, Any]) -> dict[str, Any]:
        """Persist the four active inputs and the protocol-level pachymetry type."""
        result: dict[str, Any] = {}
        for key in ("k1_d", "k2_d", "pachymetry_um", "cylinder_d"):
            value = data.get(key)
            if value in (None, ""):
                result[key] = None
            else:
                try:
                    result[key] = float(value)
                except (TypeError, ValueError):
                    result[key] = None
        result["pachymetry_measurement_type"] = self.protocol.pachymetry_measurement_type
        result["reading_number"] = 1
        return result

    @staticmethod
    def _eye_row(
        screening_uuid: str,
        verification: ImageVerification,
        eye_result: EyeReferralResult,
        image_path: str | Path | None,
        protocol_version: str,
    ) -> dict[str, Any]:
        raw = verification.raw_result
        roi = raw.get("roi", {})
        quality = raw.get("acquisition_quality") or raw.get("quality", {})
        return {
            "screening_id": screening_uuid,
            "laterality": verification.eye,
            "eye_result": verification.engine_result,
            "image_status": verification.image_status,
            "image_failure_stage": verification.failure_stage,
            "image_message": verification.message,
            "kerascan_image_id": verification.kerascan_image_id,
            "reason_codes": eye_result.reason_codes,
            "image_path": str(image_path) if image_path else None,
            "image_hash": verification.original_image_hash,
            "processed_image_hash": verification.provenance_hash,
            "processed_output_hashes": verification.processed_output_hashes,
            "analysis_artifacts": verification.artifact_manifest,
            "analysis_provenance_hash": verification.provenance_hash,
            "geometry_validation_status": verification.geometry_validation_status,
            "roi_box": roi.get("box_xyxy"),
            "roi_center": roi.get("center_full"),
            "roi_radius": roi.get("outer_radius_px"),
            "roi_confidence": roi.get("confidence"),
            "roi_method": roi.get("method"),
            "quality_gradable": quality.get("status") == "ACCEPTABLE" if "status" in quality else quality.get("gradable"),
            "quality_score": quality.get("score", quality.get("quality_score")),
            "quality_flags": quality.get("flags", []),
            "quality_metrics": quality.get("metrics", {}),
            "features": raw.get("features", {}),
            "pipeline_version": raw.get("pipeline_version"),
            "model_version": (raw.get("model") or {}).get("model_hash"),
            "protocol_version": protocol_version,
        }

    def _persist(
        self,
        form: dict[str, Any],
        od_verification: ImageVerification,
        os_verification: ImageVerification,
        od_result: EyeReferralResult,
        os_result: EyeReferralResult,
        child_result: ChildReferralResult,
        od_measurements: dict[str, Any],
        os_measurements: dict[str, Any],
        od_image_path: str | Path | None,
        os_image_path: str | Path | None,
    ) -> str:
        if self._session is None:
            return ""
        from app.database.repository import ScreeningRepository

        repo = ScreeningRepository(self._session)
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
            "overall_action": child_result.action,
            "affected_eyes": child_result.affected_eyes,
            "referral_priority": child_result.referral_priority,
            "protocol_version": self.protocol.protocol_version,
            "software_version": self.protocol.software_version,
            "pdf_generated": False,
        })
        for verification, eye_result, measurements, image_path in (
            (od_verification, od_result, od_measurements, od_image_path),
            (os_verification, os_result, os_measurements, os_image_path),
        ):
            eye_id = repo.save_eye(self._eye_row(
                screening_uuid, verification, eye_result, image_path, self.protocol.protocol_version
            ))
            repo.save_measurements([dict(self._canonical_measurements(measurements), eye_id=eye_id)])
            repo.save_image_analysis({
                "eye_id": eye_id,
                "engine_result": verification.raw_result,
                "image_status": verification.image_status,
                "failure_stage": verification.failure_stage,
                "geometry_validation_status": verification.geometry_validation_status,
                "original_image_hash": verification.original_image_hash,
                "processed_output_hashes": verification.processed_output_hashes,
                "artifact_manifest": verification.artifact_manifest,
                "provenance_hash": verification.provenance_hash,
                "model_version": (verification.raw_result.get("model") or {}).get("model_hash"),
            })
            repo.save_decision({
                "eye_id": eye_id,
                "screening_id": screening_uuid,
                "decision_level": "eye",
                "automated_result": eye_result.decision,
                "automated_reason_codes": eye_result.reason_codes,
                "final_result": eye_result.decision,
                "protocol_version": self.protocol.protocol_version,
            })
        repo.save_decision({
            "screening_id": screening_uuid,
            "decision_level": "child",
            "automated_result": child_result.decision,
            "automated_reason_codes": child_result.reason_codes,
            "final_result": child_result.decision,
            "protocol_version": self.protocol.protocol_version,
        })
        self._session.commit()
        return screening_uuid

    def conduct_screening(self, screening_data: dict[str, Any]) -> ScreeningResult:
        """Run the non-diagnostic bilateral workflow without bypassing image gates."""
        form = screening_data.get("form", {})
        valid_form, form_errors = self.validate_screening_form(form)
        if not valid_form:
            return ScreeningResult(
                screening_id=form.get("screening_id", ""), screening_uuid="",
                od_eye_result=None, os_eye_result=None, child_result=None,
                od_engine_raw=None, os_engine_raw=None, validation_errors=form_errors,
                success=False, error_message="Form validation failed.",
            )

        repo = None
        if self._session is not None:
            from app.database.repository import ScreeningRepository

            repo = ScreeningRepository(self._session)
            if repo.screening_id_exists(form["screening_id"]):
                return ScreeningResult(
                    screening_id=form["screening_id"], screening_uuid="",
                    od_eye_result=None, os_eye_result=None, child_result=None,
                    od_engine_raw=None, os_engine_raw=None,
                    validation_errors=[f"Screening ID '{form['screening_id']}' already exists."],
                    success=False, error_message="Duplicate screening ID.",
                )

        od_image_path = screening_data.get("od_image_path")
        os_image_path = screening_data.get("os_image_path")
        output_root = screening_data.get("analysis_output_dir")
        od_output = Path(output_root) / "OD" if output_root else None
        os_output = Path(output_root) / "OS" if output_root else None
        od_verification = self.verify_image(od_image_path, "OD", od_output)
        os_verification = self.verify_image(os_image_path, "OS", os_output)

        # A binary-identical upload cannot serve as both eyes. Laterality is an
        # operator assertion and this guard prevents a simple file-selection error.
        if (
            od_verification.original_image_hash
            and od_verification.original_image_hash == os_verification.original_image_hash
        ):
            duplicate_message = "Image rejected: the same uploaded file cannot be used for both OD and OS."
            od_verification = replace(
                od_verification, image_status="IMAGE_REJECTED", failure_stage="DUPLICATE_EYE_IMAGE",
                message=duplicate_message,
            )
            os_verification = replace(
                os_verification, image_status="IMAGE_REJECTED", failure_stage="DUPLICATE_EYE_IMAGE",
                message=duplicate_message,
            )
            for verification in (od_verification, os_verification):
                verification.raw_result.update({
                    "image_status": verification.image_status,
                    "image_failure_stage": verification.failure_stage,
                    "image_message": verification.message,
                })

        od_measurements = screening_data.get("od_measurements") or {}
        os_measurements = screening_data.get("os_measurements") or {}
        od_result = self._evaluate_eye("OD", od_verification, od_measurements)
        os_result = self._evaluate_eye("OS", os_verification, os_measurements)
        child_result = self._referral_engine.evaluate_child(od_result, os_result)

        screening_uuid = ""
        try:
            screening_uuid = self._persist(
                form, od_verification, os_verification, od_result, os_result, child_result,
                od_measurements, os_measurements, od_image_path, os_image_path,
            )
        except Exception as exc:
            log.exception("Local database save failed")
            if self._session is not None:
                self._session.rollback()
            return ScreeningResult(
                screening_id=form["screening_id"], screening_uuid="",
                od_eye_result=od_result, os_eye_result=os_result, child_result=child_result,
                od_engine_raw=od_verification.raw_result, os_engine_raw=os_verification.raw_result,
                od_image_verification=od_verification, os_image_verification=os_verification,
                success=False, error_message=f"Local database save failed: {exc}",
            )

        result = ScreeningResult(
            screening_id=form["screening_id"], screening_uuid=screening_uuid,
            od_eye_result=od_result, os_eye_result=os_result, child_result=child_result,
            od_engine_raw=od_verification.raw_result, os_engine_raw=os_verification.raw_result,
            od_image_verification=od_verification, os_image_verification=os_verification,
        )
        requested_pdf = screening_data.get("referral_pdf_output_path")
        if requested_pdf:
            if child_result.action != "REFER":
                result.validation_errors.append("A detailed referral PDF is available only for screen-positive referrals.")
            else:
                result.pdf_path = self.generate_referral_pdf(result, requested_pdf)
        return result

    def generate_referral_pdf(self, result: ScreeningResult, output_path: str | Path) -> str:
        """Generate and record a detailed PDF only for a final REFER action."""
        if result.child_result is None or result.child_result.action != "REFER":
            raise ValueError("A detailed referral PDF is generated only for a screen-positive REFER action.")
        from app.services.report_service import ReportService

        if self._session is not None:
            from app.database.repository import ScreeningRepository

            data = ScreeningRepository(self._session).get_screening_full(result.screening_id)
            if not data:
                raise ValueError("Cannot generate referral PDF: local screening record was not found.")
        else:
            data = self._report_data_from_result(result)
        path = ReportService().generate_pdf(data, str(output_path))
        pdf_hash = self._file_sha256(path)
        if self._session is not None and result.screening_uuid:
            from app.database.repository import ScreeningRepository

            repo = ScreeningRepository(self._session)
            repo.record_referral_pdf(result.screening_uuid, path, pdf_hash or "")
            self._session.commit()
        return path

    def _report_data_from_result(self, result: ScreeningResult) -> dict[str, Any]:
        """Small local-only fallback used by tests or command-line workflows."""
        eyes: list[dict[str, Any]] = []
        for verification, eye_result in (
            (result.od_image_verification, result.od_eye_result),
            (result.os_image_verification, result.os_eye_result),
        ):
            if verification is None or eye_result is None:
                continue
            row = self._eye_row("", verification, eye_result, None, self.protocol.protocol_version)
            row["measurements"] = []
            row["decisions"] = [{"final_result": eye_result.decision}]
            eyes.append(row)
        child = result.child_result
        return {
            "screening_id": result.screening_id,
            "overall_result": child.decision if child else "INCOMPLETE_SCREENING",
            "overall_action": child.action if child else "INCOMPLETE",
            "referral_priority": child.referral_priority if child else "NONE",
            "affected_eyes": child.affected_eyes if child else [],
            "protocol_version": self.protocol.protocol_version,
            "software_version": self.protocol.software_version,
            "eyes": eyes,
        }

    # ------------------------------------------------------------------
    # Read-only history helpers
    # ------------------------------------------------------------------

    def get_screening_history(self, screening_id: str) -> dict[str, Any]:
        if self._session is None:
            return {}
        from app.database.repository import ScreeningRepository

        return ScreeningRepository(self._session).get_screening_full(screening_id) or {}

    def search_screenings(self, query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._session is None:
            return []
        from app.database.repository import ScreeningRepository

        repo = ScreeningRepository(self._session)
        filters = filters or {}
        if query:
            return repo.search_screenings(query)
        return repo.list_screenings(
            site=filters.get("site"), date_from=filters.get("date_from"), date_to=filters.get("date_to")
        )
