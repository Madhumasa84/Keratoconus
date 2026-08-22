"""Deterministic image-and-measurement screening matrix.

This module is intentionally separate from the image engine.  It consumes only
the engine's gated result: a caller cannot turn a missing, rejected, or blocked
image into a normal image result here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal

from .protocol import ScreeningProtocol, load_protocol


ImageStatus = Literal[
    "GOOD_QUALITY_PENDING_ANALYSIS", "NORMAL_LIKE", "SUSPICIOUS",
    "IMAGE_REJECTED", "SEGMENTATION_FAILED", "TRACKING_FAILED",
    "ANALYSIS_BLOCKED", "MISSING", "INDETERMINATE", "UNGRADABLE", "NOT_CALIBRATED",
]

IMAGE_STATES = frozenset({
    "GOOD_QUALITY_PENDING_ANALYSIS", "NORMAL_LIKE", "SUSPICIOUS",
    "IMAGE_REJECTED", "SEGMENTATION_FAILED", "TRACKING_FAILED",
    "ANALYSIS_BLOCKED", "MISSING", "INDETERMINATE", "UNGRADABLE", "NOT_CALIBRATED",
})
COMPLETE_IMAGE_STATES = frozenset({"NORMAL_LIKE", "SUSPICIOUS"})

# Kept as public aliases for callers of the original application interface.
ENGINE_SUSPICIOUS = "SUSPICIOUS"
ENGINE_NORMAL = "NORMAL-LIKE"
ENGINE_UNGRADABLE = "UNGRADABLE"
ENGINE_INDETERMINATE = "INDETERMINATE"
ENGINE_NOT_CALIBRATED = "NOT_CALIBRATED"

VALID_REASON_CODES = frozenset({
    "IMAGE_CLASSIFIER_SUSPICIOUS",
    "K2_ABOVE_46_8_D",
    "PACHYMETRY_BELOW_480_UM",
    "CYLINDER_MAGNITUDE_ABOVE_1_5_D",
    "MULTIPLE_QUANTITATIVE_ABNORMALITIES",
    "MEASUREMENT_MISSING",
    "MEASUREMENT_INVALID",
    "IMAGE_MISSING",
    "IMAGE_REJECTED",
    "SEGMENTATION_FAILED",
    "TRACKING_FAILED",
    "ANALYSIS_BLOCKED",
    "IMAGE_NOT_READY",
    "IMAGE_INDETERMINATE",
    "INDETERMINATE",
    "UNGRADABLE",
    "NOT_CALIBRATED",
})

VALID_OUTPUT_CODES = frozenset({
    "HIGH_RISK_SCREEN_POSITIVE",
    "SCREEN_POSITIVE_IMAGE_ONLY",
    "DISCORDANT_SCREEN_POSITIVE",
    "SCREEN_POSITIVE_CYLINDER",
    "INDETERMINATE_SINGLE_PARAMETER",
    "SCREEN_NEGATIVE",
    "INCOMPLETE_SCREENING",
    "SCREEN_POSITIVE",
    "REPEAT_REQUIRED",
})


@dataclass(frozen=True)
class EyeScreeningInput:
    """Canonical active input model for one eye in the initial workflow."""

    eye: Literal["OD", "OS"]
    kerascan_image_id: str | None
    image_status: ImageStatus
    k1_d: float | None
    k2_d: float | None
    pachymetry_um: float | None
    cylinder_d: float | None

    @classmethod
    def from_mapping(
        cls,
        eye: str,
        measurements: dict[str, Any] | None,
        image_status: str,
        kerascan_image_id: str | None = None,
    ) -> "EyeScreeningInput":
        if eye not in {"OD", "OS"}:
            raise ValueError("eye must be OD or OS")
        if image_status not in IMAGE_STATES:
            raise ValueError(f"unknown image status: {image_status}")
        data = measurements or {}

        def number(name: str) -> float | None:
            value = data.get(name)
            if value is None or value == "":
                return None
            try:
                numeric = float(value)
                return numeric if math.isfinite(numeric) else None
            except (TypeError, ValueError):
                return None

        return cls(
            eye=eye,  # type: ignore[arg-type]
            kerascan_image_id=kerascan_image_id,
            image_status=image_status,  # type: ignore[arg-type]
            k1_d=number("k1_d"),
            k2_d=number("k2_d"),
            pachymetry_um=number("pachymetry_um"),
            cylinder_d=number("cylinder_d"),
        )


@dataclass(frozen=True)
class EyeScreeningFlags:
    image: Literal["NORMAL_LIKE", "SUSPICIOUS", "INVALID"]
    keratometry: Literal["NORMAL", "ABNORMAL", "MISSING", "INVALID"]
    pachymetry: Literal["NORMAL", "ABNORMAL", "MISSING", "INVALID"]
    refraction: Literal["NORMAL", "ABNORMAL", "MISSING", "INVALID"]
    abnormal_measurement_count: int


@dataclass
class EyeReferralResult:
    laterality: str
    decision: str
    action: str
    priority: str = "NONE"
    reason_codes: list[str] = field(default_factory=list)
    engine_result: str = ""
    image_status: str = "MISSING"
    flags: EyeScreeningFlags | None = None
    explanation: str = ""
    missing_or_invalid_fields: list[str] = field(default_factory=list)
    repeat_required: bool = False
    needs_third_reading: bool = False  # retained only for old persisted records
    protocol_version: str = ""

    def __post_init__(self) -> None:
        if self.decision not in VALID_OUTPUT_CODES:
            raise ValueError(f"Invalid decision: {self.decision}")
        unknown = set(self.reason_codes) - VALID_REASON_CODES
        if unknown:
            raise ValueError(f"Invalid reason codes: {', '.join(sorted(unknown))}")


@dataclass
class ChildReferralResult:
    decision: str
    action: str
    referral_priority: str
    od_result: EyeReferralResult | None
    os_result: EyeReferralResult | None
    reason_codes: list[str] = field(default_factory=list)
    affected_eyes: list[str] = field(default_factory=list)
    protocol_version: str = ""
    explanation: str = ""
    inter_eye_asymmetry: bool = False  # no longer contributes to this protocol


class ReferralEngine:
    """Apply the provisional school-screening decision matrix deterministically."""

    def __init__(self, protocol_path: str | Path | None = None) -> None:
        self._protocol_path = Path(protocol_path) if protocol_path else None
        self.protocol: ScreeningProtocol = load_protocol(self._protocol_path)

    def get_protocol_version(self) -> str:
        return self.protocol.protocol_version

    def get_disclaimer(self) -> str:
        return self.protocol.disclaimer

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "k2_abnormal_above_d": self.protocol.k2_abnormal_above_d,
            "pachymetry_abnormal_below_um": self.protocol.pachymetry_abnormal_below_um,
            "cylinder_magnitude_abnormal_above_d": self.protocol.cylinder_magnitude_abnormal_above_d,
        }

    def check_measurement_completeness(
        self, measurements: dict[str, Any], *, require_k1: bool = True
    ) -> tuple[bool, list[str]]:
        required = ["k2_d", "pachymetry_um", "cylinder_d"]
        if require_k1:
            required.insert(0, "k1_d")
        missing = [field for field in required if measurements.get(field) is None or measurements.get(field) == ""]
        return not missing, missing

    def _measurement_flags(self, data: EyeScreeningInput) -> tuple[EyeScreeningFlags, list[str], list[str]]:
        """Return flags, stable positive codes, and invalid/missing field names."""
        invalid: list[str] = []
        codes: list[str] = []

        if data.k1_d is None or data.k2_d is None:
            keratometry = "MISSING"
            if data.k1_d is None:
                invalid.append("K1 flat (D)")
            if data.k2_d is None:
                invalid.append("K2 steep (D)")
        elif data.k1_d > data.k2_d:
            keratometry = "INVALID"
            invalid.append("K1 flat (D) must be less than or equal to K2 steep (D)")
        elif data.k2_d > self.protocol.k2_abnormal_above_d:
            keratometry = "ABNORMAL"
            codes.append("K2_ABOVE_46_8_D")
        else:
            keratometry = "NORMAL"

        if data.pachymetry_um is None:
            pachymetry = "MISSING"
            invalid.append("Pachymetry (µm)")
        elif data.pachymetry_um < self.protocol.pachymetry_abnormal_below_um:
            pachymetry = "ABNORMAL"
            codes.append("PACHYMETRY_BELOW_480_UM")
        else:
            pachymetry = "NORMAL"

        if data.cylinder_d is None:
            refraction = "MISSING"
            invalid.append("Cylinder (D)")
        elif abs(data.cylinder_d) > self.protocol.cylinder_magnitude_abnormal_above_d:
            refraction = "ABNORMAL"
            codes.append("CYLINDER_MAGNITUDE_ABOVE_1_5_D")
        else:
            refraction = "NORMAL"

        image = data.image_status if data.image_status in COMPLETE_IMAGE_STATES else "INVALID"
        count = sum(status == "ABNORMAL" for status in (keratometry, pachymetry, refraction))
        return (
            EyeScreeningFlags(
                image=image,  # type: ignore[arg-type]
                keratometry=keratometry,  # type: ignore[arg-type]
                pachymetry=pachymetry,  # type: ignore[arg-type]
                refraction=refraction,  # type: ignore[arg-type]
                abnormal_measurement_count=count,
            ),
            codes,
            invalid,
        )

    def apply_quantitative_thresholds(self, measurements: dict[str, Any]) -> dict[str, bool]:
        """Compatibility accessor using the current strict boundary semantics."""
        data = EyeScreeningInput.from_mapping("OD", measurements, "NORMAL_LIKE")
        _, codes, _ = self._measurement_flags(data)
        return {
            "K2_ABOVE_46_8_D": "K2_ABOVE_46_8_D" in codes,
            "PACHYMETRY_BELOW_480_UM": "PACHYMETRY_BELOW_480_UM" in codes,
            "CYLINDER_MAGNITUDE_ABOVE_1_5_D": "CYLINDER_MAGNITUDE_ABOVE_1_5_D" in codes,
        }

    @staticmethod
    def _image_reason(image_status: str) -> tuple[str, str]:
        mapping = {
            "MISSING": ("IMAGE_MISSING", "A KeraScan image is required for this eye."),
            "IMAGE_REJECTED": ("IMAGE_REJECTED", "Image rejected; upload a new good-quality KeraScan image."),
            "SEGMENTATION_FAILED": ("SEGMENTATION_FAILED", "Image segmentation failed; upload a new good-quality KeraScan image."),
            "TRACKING_FAILED": ("TRACKING_FAILED", "Polar ring tracking failed; upload a new good-quality KeraScan image."),
            "ANALYSIS_BLOCKED": ("ANALYSIS_BLOCKED", "Image analysis is blocked because the verified KeraScan hardware ring count has not been configured."),
            "GOOD_QUALITY_PENDING_ANALYSIS": ("IMAGE_NOT_READY", "Good-quality image is pending completed image analysis."),
            "INDETERMINATE": ("IMAGE_INDETERMINATE", "Image analysis produced an indeterminate result; repeat measurement or clinical review required."),
            "UNGRADABLE": ("IMAGE_REJECTED", "Image quality or geometry was ungradable; upload a new good-quality image."),
            "NOT_CALIBRATED": ("ANALYSIS_BLOCKED", "Image analysis is not clinically calibrated for this device configuration."),
        }
        return mapping.get(image_status, ("IMAGE_REJECTED", "Image verification is not valid."))

    def evaluate_input(self, data: EyeScreeningInput, *, engine_result: str | None = None) -> EyeReferralResult:
        flags, measurement_codes, invalid_fields = self._measurement_flags(data)
        proto = self.get_protocol_version()

        # A rejected, missing, or blocked image can never acquire a normal label
        # because measurements happen to be normal.
        if data.image_status not in COMPLETE_IMAGE_STATES:
            # An unusable image must not suppress a referral the measurements
            # already justify. Two or more abnormal domains still refer, so the
            # child reaches tomography; a repeat image is advised alongside.
            if not invalid_fields and flags.abnormal_measurement_count >= 2:
                return EyeReferralResult(
                    laterality=data.eye,
                    decision="DISCORDANT_SCREEN_POSITIVE",
                    action="REFER",
                    priority="PRIORITY_1",
                    reason_codes=list(measurement_codes) + ["MULTIPLE_QUANTITATIVE_ABNORMALITIES"],
                    engine_result=engine_result or "UNGRADABLE",
                    image_status=data.image_status,
                    flags=flags,
                    explanation=(
                        "Two or more quantitative screening domains were abnormal. The KeraScan image "
                        "could not be graded, so referral is based on the measurements alone; repeat "
                        "the image when possible."
                    ),
                    repeat_required=True,
                    protocol_version=proto,
                )
            code, explanation = self._image_reason(data.image_status)
            return EyeReferralResult(
                laterality=data.eye,
                decision="INCOMPLETE_SCREENING",
                action="INCOMPLETE",
                reason_codes=[code] + measurement_codes,
                engine_result=engine_result or "UNGRADABLE",
                image_status=data.image_status,
                flags=flags,
                explanation=explanation,
                missing_or_invalid_fields=invalid_fields,
                protocol_version=proto,
            )

        if invalid_fields:
            code = "MEASUREMENT_MISSING" if any(
                field in {"K1 flat (D)", "K2 steep (D)", "Pachymetry (µm)", "Cylinder (D)"}
                for field in invalid_fields
            ) else "MEASUREMENT_INVALID"
            return EyeReferralResult(
                laterality=data.eye,
                decision="INCOMPLETE_SCREENING",
                action="INCOMPLETE",
                reason_codes=[code] + measurement_codes,
                engine_result=engine_result or data.image_status,
                image_status=data.image_status,
                flags=flags,
                explanation="Complete and correct the required screening measurements before finalizing.",
                missing_or_invalid_fields=invalid_fields,
                protocol_version=proto,
            )

        n_abnormal = flags.abnormal_measurement_count
        if data.image_status == "SUSPICIOUS":
            reasons = ["IMAGE_CLASSIFIER_SUSPICIOUS"] + measurement_codes
            if n_abnormal >= 1:
                if n_abnormal >= 2:
                    reasons.append("MULTIPLE_QUANTITATIVE_ABNORMALITIES")
                return EyeReferralResult(
                    laterality=data.eye,
                    decision="HIGH_RISK_SCREEN_POSITIVE",
                    action="REFER",
                    priority="PRIORITY_1",
                    reason_codes=reasons,
                    engine_result=engine_result or "SUSPICIOUS",
                    image_status=data.image_status,
                    flags=flags,
                    explanation="Suspicious KeraScan image with at least one additional abnormal screening measurement.",
                    protocol_version=proto,
                )
            return EyeReferralResult(
                laterality=data.eye,
                decision="SCREEN_POSITIVE_IMAGE_ONLY",
                action="REFER",
                priority="PRIORITY_2",
                reason_codes=reasons,
                engine_result=engine_result or "SUSPICIOUS",
                image_status=data.image_status,
                flags=flags,
                explanation=(
                    "The KeraScan image was reproducibly classified as suspicious although the "
                    "entered measurements were within the provisional study thresholds."
                ),
                protocol_version=proto,
            )

        # The only remaining completed state is NORMAL_LIKE.
        if n_abnormal >= 2:
            reasons = list(measurement_codes) + ["MULTIPLE_QUANTITATIVE_ABNORMALITIES"]
            return EyeReferralResult(
                laterality=data.eye,
                decision="DISCORDANT_SCREEN_POSITIVE",
                action="REFER",
                priority="PRIORITY_1",
                reason_codes=reasons,
                engine_result=engine_result or "NORMAL-LIKE",
                image_status=data.image_status,
                flags=flags,
                explanation="Two or more quantitative screening domains were abnormal despite a normal-like KeraScan image.",
                protocol_version=proto,
            )
        if n_abnormal == 1:
            # A raised cylinder is a standalone referral trigger under the
            # school-screening criteria, so it is not sent for repeat the way an
            # isolated keratometry or pachymetry reading is.
            if "CYLINDER_MAGNITUDE_ABOVE_1_5_D" in measurement_codes:
                return EyeReferralResult(
                    laterality=data.eye,
                    decision="SCREEN_POSITIVE_CYLINDER",
                    action="REFER",
                    priority="PRIORITY_2",
                    reason_codes=list(measurement_codes),
                    engine_result=engine_result or "NORMAL-LIKE",
                    image_status=data.image_status,
                    flags=flags,
                    explanation=(
                        "Astigmatic cylinder magnitude exceeded the screening threshold. "
                        "Corneal imaging is recommended even though the KeraScan image was normal-like."
                    ),
                    protocol_version=proto,
                )
            domain = {
                "K2_ABOVE_46_8_D": "keratometry (K2)",
                "PACHYMETRY_BELOW_480_UM": "pachymetry",
            }[measurement_codes[0]]
            return EyeReferralResult(
                laterality=data.eye,
                decision="INDETERMINATE_SINGLE_PARAMETER",
                action="REPEAT_MEASUREMENT",
                reason_codes=measurement_codes,
                engine_result=engine_result or "NORMAL-LIKE",
                image_status=data.image_status,
                flags=flags,
                explanation=(
                    f"One isolated quantitative screening parameter was abnormal ({domain}). "
                    "Repeat the measurement; refer if the abnormal value is reproduced."
                ),
                repeat_required=True,
                protocol_version=proto,
            )
        return EyeReferralResult(
            laterality=data.eye,
            decision="SCREEN_NEGATIVE",
            action="NO_IMMEDIATE_REFERRAL",
            reason_codes=[],
            engine_result=engine_result or "NORMAL-LIKE",
            image_status=data.image_status,
            flags=flags,
            explanation="No KeraScan screening criterion was positive at this encounter.",
            protocol_version=proto,
        )

    def evaluate_eye(
        self,
        laterality: str,
        image_result: str,
        measurements: dict[str, Any],
        repeat_count: int = 1,
        *,
        image_status: str | None = None,
        kerascan_image_id: str | None = None,
    ) -> EyeReferralResult:
        """Evaluate an eye; service callers must supply the verified image state.

        The no-``image_status`` fallback preserves the old programmatic API for
        archived records only.  The active :class:`ScreeningService` always
        provides a state derived from the complete image pipeline.
        """
        del repeat_count  # the initial protocol replaces readings instead of storing repeats
        if image_status is None:
            image_status = {
                ENGINE_NORMAL: "NORMAL_LIKE",
                ENGINE_SUSPICIOUS: "SUSPICIOUS",
            }.get(image_result, "IMAGE_REJECTED")
            # Old calls lacked K1; retain read compatibility but never use this
            # path in the active UI/service workflow.
            legacy_missing_k1 = measurements.get("k1_d") is None
            if legacy_missing_k1:
                measurements = dict(measurements, k1_d=measurements.get("k2_d"))
        data = EyeScreeningInput.from_mapping(laterality, measurements, image_status, kerascan_image_id)
        return self.evaluate_input(data, engine_result=image_result)

    def evaluate_child(
        self,
        od_result: EyeReferralResult,
        os_result: EyeReferralResult,
        od_measurements: dict[str, Any] | None = None,
        os_measurements: dict[str, Any] | None = None,
    ) -> ChildReferralResult:
        """Aggregate only completed eyes; incomplete input always remains incomplete."""
        del od_measurements, os_measurements
        results = [od_result, os_result]
        proto = self.get_protocol_version()
        if any(result.action == "INCOMPLETE" for result in results):
            reasons = [code for result in results for code in result.reason_codes]
            return ChildReferralResult(
                decision="INCOMPLETE_SCREENING",
                action="INCOMPLETE",
                referral_priority="NONE",
                od_result=od_result,
                os_result=os_result,
                reason_codes=list(dict.fromkeys(reasons)),
                protocol_version=proto,
                explanation="Complete valid KeraScan image verification and all required measurements for both eyes.",
            )
        referral_eyes = [result.laterality for result in results if result.action == "REFER"]
        if referral_eyes:
            priority = "PRIORITY_1" if any(result.priority == "PRIORITY_1" for result in results) else "PRIORITY_2"
            reasons = [code for result in results if result.action == "REFER" for code in result.reason_codes]
            return ChildReferralResult(
                decision="SCREEN_POSITIVE",
                action="REFER",
                referral_priority=priority,
                od_result=od_result,
                os_result=os_result,
                reason_codes=list(dict.fromkeys(reasons)),
                affected_eyes=referral_eyes,
                protocol_version=proto,
                explanation="At least one eye met the KeraScan school-screening referral criteria.",
            )
        if any(result.action == "REPEAT_MEASUREMENT" for result in results):
            return ChildReferralResult(
                decision="REPEAT_REQUIRED",
                action="REPEAT_MEASUREMENT",
                referral_priority="NONE",
                od_result=od_result,
                os_result=os_result,
                reason_codes=[code for result in results for code in result.reason_codes],
                protocol_version=proto,
                explanation="One isolated quantitative screening parameter was abnormal. Repeat the measurement before finalizing.",
            )
        return ChildReferralResult(
            decision="SCREEN_NEGATIVE",
            action="NO_IMMEDIATE_REFERRAL",
            referral_priority="NONE",
            od_result=od_result,
            os_result=os_result,
            protocol_version=proto,
            explanation="No KeraScan screening criterion was positive at this encounter.",
        )
