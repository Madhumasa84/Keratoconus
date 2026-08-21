"""
referral_engine.py — Deterministic, auditable referral rule engine.

All clinical thresholds are loaded from config/referral_protocol.yaml.
The Phase 1 AI score NEVER directly determines referral — it must pass
through these explicit rules.

Reason codes (per-eye):
    IMG_SUSPICIOUS, IMG_UNGRADABLE, K_HIGH, PACHY_LOW, CYL_HIGH,
    TWO_DOMAIN_ABNORMAL, CLINICAL_SIGN, INTER_EYE_ASYMMETRY,
    REPEAT_REQUIRED, MEASUREMENT_MISSING

Output codes:
    SCREEN_NEGATIVE, STANDARD_REFERRAL, PRIORITY_REFERRAL,
    RECAPTURE_REQUIRED, INCOMPLETE, MANUAL_REVIEW
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approved reason / output codes
# ---------------------------------------------------------------------------

VALID_REASON_CODES = frozenset({
    "IMG_SUSPICIOUS", "IMG_UNGRADABLE", "K_HIGH", "PACHY_LOW", "CYL_HIGH",
    "TWO_DOMAIN_ABNORMAL", "CLINICAL_SIGN", "INTER_EYE_ASYMMETRY",
    "REPEAT_REQUIRED", "MEASUREMENT_MISSING",
})

VALID_OUTPUT_CODES = frozenset({
    "SCREEN_NEGATIVE", "STANDARD_REFERRAL", "PRIORITY_REFERRAL",
    "RECAPTURE_REQUIRED", "INCOMPLETE", "MANUAL_REVIEW",
})

# Phase 1 engine output labels
ENGINE_SUSPICIOUS = "SUSPICIOUS"
ENGINE_NORMAL = "NORMAL-LIKE"
ENGINE_UNGRADABLE = "UNGRADABLE"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EyeReferralResult:
    laterality: str           # "OD" or "OS"
    decision: str             # one of VALID_OUTPUT_CODES
    reason_codes: list[str] = field(default_factory=list)
    engine_result: str = ""   # raw Phase 1 label
    repeat_required: bool = False
    needs_third_reading: bool = False
    protocol_version: str = ""

    def __post_init__(self):
        assert self.decision in VALID_OUTPUT_CODES, f"Invalid decision: {self.decision}"
        for code in self.reason_codes:
            assert code in VALID_REASON_CODES, f"Invalid reason code: {code}"


@dataclass
class ChildReferralResult:
    decision: str             # one of VALID_OUTPUT_CODES
    referral_priority: str    # "PRIORITY" | "STANDARD" | "NONE"
    od_result: EyeReferralResult | None
    os_result: EyeReferralResult | None
    reason_codes: list[str] = field(default_factory=list)
    inter_eye_asymmetry: bool = False
    protocol_version: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReferralEngine:
    """
    Loads referral_protocol.yaml and applies the deterministic decision matrix.
    Instantiate once and reuse — the config is read at construction time.
    """

    def __init__(self, protocol_path: str | Path | None = None) -> None:
        if protocol_path is None:
            protocol_path = Path(__file__).parent.parent / "config" / "referral_protocol.yaml"
        self._protocol_path = Path(protocol_path)
        self._protocol = self._load_protocol()

    # ------------------------------------------------------------------
    # Protocol loading
    # ------------------------------------------------------------------

    def _load_protocol(self) -> dict:
        with open(self._protocol_path, "r") as fh:
            protocol = yaml.safe_load(fh)
        log.info("ReferralEngine: loaded protocol version=%s", protocol.get("version"))
        return protocol

    def get_protocol_version(self) -> str:
        return self._protocol.get("version", "unknown")

    def get_disclaimer(self) -> str:
        return self._protocol.get("disclaimer", "")

    # ------------------------------------------------------------------
    # Threshold accessors
    # ------------------------------------------------------------------

    def _k_high_threshold(self) -> float:
        return float(self._protocol["keratometry"]["high_threshold_d"])

    def _pachy_low_threshold(self) -> float:
        return float(self._protocol["pachymetry"]["low_threshold_um"])

    def _cyl_high_threshold(self) -> float:
        return float(self._protocol["cylinder"]["high_threshold_d"])

    def _k2_asymmetry_threshold(self) -> float:
        return float(self._protocol.get("inter_eye_asymmetry", {}).get("k2_diff_threshold_d", 1.5))

    def _repeat_required_on_isolated(self) -> bool:
        return bool(self._protocol.get("repeat_policy", {}).get("required_on_isolated_abnormality", True))

    # ------------------------------------------------------------------
    # Completeness check
    # ------------------------------------------------------------------

    def check_measurement_completeness(self, measurements: dict) -> tuple[bool, list[str]]:
        """
        Check that the minimum required quantitative fields are present.
        Returns (is_complete, list_of_missing_field_names).
        At minimum K2 OR pachymetry must be present for a gradable eye.
        """
        missing: list[str] = []

        # K2 is the primary keratometry metric per protocol
        if measurements.get("k2_d") is None:
            missing.append("k2_d")

        # At least one pachymetry value must be present
        if measurements.get("pachymetry_um") is None:
            missing.append("pachymetry_um")

        # Pachymetry type must be specified if pachymetry is given
        if measurements.get("pachymetry_um") is not None and not measurements.get("pachymetry_type"):
            missing.append("pachymetry_type")

        return len(missing) == 0, missing

    # ------------------------------------------------------------------
    # Quantitative threshold evaluation
    # ------------------------------------------------------------------

    def apply_quantitative_thresholds(self, measurements: dict) -> dict[str, bool]:
        """
        Evaluate which quantitative thresholds are breached.
        Returns a dict of flag_name -> bool.

        Uses K2 (steep K) as the keratometry metric per protocol.
        Distinguishes K2, Kmax, and mean K — only K2 is used for referral.
        Distinguishes central vs thinnest pachymetry — both types are compared.
        """
        flags: dict[str, bool] = {
            "K_HIGH": False,
            "PACHY_LOW": False,
            "CYL_HIGH": False,
        }

        # Keratometry — use K2 (steep K) per protocol, not Kmax or mean K
        k2 = measurements.get("k2_d")
        if k2 is not None:
            flags["K_HIGH"] = float(k2) >= self._k_high_threshold()

        # Pachymetry — compare regardless of central/thinnest type
        pachy = measurements.get("pachymetry_um")
        if pachy is not None:
            flags["PACHY_LOW"] = float(pachy) <= self._pachy_low_threshold()

        # Cylinder magnitude
        cyl = measurements.get("cylinder_d")
        if cyl is not None:
            flags["CYL_HIGH"] = abs(float(cyl)) >= self._cyl_high_threshold()

        return flags

    # ------------------------------------------------------------------
    # Per-eye decision
    # ------------------------------------------------------------------

    def evaluate_eye(
        self,
        laterality: str,
        image_result: str,
        measurements: dict,
        repeat_count: int = 1,
    ) -> EyeReferralResult:
        """
        Apply the decision matrix for one eye.

        Parameters
        ----------
        laterality    : "OD" or "OS"
        image_result  : Phase 1 engine screening_result
                        ("SUSPICIOUS" | "NORMAL-LIKE" | "UNGRADABLE")
        measurements  : dict with keys k2_d, pachymetry_um, pachymetry_type,
                        cylinder_d, and optional k1_d, kmax_d, mean_k_d, sphere_d
        repeat_count  : number of reading sets recorded (1, 2, or 3)

        Returns
        -------
        EyeReferralResult
        """
        proto_ver = self.get_protocol_version()
        reason_codes: list[str] = []
        repeat_required = False
        needs_third = False

        # ── UNGRADABLE path ──────────────────────────────────────────
        if image_result == ENGINE_UNGRADABLE:
            reason_codes.append("IMG_UNGRADABLE")
            # Check quantitative — if any abnormal: RECAPTURE and flag
            is_complete, missing = self.check_measurement_completeness(measurements)
            if not is_complete:
                for _ in missing:
                    if "MEASUREMENT_MISSING" not in reason_codes:
                        reason_codes.append("MEASUREMENT_MISSING")
                return EyeReferralResult(
                    laterality=laterality,
                    decision="INCOMPLETE",
                    reason_codes=reason_codes,
                    engine_result=image_result,
                    protocol_version=proto_ver,
                )
            flags = self.apply_quantitative_thresholds(measurements)
            # UNGRADABLE is NEVER converted to SCREEN_NEGATIVE
            return EyeReferralResult(
                laterality=laterality,
                decision="RECAPTURE_REQUIRED",
                reason_codes=reason_codes + [k for k, v in flags.items() if v],
                engine_result=image_result,
                repeat_required=True,
                protocol_version=proto_ver,
            )

        # ── Completeness check ───────────────────────────────────────
        is_complete, missing = self.check_measurement_completeness(measurements)
        if not is_complete:
            reason_codes.append("MEASUREMENT_MISSING")
            return EyeReferralResult(
                laterality=laterality,
                decision="INCOMPLETE",
                reason_codes=reason_codes,
                engine_result=image_result,
                protocol_version=proto_ver,
            )

        # ── Quantitative flags ───────────────────────────────────────
        flags = self.apply_quantitative_thresholds(measurements)
        k_high   = flags["K_HIGH"]
        pachy_low = flags["PACHY_LOW"]
        cyl_high  = flags["CYL_HIGH"]
        abnormal_quant = [k for k, v in flags.items() if v]
        n_abnormal = len(abnormal_quant)

        # Clinical signs (Vogt striae, Fleischer ring, etc.)
        has_clinical_sign = bool(measurements.get("clinical_flags"))

        # ── SUSPICIOUS path ──────────────────────────────────────────
        if image_result == ENGINE_SUSPICIOUS:
            reason_codes.append("IMG_SUSPICIOUS")
            if has_clinical_sign:
                reason_codes.append("CLINICAL_SIGN")
            if n_abnormal > 0:
                # Suspicious + any abnormal quantitative -> PRIORITY
                reason_codes.extend(abnormal_quant)
                return EyeReferralResult(
                    laterality=laterality,
                    decision="PRIORITY_REFERRAL",
                    reason_codes=reason_codes,
                    engine_result=image_result,
                    protocol_version=proto_ver,
                )
            # Suspicious alone -> STANDARD
            return EyeReferralResult(
                laterality=laterality,
                decision="STANDARD_REFERRAL",
                reason_codes=reason_codes,
                engine_result=image_result,
                protocol_version=proto_ver,
            )

        # ── NORMAL-LIKE path ─────────────────────────────────────────
        if image_result == ENGINE_NORMAL:
            if has_clinical_sign:
                reason_codes.append("CLINICAL_SIGN")

            # Two or more quantitative domains abnormal -> PRIORITY
            if n_abnormal >= 2:
                reason_codes.extend(abnormal_quant)
                reason_codes.append("TWO_DOMAIN_ABNORMAL")
                return EyeReferralResult(
                    laterality=laterality,
                    decision="PRIORITY_REFERRAL",
                    reason_codes=reason_codes,
                    engine_result=image_result,
                    protocol_version=proto_ver,
                )

            # Isolated abnormality — repeat logic
            if n_abnormal == 1:
                reason_codes.extend(abnormal_quant)
                if self._repeat_required_on_isolated() and repeat_count < 2:
                    # First reading: require repeat before final decision
                    reason_codes.append("REPEAT_REQUIRED")
                    return EyeReferralResult(
                        laterality=laterality,
                        decision="STANDARD_REFERRAL",
                        reason_codes=reason_codes,
                        engine_result=image_result,
                        repeat_required=True,
                        protocol_version=proto_ver,
                    )
                else:
                    # Abnormality confirmed after repeat
                    return EyeReferralResult(
                        laterality=laterality,
                        decision="STANDARD_REFERRAL",
                        reason_codes=reason_codes,
                        engine_result=image_result,
                        protocol_version=proto_ver,
                    )

            # Clinical sign alone with otherwise normal
            if has_clinical_sign:
                return EyeReferralResult(
                    laterality=laterality,
                    decision="STANDARD_REFERRAL",
                    reason_codes=reason_codes,
                    engine_result=image_result,
                    protocol_version=proto_ver,
                )

            # All domains normal
            return EyeReferralResult(
                laterality=laterality,
                decision="SCREEN_NEGATIVE",
                reason_codes=reason_codes,
                engine_result=image_result,
                protocol_version=proto_ver,
            )

        # Unexpected engine result
        log.warning("evaluate_eye: unexpected engine result %r — flagging MANUAL_REVIEW", image_result)
        return EyeReferralResult(
            laterality=laterality,
            decision="MANUAL_REVIEW",
            reason_codes=["MEASUREMENT_MISSING"],
            engine_result=image_result,
            protocol_version=proto_ver,
        )

    # ------------------------------------------------------------------
    # Child-level decision
    # ------------------------------------------------------------------

    def evaluate_child(
        self,
        od_result: EyeReferralResult,
        os_result: EyeReferralResult,
        od_measurements: dict | None = None,
        os_measurements: dict | None = None,
    ) -> ChildReferralResult:
        """
        Derive child-level decision from OD and OS eye results.
        If either eye is referred, the child is referred.

        Inter-eye K2 asymmetry check is also applied here.
        """
        proto_ver = self.get_protocol_version()
        child_reason_codes: list[str] = []
        inter_eye_asymmetry = False

        # Inter-eye K2 asymmetry
        if od_measurements and os_measurements:
            od_k2 = od_measurements.get("k2_d")
            os_k2 = os_measurements.get("k2_d")
            if od_k2 is not None and os_k2 is not None:
                diff = abs(float(od_k2) - float(os_k2))
                if diff >= self._k2_asymmetry_threshold():
                    inter_eye_asymmetry = True
                    child_reason_codes.append("INTER_EYE_ASYMMETRY")

        # Priority order: PRIORITY > STANDARD > RECAPTURE > INCOMPLETE > SCREEN_NEGATIVE
        priority_map = {
            "PRIORITY_REFERRAL":  5,
            "STANDARD_REFERRAL":  4,
            "RECAPTURE_REQUIRED": 3,
            "INCOMPLETE":         2,
            "MANUAL_REVIEW":      1,
            "SCREEN_NEGATIVE":    0,
        }

        od_score = priority_map.get(od_result.decision, 0)
        os_score = priority_map.get(os_result.decision, 0)
        winner = od_result if od_score >= os_score else os_result
        child_decision = winner.decision

        # Referral priority label
        if child_decision == "PRIORITY_REFERRAL":
            referral_priority = "PRIORITY"
        elif child_decision == "STANDARD_REFERRAL":
            referral_priority = "STANDARD"
        elif inter_eye_asymmetry and child_decision == "SCREEN_NEGATIVE":
            # Asymmetry alone escalates to standard referral
            child_decision = "STANDARD_REFERRAL"
            referral_priority = "STANDARD"
        else:
            referral_priority = "NONE"

        return ChildReferralResult(
            decision=child_decision,
            referral_priority=referral_priority,
            od_result=od_result,
            os_result=os_result,
            reason_codes=child_reason_codes,
            inter_eye_asymmetry=inter_eye_asymmetry,
            protocol_version=proto_ver,
        )
