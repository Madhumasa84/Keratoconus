"""
Unit tests for the KERASCAN Phase 2 referral engine.
Tests all 22 required decision combinations.
"""
import pytest
from app.services.referral_engine import (
    ReferralEngine, EyeReferralResult, ChildReferralResult,
    VALID_REASON_CODES, VALID_OUTPUT_CODES,
    ENGINE_SUSPICIOUS, ENGINE_NORMAL, ENGINE_UNGRADABLE,
)


# Helpers
NORMAL_MEAS = {
    "k2_d": 44.0, "pachymetry_um": 530.0, "pachymetry_type": "central",
    "cylinder_d": 0.5,
}
HIGH_K_MEAS = {
    "k2_d": 48.0, "pachymetry_um": 530.0, "pachymetry_type": "central",
    "cylinder_d": 0.5,
}
LOW_PACHY_MEAS = {
    "k2_d": 44.0, "pachymetry_um": 470.0, "pachymetry_type": "central",
    "cylinder_d": 0.5,
}
HIGH_CYL_MEAS = {
    "k2_d": 44.0, "pachymetry_um": 530.0, "pachymetry_type": "central",
    "cylinder_d": 2.5,
}
HIGH_K_LOW_PACHY = {
    "k2_d": 48.0, "pachymetry_um": 470.0, "pachymetry_type": "central",
    "cylinder_d": 0.5,
}
HIGH_K_HIGH_CYL = {
    "k2_d": 48.0, "pachymetry_um": 530.0, "pachymetry_type": "central",
    "cylinder_d": 3.0,
}
MISSING_K2 = {
    "k2_d": None, "pachymetry_um": 530.0, "pachymetry_type": "central",
    "cylinder_d": 0.5,
}
MISSING_PACHY = {
    "k2_d": 44.0, "pachymetry_um": None, "pachymetry_type": None,
    "cylinder_d": 0.5,
}


# Test 01: Both eyes normal -> SCREEN_NEGATIVE
def test_both_eyes_normal_screen_negative(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, NORMAL_MEAS)
    os = referral_engine.evaluate_eye("OS", ENGINE_NORMAL, NORMAL_MEAS)
    child = referral_engine.evaluate_child(od, os, NORMAL_MEAS, NORMAL_MEAS)
    assert od.decision == "SCREEN_NEGATIVE"
    assert os.decision == "SCREEN_NEGATIVE"
    assert child.decision == "SCREEN_NEGATIVE"


# Test 02: OD SUSPICIOUS image -> STANDARD_REFERRAL with IMG_SUSPICIOUS
def test_od_suspicious_standard_referral(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_SUSPICIOUS, NORMAL_MEAS)
    assert od.decision == "STANDARD_REFERRAL"
    assert "IMG_SUSPICIOUS" in od.reason_codes


# Test 03: SUSPICIOUS image alone (no quantitative abnormality) -> STANDARD_REFERRAL
def test_suspicious_alone_is_standard_referral(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_SUSPICIOUS, NORMAL_MEAS)
    assert od.decision == "STANDARD_REFERRAL"
    assert "IMG_SUSPICIOUS" in od.reason_codes
    assert "K_HIGH" not in od.reason_codes


# Test 04: SUSPICIOUS + K_HIGH -> PRIORITY_REFERRAL
def test_suspicious_plus_k_high_priority(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_SUSPICIOUS, HIGH_K_MEAS)
    assert od.decision == "PRIORITY_REFERRAL"
    assert "IMG_SUSPICIOUS" in od.reason_codes
    assert "K_HIGH" in od.reason_codes


# Test 05: SUSPICIOUS + PACHY_LOW -> PRIORITY_REFERRAL
def test_suspicious_plus_pachy_low_priority(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_SUSPICIOUS, LOW_PACHY_MEAS)
    assert od.decision == "PRIORITY_REFERRAL"
    assert "IMG_SUSPICIOUS" in od.reason_codes
    assert "PACHY_LOW" in od.reason_codes


# Test 06: NORMAL-LIKE + K_HIGH + PACHY_LOW -> PRIORITY_REFERRAL
def test_normal_k_high_pachy_low_priority(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, HIGH_K_LOW_PACHY)
    assert od.decision == "PRIORITY_REFERRAL"
    assert "K_HIGH" in od.reason_codes
    assert "PACHY_LOW" in od.reason_codes


# Test 07: NORMAL-LIKE + K_HIGH + CYL_HIGH -> PRIORITY_REFERRAL with TWO_DOMAIN_ABNORMAL
def test_normal_two_domains_priority(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, HIGH_K_HIGH_CYL)
    assert od.decision == "PRIORITY_REFERRAL"
    assert "TWO_DOMAIN_ABNORMAL" in od.reason_codes


# Test 08: NORMAL-LIKE + isolated K_HIGH, first reading -> REPEAT_REQUIRED
def test_isolated_k_high_first_reading_repeat(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, HIGH_K_MEAS, repeat_count=1)
    assert od.decision == "STANDARD_REFERRAL"
    assert "K_HIGH" in od.reason_codes
    assert "REPEAT_REQUIRED" in od.reason_codes
    assert od.repeat_required is True


# Test 09: NORMAL-LIKE + isolated K_HIGH after repeat -> STANDARD_REFERRAL (no repeat flag)
def test_isolated_k_high_after_repeat_standard(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, HIGH_K_MEAS, repeat_count=2)
    assert od.decision == "STANDARD_REFERRAL"
    assert "K_HIGH" in od.reason_codes
    assert "REPEAT_REQUIRED" not in od.reason_codes


# Test 10: NORMAL-LIKE + isolated PACHY_LOW -> STANDARD_REFERRAL + REPEAT_REQUIRED
def test_isolated_pachy_low_repeat_required(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, LOW_PACHY_MEAS, repeat_count=1)
    assert od.decision == "STANDARD_REFERRAL"
    assert "PACHY_LOW" in od.reason_codes
    assert "REPEAT_REQUIRED" in od.reason_codes


# Test 11: NORMAL-LIKE + isolated CYL_HIGH -> STANDARD_REFERRAL + REPEAT_REQUIRED
def test_isolated_cyl_high_repeat_required(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, HIGH_CYL_MEAS, repeat_count=1)
    assert od.decision == "STANDARD_REFERRAL"
    assert "CYL_HIGH" in od.reason_codes
    assert "REPEAT_REQUIRED" in od.reason_codes


# Test 12: UNGRADABLE + all measurements normal -> RECAPTURE_REQUIRED (NOT SCREEN_NEGATIVE)
def test_ungradable_normal_measurements_recapture(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_UNGRADABLE, NORMAL_MEAS)
    assert od.decision == "RECAPTURE_REQUIRED"
    assert od.decision != "SCREEN_NEGATIVE"
    assert "IMG_UNGRADABLE" in od.reason_codes


# Test 13: UNGRADABLE + K_HIGH -> RECAPTURE_REQUIRED
def test_ungradable_k_high_recapture(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_UNGRADABLE, HIGH_K_MEAS)
    assert od.decision == "RECAPTURE_REQUIRED"
    assert "IMG_UNGRADABLE" in od.reason_codes
    assert "K_HIGH" in od.reason_codes


# Test 14: Missing K2 -> INCOMPLETE + MEASUREMENT_MISSING
def test_missing_k2_incomplete(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, MISSING_K2)
    assert od.decision == "INCOMPLETE"
    assert "MEASUREMENT_MISSING" in od.reason_codes


# Test 15: Missing pachymetry -> INCOMPLETE + MEASUREMENT_MISSING
def test_missing_pachymetry_incomplete(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, MISSING_PACHY)
    assert od.decision == "INCOMPLETE"
    assert "MEASUREMENT_MISSING" in od.reason_codes


# Test 16: OS normal, OD SUSPICIOUS -> child STANDARD_REFERRAL (either-eye rule)
def test_either_eye_referral(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_SUSPICIOUS, NORMAL_MEAS)
    os = referral_engine.evaluate_eye("OS", ENGINE_NORMAL, NORMAL_MEAS)
    child = referral_engine.evaluate_child(od, os, NORMAL_MEAS, NORMAL_MEAS)
    assert child.decision == "STANDARD_REFERRAL"


# Test 17: OS PRIORITY_REFERRAL, OD SCREEN_NEGATIVE -> child PRIORITY_REFERRAL
def test_priority_escalates_child(referral_engine):
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, NORMAL_MEAS)
    os = referral_engine.evaluate_eye("OS", ENGINE_SUSPICIOUS, HIGH_K_MEAS)
    child = referral_engine.evaluate_child(od, os, NORMAL_MEAS, HIGH_K_MEAS)
    assert child.decision == "PRIORITY_REFERRAL"


# Test 18: K2 OD-OS difference > 1.5D -> INTER_EYE_ASYMMETRY
def test_inter_eye_asymmetry(referral_engine):
    meas_od = dict(NORMAL_MEAS, k2_d=44.0)
    meas_os = dict(NORMAL_MEAS, k2_d=46.0)  # diff = 2.0D > 1.5D threshold
    od = referral_engine.evaluate_eye("OD", ENGINE_NORMAL, meas_od)
    os = referral_engine.evaluate_eye("OS", ENGINE_NORMAL, meas_os)
    child = referral_engine.evaluate_child(od, os, meas_od, meas_os)
    assert child.inter_eye_asymmetry is True
    assert "INTER_EYE_ASYMMETRY" in child.reason_codes


# Test 19: UNGRADABLE is NEVER converted to SCREEN_NEGATIVE
def test_ungradable_never_screen_negative(referral_engine):
    for meas in [NORMAL_MEAS, HIGH_K_MEAS, LOW_PACHY_MEAS, MISSING_K2, MISSING_PACHY]:
        result = referral_engine.evaluate_eye("OD", ENGINE_UNGRADABLE, meas)
        assert result.decision != "SCREEN_NEGATIVE", (
            f"UNGRADABLE returned SCREEN_NEGATIVE for measurements: {meas}"
        )


# Test 20: Conflicting repeat measurements (K2 diff > 0.5D)
def test_conflicting_repeat_measurements():
    from app.services.screening_service import ScreeningService
    svc = ScreeningService()
    r1 = {"k2_d": 44.0}
    r2 = {"k2_d": 44.8}  # diff = 0.8D > threshold 0.5D
    assert svc.check_measurement_agreement(r1, r2) is False


# Test 21: All reason codes are from approved set
def test_all_reason_codes_approved(referral_engine):
    for meas in [NORMAL_MEAS, HIGH_K_MEAS, LOW_PACHY_MEAS, HIGH_CYL_MEAS, MISSING_K2]:
        for engine_res in [ENGINE_NORMAL, ENGINE_SUSPICIOUS, ENGINE_UNGRADABLE]:
            result = referral_engine.evaluate_eye("OD", engine_res, meas)
            for code in result.reason_codes:
                assert code in VALID_REASON_CODES, f"Invalid reason code: {code}"


# Test 22: All output codes are from approved set
def test_all_output_codes_approved(referral_engine):
    for meas in [NORMAL_MEAS, HIGH_K_MEAS, MISSING_K2]:
        for engine_res in [ENGINE_NORMAL, ENGINE_SUSPICIOUS, ENGINE_UNGRADABLE]:
            result = referral_engine.evaluate_eye("OD", engine_res, meas)
            assert result.decision in VALID_OUTPUT_CODES, f"Invalid output code: {result.decision}"
