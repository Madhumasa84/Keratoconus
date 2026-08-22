"""Decision-matrix regression tests for the provisional school protocol."""
from __future__ import annotations

import pytest

from app.services.referral_engine import (
    ENGINE_NORMAL,
    ENGINE_SUSPICIOUS,
    EyeScreeningInput,
    ReferralEngine,
    VALID_OUTPUT_CODES,
    VALID_REASON_CODES,
)


def measurements(**changes):
    base = {"k1_d": 43.0, "k2_d": 44.0, "pachymetry_um": 530.0, "cylinder_d": 0.50}
    base.update(changes)
    return base


def evaluate(engine, eye="OD", image_status="NORMAL_LIKE", **changes):
    return engine.evaluate_eye(
        eye,
        ENGINE_SUSPICIOUS if image_status == "SUSPICIOUS" else ENGINE_NORMAL,
        measurements(**changes),
        image_status=image_status,
        kerascan_image_id=f"{eye}-image",
    )


def test_protocol_uses_required_versioned_thresholds(referral_engine):
    assert referral_engine.get_protocol_version() == "kerascan-school-screening-provisional-1"
    assert referral_engine.thresholds == {
        "k2_abnormal_above_d": 46.8,
        "pachymetry_abnormal_below_um": 480.0,
        "cylinder_magnitude_abnormal_above_d": 1.5,
    }
    assert referral_engine.protocol.pachymetry_measurement_type == "device_reported"


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("k2_d", 46.8, "NORMAL"),
        ("k2_d", 46.81, "ABNORMAL"),
        ("pachymetry_um", 480.0, "NORMAL"),
        ("pachymetry_um", 479.0, "ABNORMAL"),
        ("cylinder_d", 1.50, "NORMAL"),
        ("cylinder_d", -1.50, "NORMAL"),
        ("cylinder_d", 1.51, "ABNORMAL"),
        ("cylinder_d", -1.51, "ABNORMAL"),
    ],
)
def test_measurement_boundary_behaviour(referral_engine, key, value, expected):
    result = evaluate(referral_engine, **{key: value})
    if key == "k2_d":
        assert result.flags.keratometry == expected
    elif key == "pachymetry_um":
        assert result.flags.pachymetry == expected
    else:
        assert result.flags.refraction == expected


def test_k1_is_recorded_and_must_not_exceed_k2(referral_engine):
    invalid = evaluate(referral_engine, k1_d=47.0, k2_d=46.0)
    assert invalid.decision == "INCOMPLETE_SCREENING"
    assert "MEASUREMENT_INVALID" in invalid.reason_codes
    assert any("K1" in field for field in invalid.missing_or_invalid_fields)


@pytest.mark.parametrize(
    ("image_status", "changes", "decision", "action", "priority", "required_codes"),
    [
        ("NORMAL_LIKE", {}, "SCREEN_NEGATIVE", "NO_IMMEDIATE_REFERRAL", "NONE", []),
        ("SUSPICIOUS", {}, "SCREEN_POSITIVE_IMAGE_ONLY", "REFER", "PRIORITY_2", ["IMAGE_CLASSIFIER_SUSPICIOUS"]),
        ("SUSPICIOUS", {"k2_d": 47.0}, "HIGH_RISK_SCREEN_POSITIVE", "REFER", "PRIORITY_1", ["IMAGE_CLASSIFIER_SUSPICIOUS", "K2_ABOVE_46_8_D"]),
        ("SUSPICIOUS", {"pachymetry_um": 470.0}, "HIGH_RISK_SCREEN_POSITIVE", "REFER", "PRIORITY_1", ["PACHYMETRY_BELOW_480_UM"]),
        ("SUSPICIOUS", {"cylinder_d": -2.0}, "HIGH_RISK_SCREEN_POSITIVE", "REFER", "PRIORITY_1", ["CYLINDER_MAGNITUDE_ABOVE_1_5_D"]),
        ("NORMAL_LIKE", {"k2_d": 47.0, "pachymetry_um": 470.0}, "DISCORDANT_SCREEN_POSITIVE", "REFER", "PRIORITY_1", ["K2_ABOVE_46_8_D", "PACHYMETRY_BELOW_480_UM", "MULTIPLE_QUANTITATIVE_ABNORMALITIES"]),
        ("NORMAL_LIKE", {"k2_d": 47.0, "cylinder_d": 2.0}, "DISCORDANT_SCREEN_POSITIVE", "REFER", "PRIORITY_1", ["K2_ABOVE_46_8_D", "CYLINDER_MAGNITUDE_ABOVE_1_5_D"]),
        ("NORMAL_LIKE", {"pachymetry_um": 470.0, "cylinder_d": 2.0}, "DISCORDANT_SCREEN_POSITIVE", "REFER", "PRIORITY_1", ["PACHYMETRY_BELOW_480_UM", "CYLINDER_MAGNITUDE_ABOVE_1_5_D"]),
        ("NORMAL_LIKE", {"k2_d": 47.0, "pachymetry_um": 470.0, "cylinder_d": 2.0}, "DISCORDANT_SCREEN_POSITIVE", "REFER", "PRIORITY_1", ["MULTIPLE_QUANTITATIVE_ABNORMALITIES"]),
        ("NORMAL_LIKE", {"k2_d": 47.0}, "INDETERMINATE_SINGLE_PARAMETER", "REPEAT_MEASUREMENT", "NONE", ["K2_ABOVE_46_8_D"]),
    ],
)
def test_complete_per_eye_matrix(referral_engine, image_status, changes, decision, action, priority, required_codes):
    result = evaluate(referral_engine, image_status=image_status, **changes)
    assert result.decision == decision
    assert result.action == action
    assert result.priority == priority
    assert set(required_codes).issubset(result.reason_codes)


@pytest.mark.parametrize("image_status", [
    "IMAGE_REJECTED", "SEGMENTATION_FAILED", "TRACKING_FAILED", "ANALYSIS_BLOCKED", "MISSING",
])
def test_noncompleted_image_can_never_be_normal(referral_engine, image_status):
    result = evaluate(referral_engine, image_status=image_status)
    assert result.decision == "INCOMPLETE_SCREENING"
    assert result.action == "INCOMPLETE"
    assert result.image_status == image_status
    assert result.decision != "SCREEN_NEGATIVE"


@pytest.mark.parametrize("missing_key", ["k1_d", "k2_d", "pachymetry_um", "cylinder_d"])
def test_missing_mandatory_measurement_is_incomplete(referral_engine, missing_key):
    result = evaluate(referral_engine, **{missing_key: None})
    assert result.decision == "INCOMPLETE_SCREENING"
    assert "MEASUREMENT_MISSING" in result.reason_codes


def test_one_positive_eye_and_one_negative_eye_is_screen_positive(referral_engine):
    od = evaluate(referral_engine, "OD", "SUSPICIOUS")
    os = evaluate(referral_engine, "OS", "NORMAL_LIKE")
    child = referral_engine.evaluate_child(od, os)
    assert child.decision == "SCREEN_POSITIVE"
    assert child.action == "REFER"
    assert child.referral_priority == "PRIORITY_2"
    assert child.affected_eyes == ["OD"]


def test_both_positive_eyes_are_listed(referral_engine):
    od = evaluate(referral_engine, "OD", "SUSPICIOUS", k2_d=47.0)
    os = evaluate(referral_engine, "OS", "NORMAL_LIKE", k2_d=47.0, pachymetry_um=470.0)
    child = referral_engine.evaluate_child(od, os)
    assert child.decision == "SCREEN_POSITIVE"
    assert child.referral_priority == "PRIORITY_1"
    assert child.affected_eyes == ["OD", "OS"]


def test_indeterminate_eye_makes_child_repeat_required(referral_engine):
    # An isolated keratometry value is the repeat case. An isolated cylinder is
    # NOT: it refers on its own under the school-screening criteria.
    od = evaluate(referral_engine, "OD", "NORMAL_LIKE", k2_d=48.0)
    os = evaluate(referral_engine, "OS", "NORMAL_LIKE")
    child = referral_engine.evaluate_child(od, os)
    assert child.decision == "REPEAT_REQUIRED"
    assert child.action == "REPEAT_MEASUREMENT"


def test_incomplete_eye_overrides_completed_negative_child_result(referral_engine):
    od = evaluate(referral_engine, "OD", "NORMAL_LIKE")
    os = evaluate(referral_engine, "OS", "ANALYSIS_BLOCKED")
    child = referral_engine.evaluate_child(od, os)
    assert child.decision == "INCOMPLETE_SCREENING"
    assert child.action == "INCOMPLETE"


def test_codes_and_decisions_are_stable(referral_engine):
    cases = [
        evaluate(referral_engine, image_status="SUSPICIOUS", k2_d=47.0),
        evaluate(referral_engine, image_status="NORMAL_LIKE", pachymetry_um=470.0),
        evaluate(referral_engine, image_status="TRACKING_FAILED"),
    ]
    for case in cases:
        assert case.decision in VALID_OUTPUT_CODES
        assert set(case.reason_codes) <= VALID_REASON_CODES


# ---------------------------------------------------------------------------
# School-screening referral matrix (sensitivity-first OR rule)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "image_status,k2,pachymetry,cylinder,expected_action",
    [
        # A suspicious image refers on its own, whatever the measurements say.
        ("SUSPICIOUS", 44.0, 530.0, 0.5, "REFER"),
        ("SUSPICIOUS", 48.0, 470.0, 2.5, "REFER"),
        # Normal image: two or more abnormal domains refer.
        ("NORMAL_LIKE", 48.0, 470.0, 0.5, "REFER"),
        ("NORMAL_LIKE", 48.0, 530.0, 2.5, "REFER"),
        ("NORMAL_LIKE", 44.0, 470.0, 2.5, "REFER"),
        # A raised cylinder alone is a standalone referral trigger.
        ("NORMAL_LIKE", 44.0, 530.0, 2.5, "REFER"),
        # An isolated K or pachymetry value is repeated, not referred.
        ("NORMAL_LIKE", 48.0, 530.0, 0.5, "REPEAT_MEASUREMENT"),
        ("NORMAL_LIKE", 44.0, 470.0, 0.5, "REPEAT_MEASUREMENT"),
        # Nothing abnormal.
        ("NORMAL_LIKE", 44.0, 530.0, 0.5, "NO_IMMEDIATE_REFERRAL"),
        # An unusable image is never a negative.
        ("TRACKING_FAILED", 44.0, 530.0, 0.5, "INCOMPLETE"),
    ],
)
def test_school_screening_referral_matrix(referral_engine, image_status, k2, pachymetry, cylinder, expected_action):
    result = referral_engine.evaluate_eye(
        "OD",
        image_status,
        {"k1_d": 42.0, "k2_d": k2, "pachymetry_um": pachymetry, "cylinder_d": cylinder},
        image_status=image_status,
    )
    assert result.action == expected_action


def test_isolated_cylinder_refers_but_isolated_keratometry_repeats(referral_engine):
    """The two isolated-abnormality cases must not be treated the same way."""
    measurements = {"k1_d": 42.0, "k2_d": 44.0, "pachymetry_um": 530.0, "cylinder_d": 2.5}
    cylinder_only = referral_engine.evaluate_eye("OD", "NORMAL_LIKE", measurements, image_status="NORMAL_LIKE")
    assert cylinder_only.action == "REFER"
    assert cylinder_only.decision == "SCREEN_POSITIVE_CYLINDER"

    keratometry_only = referral_engine.evaluate_eye(
        "OD", "NORMAL_LIKE",
        {"k1_d": 42.0, "k2_d": 48.0, "pachymetry_um": 530.0, "cylinder_d": 0.5},
        image_status="NORMAL_LIKE",
    )
    assert keratometry_only.action == "REPEAT_MEASUREMENT"
    assert keratometry_only.repeat_required is True


def test_ungradable_image_does_not_suppress_a_measurement_based_referral(referral_engine):
    """An unusable image must not hide a referral the measurements already justify."""
    result = referral_engine.evaluate_eye(
        "OD", "TRACKING_FAILED",
        {"k1_d": 42.0, "k2_d": 49.0, "pachymetry_um": 470.0, "cylinder_d": 1.48},
        image_status="TRACKING_FAILED",
    )
    assert result.action == "REFER"
    assert result.repeat_required is True, "a repeat image should still be requested"

    # One abnormal domain is not enough to refer without a gradable image, but it
    # must never be reported as negative either.
    single = referral_engine.evaluate_eye(
        "OD", "TRACKING_FAILED",
        {"k1_d": 42.0, "k2_d": 49.0, "pachymetry_um": 530.0, "cylinder_d": 0.5},
        image_status="TRACKING_FAILED",
    )
    assert single.action == "INCOMPLETE"
    assert single.decision != "SCREEN_NEGATIVE"
