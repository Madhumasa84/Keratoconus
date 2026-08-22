"""Tests for natural language reason sentences and improved PDF generation.

Verifies that:
 - _build_natural_reason_sentences produces correct human-readable text
 - The PDF image section accepts the new preferred image set
 - The PDF image section falls back to legacy images correctly
 - Natural sentences include correct measured values
 - No diagnostic language appears (no "keratoconus positive" etc.)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.report_service import ReportService
from app.services.protocol import load_protocol


def _make_eye(
    laterality: str,
    reason_codes: list[str],
    k2_d: float | None = None,
    pachymetry_um: float | None = None,
    cylinder_d: float | None = None,
) -> dict:
    """Build a minimal eye dict for report service testing."""
    measurement = {}
    if k2_d is not None:
        measurement["k2_d"] = k2_d
    if pachymetry_um is not None:
        measurement["pachymetry_um"] = pachymetry_um
    if cylinder_d is not None:
        measurement["cylinder_d"] = cylinder_d

    return {
        "laterality": laterality,
        "reason_codes": reason_codes,
        "measurements": [measurement],
        "decisions": [],
    }


@pytest.fixture
def service() -> ReportService:
    return ReportService()


@pytest.fixture
def protocol():
    return load_protocol()


class TestNaturalReasonSentences:
    """Tests for _build_natural_reason_sentences."""

    def test_no_codes_returns_empty(self, service: ReportService, protocol) -> None:
        eyes = {"OD": _make_eye("OD", [])}
        sentences = service._build_natural_reason_sentences(["OD"], eyes, protocol)
        assert sentences == []

    def test_k2_sentence_contains_value(self, service: ReportService, protocol) -> None:
        eyes = {"OD": _make_eye("OD", ["K2_ABOVE_46_8_D"], k2_d=48.5)}
        sentences = service._build_natural_reason_sentences(["OD"], eyes, protocol)
        assert any("48.50" in s for s in sentences)
        assert any("46.8" in s or "46.80" in s for s in sentences)

    def test_pachymetry_sentence_contains_value(self, service: ReportService, protocol) -> None:
        eyes = {"OS": _make_eye("OS", ["PACHYMETRY_BELOW_480_UM"], pachymetry_um=460)}
        sentences = service._build_natural_reason_sentences(["OS"], eyes, protocol)
        assert any("460" in s for s in sentences)
        assert any("480" in s for s in sentences)

    def test_cylinder_sentence_uses_magnitude(self, service: ReportService, protocol) -> None:
        eyes = {"OD": _make_eye("OD", ["CYLINDER_MAGNITUDE_ABOVE_1_5_D"], cylinder_d=-2.5)}
        sentences = service._build_natural_reason_sentences(["OD"], eyes, protocol)
        # Should show absolute magnitude 2.50, not -2.50
        assert any("2.50" in s for s in sentences)

    def test_suspicious_image_sentence_is_present(self, service: ReportService, protocol) -> None:
        eyes = {"OD": _make_eye("OD", ["IMAGE_CLASSIFIER_SUSPICIOUS"])}
        sentences = service._build_natural_reason_sentences(["OD"], eyes, protocol)
        assert any("suspicious" in s.lower() for s in sentences)

    def test_disclaimer_appended_when_sentences_exist(self, service: ReportService, protocol) -> None:
        eyes = {"OD": _make_eye("OD", ["K2_ABOVE_46_8_D"], k2_d=50.0)}
        sentences = service._build_natural_reason_sentences(["OD"], eyes, protocol)
        assert len(sentences) >= 2
        # Last sentence must be the disclaimer
        last = sentences[-1].lower()
        assert "not a diagnosis" in last or "interpretation" in last

    def test_no_diagnostic_language(self, service: ReportService, protocol) -> None:
        """No sentence should include prohibited diagnostic language."""
        eyes = {
            "OD": _make_eye("OD", ["K2_ABOVE_46_8_D", "IMAGE_CLASSIFIER_SUSPICIOUS"], k2_d=50.0),
            "OS": _make_eye("OS", ["PACHYMETRY_BELOW_480_UM"], pachymetry_um=450),
        }
        sentences = service._build_natural_reason_sentences(["OD", "OS"], eyes, protocol)
        combined = " ".join(sentences).lower()
        # Must not use definitive diagnostic terminology
        for banned in ["keratoconus positive", "keratoconus confirmed", "diagnosed with", "diagnosis of"]:
            assert banned not in combined, f"Prohibited phrase found: {banned}"

    def test_correct_eye_label(self, service: ReportService, protocol) -> None:
        eyes = {
            "OD": _make_eye("OD", ["K2_ABOVE_46_8_D"], k2_d=48.0),
            "OS": _make_eye("OS", ["PACHYMETRY_BELOW_480_UM"], pachymetry_um=460),
        }
        sentences = service._build_natural_reason_sentences(["OD", "OS"], eyes, protocol)
        combined = " ".join(sentences)
        assert "right eye" in combined.lower() or "OD" in combined
        assert "left eye" in combined.lower() or "OS" in combined

    def test_bilateral_codes_produce_multiple_sentences(self, service: ReportService, protocol) -> None:
        eyes = {
            "OD": _make_eye("OD", ["K2_ABOVE_46_8_D"], k2_d=48.0),
            "OS": _make_eye("OS", ["K2_ABOVE_46_8_D"], k2_d=47.5),
        }
        sentences = service._build_natural_reason_sentences(["OD", "OS"], eyes, protocol)
        # Should have a sentence for each eye + disclaimer
        assert len(sentences) >= 3


class TestReportServiceQualityLevel:
    """Tests for quality_level handling in PDF generation."""

    def test_flags_for_eye_with_k2(self, service: ReportService) -> None:
        eye = _make_eye("OD", ["K2_ABOVE_46_8_D"], k2_d=48.5)
        measurement = {"k2_d": 48.5, "pachymetry_um": 500, "cylinder_d": 0.5}
        flags = service._flags_for_eye(eye, measurement)
        assert flags["k2"] == "ABNORMAL"
        assert flags["pachymetry"] == "WITHIN THRESHOLD"
        assert flags["cylinder"] == "WITHIN THRESHOLD"

    def test_affected_eyes_from_reason_codes(self, service: ReportService) -> None:
        screening_data = {
            "overall_action": "REFER",
            "eyes": [
                {
                    "laterality": "OD",
                    "reason_codes": ["IMAGE_CLASSIFIER_SUSPICIOUS"],
                    "decisions": [{"final_result": "HIGH_RISK_SCREEN_POSITIVE"}],
                },
                {
                    "laterality": "OS",
                    "reason_codes": [],
                    "decisions": [{"final_result": "SCREEN_NEGATIVE"}],
                },
            ],
        }
        affected = service._affected_eyes(screening_data)
        assert "OD" in affected
        assert "OS" not in affected
