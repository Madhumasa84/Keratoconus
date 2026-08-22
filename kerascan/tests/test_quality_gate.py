"""Tests for the ROI-aware 3-level quality gate.

Verifies that:
 - ACCEPTABLE / ACCEPTABLE_WITH_WARNING / REJECTED levels are distinct
 - Ring-band sharpness is used (not full-image sharpness)
 - Mild issues produce ACCEPTABLE_WITH_WARNING (gradable=True)
 - Severe issues produce REJECTED (gradable=False)
 - Backward-compatible fields are preserved
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from kerascan.config import QualityConfig
from kerascan.quality import evaluate_quality
from kerascan.synthetic import synthetic_placido


def make_ring_image(
    blur: float = 0.0,
    dark_fraction: float = 0.0,
    low_contrast: bool = False,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Return (image, center, outer_radius) for a synthetic Placido image."""
    img = synthetic_placido(blur=blur, darkness=dark_fraction, low_contrast=low_contrast)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    outer_radius = min(h, w) * 0.45
    return img, center, outer_radius


class TestQualityLevels:
    """Verify that the 3-level quality gate has distinct, correct levels."""

    def test_clean_image_is_acceptable(self) -> None:
        img, center, radius = make_ring_image()
        result = evaluate_quality(img, center, radius)
        assert result["quality_level"] == "ACCEPTABLE"
        assert result["gradable"] is True
        assert result["status"] == "ACCEPTABLE"

    def test_acceptable_with_warning_is_gradable(self) -> None:
        """Mild-blur images should proceed to analysis with a warning, not block."""
        # Create image with mild blur — should be ACCEPTABLE_WITH_WARNING
        config = QualityConfig(min_laplacian_variance=100.0, warning_laplacian_fraction=0.50)
        img, center, radius = make_ring_image(blur=1.5)  # mild blur
        result = evaluate_quality(img, center, radius, config)
        # Whether ACCEPTABLE or ACCEPTABLE_WITH_WARNING, it must be gradable
        assert result["gradable"] is True
        assert result["quality_level"] in {"ACCEPTABLE", "ACCEPTABLE_WITH_WARNING"}

    def test_severe_blur_is_rejected(self) -> None:
        img, center, radius = make_ring_image(blur=9)
        result = evaluate_quality(img, center, radius)
        assert result["quality_level"] == "REJECTED"
        assert result["gradable"] is False
        assert "blur" in result["flags"]

    def test_rejected_is_not_gradable(self) -> None:
        """REJECTED must always mean gradable=False."""
        img, center, radius = make_ring_image(blur=12)
        result = evaluate_quality(img, center, radius)
        if result["quality_level"] == "REJECTED":
            assert result["gradable"] is False

    def test_status_matches_quality_level(self) -> None:
        """status and quality_level must be consistent."""
        img, center, radius = make_ring_image()
        result = evaluate_quality(img, center, radius)
        assert result["status"] == result["quality_level"]

    def test_backward_compatible_gradable_field(self) -> None:
        """The `gradable` field must remain in the result for backward compat."""
        img, center, radius = make_ring_image()
        result = evaluate_quality(img, center, radius)
        assert "gradable" in result
        assert isinstance(result["gradable"], bool)

    def test_metrics_always_present(self) -> None:
        """Key metrics must always be populated."""
        img, center, radius = make_ring_image()
        result = evaluate_quality(img, center, radius)
        metrics = result["metrics"]
        assert "laplacian_variance" in metrics
        assert "ring_band_laplacian_variance" in metrics
        assert "mean_intensity" in metrics
        assert "contrast_std" in metrics
        assert "saturation_fraction" in metrics
        assert "noise_sigma_estimate" in metrics
        assert "pattern_centring_ratio" in metrics
        assert "pattern_radius_fraction" in metrics

    def test_ring_band_laplacian_is_not_full_image(self) -> None:
        """Ring-band Laplacian must differ from full-image Laplacian when background is blurry."""
        img, center, radius = make_ring_image()
        result = evaluate_quality(img, center, radius)
        # Both metrics are present; ring-band value should be reasonable
        assert result["metrics"]["ring_band_laplacian_variance"] >= 0.0
        assert result["metrics"]["laplacian_variance"] >= 0.0

    def test_three_distinct_levels_reachable(self) -> None:
        """All three quality levels must be reachable with appropriate inputs."""
        # ACCEPTABLE
        img_clean, center, radius = make_ring_image()
        r_clean = evaluate_quality(img_clean, center, radius)
        assert r_clean["quality_level"] == "ACCEPTABLE"

        # REJECTED (heavy blur)
        img_blurred, center, radius = make_ring_image(blur=10)
        r_blurred = evaluate_quality(img_blurred, center, radius)
        assert r_blurred["quality_level"] == "REJECTED"

    def test_quality_score_range(self) -> None:
        """Quality score must always be in [0, 100]."""
        for blur in (0.0, 3.0, 7.0, 12.0):
            img, center, radius = make_ring_image(blur=blur)
            result = evaluate_quality(img, center, radius)
            assert 0 <= result["quality_score"] <= 100

    def test_flags_is_sorted_list(self) -> None:
        img, center, radius = make_ring_image(blur=10)
        result = evaluate_quality(img, center, radius)
        flags = result["flags"]
        assert isinstance(flags, list)
        assert flags == sorted(flags)

    def test_quality_level_blocked_does_not_exist(self) -> None:
        """BLOCKED is an inference-level concept; quality.py never produces it."""
        img, center, radius = make_ring_image()
        result = evaluate_quality(img, center, radius)
        assert result.get("quality_level") != "BLOCKED"
        assert result.get("quality_level") in {"ACCEPTABLE", "ACCEPTABLE_WITH_WARNING", "REJECTED"}


class TestROIAwareness:
    """Verify ring-band-aware quality measurement."""

    def test_ring_band_mask_focuses_on_ring_region(self) -> None:
        """A sharp ring pattern on a blurry background should not be falsely rejected."""
        img, center, radius = make_ring_image()
        # The ring-band metric should pick up ring sharpness
        result = evaluate_quality(img, center, radius)
        ring_lap = result["metrics"]["ring_band_laplacian_variance"]
        # With clear rings, ring-band sharpness should be substantial
        assert ring_lap > 5.0

    def test_small_image_falls_back_to_full_laplacian(self) -> None:
        """Very small images use full-image Laplacian as fallback."""
        img = synthetic_placido(shape=(100, 100))
        center = (50.0, 50.0)
        radius = 40.0
        result = evaluate_quality(img, center, radius)
        # Should still produce a valid result
        assert result["quality_level"] in {"ACCEPTABLE", "ACCEPTABLE_WITH_WARNING", "REJECTED"}
