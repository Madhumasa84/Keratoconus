"""The analysis must refuse regions that are not actually a Placido pattern.

A concentric detection on skin, eyelashes or eyebrow produces a confident but
meaningless result -- including a false NORMAL-LIKE, the worst outcome for a
screening tool, because the child is cleared and never referred.

This is deliberately NOT a quality judgement: it is enforced even when the
acquisition-quality gate is disabled for clinician-selected captures, because
"no cornea in the analysed region" is not the same as "a poor photo of one".
"""
from __future__ import annotations

import numpy as np
import pytest

from kerascan.config import EngineConfig, QualityConfig
from kerascan.inference import KerascanEngine
from kerascan.quality import mire_contrast_amplitude
from kerascan.synthetic import synthetic_placido


def _smooth_skin(size: int = 480) -> np.ndarray:
    """A smoothly shaded patch, as skin or an eyebrow presents to the detector."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    centre = size / 2
    radius = np.hypot(xx - centre, yy - centre)
    # Gentle concentric shading: near-perfectly circular, but no specular mires.
    shade = 150 + 18 * np.cos(radius / 26.0)
    return np.dstack([np.clip(shade, 0, 255).astype(np.uint8)] * 3)


def test_mire_amplitude_separates_a_ring_pattern_from_smooth_shading():
    rings = synthetic_placido(rings=8)
    gray_rings = rings[:, :, 0].astype(np.uint8) if rings.ndim == 3 else rings
    skin = _smooth_skin()[:, :, 0]

    centre = (gray_rings.shape[1] / 2, gray_rings.shape[0] / 2)
    ring_amplitude = mire_contrast_amplitude(gray_rings, centre, gray_rings.shape[0] * 0.40)
    skin_amplitude = mire_contrast_amplitude(skin, (skin.shape[1] / 2, skin.shape[0] / 2), skin.shape[0] * 0.40)

    assert ring_amplitude > skin_amplitude
    assert skin_amplitude < QualityConfig().min_mire_contrast_amplitude


def test_smooth_shading_is_refused_even_with_the_quality_gate_disabled():
    """The decisive case: quality gating off must NOT let skin through.

    Disabling the quality gate is what allowed a real sample photo of a forehead
    to be reported as NORMAL-LIKE. This gate is the thing standing in its place.
    """
    config = EngineConfig(quality=QualityConfig(enforce_gate=False))
    result = KerascanEngine(config).analyze(_smooth_skin())

    assert result["screening_result"] not in {"NORMAL-LIKE", "SUSPICIOUS", "INDETERMINATE"}
    assert result["classification_performed"] is False
    assert "no_placido_pattern_located" in result["acquisition_quality"]["flags"]


def test_a_real_ring_pattern_still_passes_the_gate():
    config = EngineConfig(quality=QualityConfig(enforce_gate=False))
    result = KerascanEngine(config).analyze(synthetic_placido(rings=8))
    assert "no_placido_pattern_located" not in result["acquisition_quality"]["flags"]
    assert result["failure_stage"] != "ACQUISITION"


def test_the_app_layer_also_refuses_smooth_shading():
    """The gate must hold through the app, which disables quality gating.

    Regression for a real defect: with quality gating off, three patient photos
    that had locked onto skin were analysed anyway, and one was reported
    NORMAL_LIKE from an image of a forehead. Those photos are not retained in
    this repository, so the case is reproduced synthetically here.
    """
    import cv2

    from app.services.screening_service import ScreeningService

    service = ScreeningService()
    assert service._enforce_quality_gate is False, "app deliberately runs with quality gating off"

    skin = _smooth_skin()
    amplitude = mire_contrast_amplitude(
        cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY), (skin.shape[1] / 2, skin.shape[0] / 2), skin.shape[0] * 0.40
    )
    assert amplitude < service._engine_config.quality.min_mire_contrast_amplitude

    result = service._get_image_engine().analyze(skin)
    assert result["screening_result"] not in {"NORMAL-LIKE", "SUSPICIOUS", "INDETERMINATE"}
