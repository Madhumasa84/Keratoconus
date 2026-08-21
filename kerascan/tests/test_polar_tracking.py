"""Deterministic regression tests for polar peak localisation and ring tracking."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from kerascan.config import EngineConfig, RadialConfig, TrackingConfig
from kerascan.graph_tracking import track_rings
from kerascan.inference import KerascanEngine
from kerascan.radial_scan import RadialResult, radial_scan
from kerascan.synthetic import synthetic_placido


def _ring_image(
    *,
    elliptical: bool = False,
    decentred: bool = False,
    distorted: bool = False,
    missing: tuple[int, int] | None = None,
    blur: float = 0.0,
    uneven: bool = False,
    noise: float = 0.0,
    lid: bool = False,
    extra_boundary: bool = False,
    bridge: bool = False,
    rotation: float = 0.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float], float, int]:
    """Synthetic image-space rings only; no disease label is implied."""
    rng = np.random.default_rng(17)
    size, centre, count = 360, (180.0, 180.0), 7
    image = np.full((size, size), 34.0, dtype=np.float32)
    if uneven:
        image += np.linspace(-14, 18, size, dtype=np.float32)[None, :]
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    for ring in range(count):
        radius = (24 + ring * 17) * scale
        local_centre = (centre[0] + (ring * .45 if decentred else 0), centre[1] - (ring * .30 if decentred else 0))
        deformation = 1.0
        if elliptical:
            deformation = deformation + .08 * np.cos(2 * (theta + rotation))
        if distorted:
            deformation = deformation + .045 * np.sin(3 * theta + rotation)
        radial = radius * deformation
        degrees = (np.rad2deg(theta) % 360)
        visible = np.ones_like(theta, dtype=bool)
        if missing is not None:
            visible &= ~((degrees >= missing[0]) & (degrees <= missing[1]))
        points = np.column_stack((local_centre[0] + radial * np.cos(theta), local_centre[1] + radial * np.sin(theta)))
        points = points[visible].astype(np.int32)
        if len(points) > 1:
            cv2.polylines(image, [points], False, 210, 2, cv2.LINE_AA)
    if extra_boundary:
        cv2.circle(image, (180, 180), 158, 190, 2, cv2.LINE_AA)
    if bridge:
        # Deliberate binary-mask bridge artefact. The polar intensity method may
        # see local contamination, but must not reuse a point as two rings.
        cv2.line(image, (180, 180), (310, 210), 210, 3, cv2.LINE_AA)
    if lid:
        cv2.rectangle(image, (0, 0), (size, 75), 24, -1)
    if noise:
        image += rng.normal(0, noise, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), centre, 151.0, count


def _candidate_scan(
    *,
    meridians: int = 72,
    short_gap: tuple[int, int] | None = None,
    long_gap: tuple[int, int] | None = None,
    duplicate: bool = False,
) -> RadialResult:
    references = np.asarray([20.0, 36.0, 52.0, 68.0], dtype=float)
    angles = np.linspace(0.0, 360.0, meridians, endpoint=False)
    candidates: list[np.ndarray] = []
    strengths: list[np.ndarray] = []
    for angle in range(meridians):
        values = references.copy()
        if short_gap is not None and short_gap[0] <= angle < short_gap[1]:
            values = np.delete(values, 1)
        if long_gap is not None and long_gap[0] <= angle < long_gap[1]:
            values = np.delete(values, 2)
        if duplicate:
            values = np.insert(values, 1, values[0])
        candidates.append(values)
        strengths.append(np.full(len(values), 10.0))
    positions = np.arange(4.0, 90.0, 1.0)
    polar = np.zeros((meridians, len(positions)), dtype=float)
    return RadialResult(angles, positions, polar, polar, candidates, strengths, references, "provisional_polar_profile", 1.0, 0.0)


def _assert_valid_order(tracked) -> None:
    assert not np.any(np.isfinite(tracked.radii) & (tracked.radii <= 0))
    spacing = np.diff(tracked.radii, axis=0)
    assert not np.any(np.isfinite(spacing) & (spacing <= 0))
    for angle in range(tracked.radii.shape[1]):
        values = tracked.radii[:, angle]
        values = values[np.isfinite(values)]
        assert np.all(np.diff(values) > 0)
    for ring in range(tracked.radii.shape[0]):
        direct = np.flatnonzero(tracked.observed[ring])
        if len(direct) > 1:
            jumps = np.abs(np.diff(tracked.radii[ring, direct]))
            assert np.all(jumps < 12.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # perfect concentric rings
        {"elliptical": True},
        {"decentred": True},
        {"distorted": True},
        {"blur": 1.0},
        {"uneven": True},
        {"noise": 5.0},
        {"lid": True},
        {"extra_boundary": True},
        {"bridge": True},
        {"rotation": .45, "scale": .92},  # rotation and scale change
    ],
)
def test_intensity_peak_tracking_preserves_strict_order(kwargs):
    image, centre, outer, count = _ring_image(**kwargs)
    config = RadialConfig(meridians=180, expected_ring_count=count, max_radial_jump_px=10.0)
    scan = radial_scan(image, centre, outer, config)
    tracked = track_rings(scan, TrackingConfig(min_direct_coverage=.40, min_tracking_confidence=.25), config)
    assert scan.polar_image.shape[0] == 180
    assert scan.ring_count_source == "verified_device_config"
    assert len(scan.reference_radii) == count
    _assert_valid_order(tracked)


def test_short_gap_is_interpolated_but_long_gap_is_preserved_missing():
    config = RadialConfig(meridians=72, max_radial_jump_px=8.0)
    short = track_rings(_candidate_scan(short_gap=(10, 13)), radial=config)
    assert np.all(short.interpolated[1, 10:13])
    assert np.all(np.isfinite(short.radii[1, 10:13]))
    long = track_rings(_candidate_scan(long_gap=(20, 35)), radial=config)
    assert not np.any(long.interpolated[2, 20:35])
    assert np.all(np.isnan(long.radii[2, 20:35]))
    _assert_valid_order(short)
    _assert_valid_order(long)


def test_duplicate_candidates_and_potential_identity_crossing_cannot_duplicate_points():
    config = RadialConfig(meridians=72, max_radial_jump_px=8.0)
    tracked = track_rings(_candidate_scan(duplicate=True), radial=config)
    assert tracked.duplicate_removals >= 72
    _assert_valid_order(tracked)
    for angle in range(tracked.radii.shape[1]):
        values = tracked.radii[:, angle]
        values = values[np.isfinite(values)]
        assert len(values) == len(np.unique(values))


def test_cyclic_boundary_is_tracked_without_cross_seam_interpolation():
    config = RadialConfig(meridians=72, max_radial_jump_px=8.0)
    scan = _candidate_scan(short_gap=(0, 3))
    tracked = track_rings(scan, radial=config)
    assert np.all(np.isnan(tracked.radii[1, :3]))
    assert not np.any(tracked.interpolated[1, :3])
    _assert_valid_order(tracked)


def test_engine_uses_single_acquisition_score_and_skips_without_verified_hardware_count():
    image = synthetic_placido(rings=8)
    unconfigured = KerascanEngine().analyze(image)
    assert unconfigured["classification_skipped"] is True
    assert unconfigured["failure_stage"] == "CONFIGURATION"
    configured = KerascanEngine(EngineConfig(radial=RadialConfig(expected_ring_count=8, meridians=180))).analyze(image)
    assert configured["classification_performed"] is False
    assert configured["classification_skipped"] is True
    assert configured["geometry_status"] == "NOT_CALIBRATED"
    assert configured["quality"]["quality_score"] == configured["acquisition_quality"]["score"]


def test_acquisition_and_algorithm_failures_have_different_messages():
    acquisition = KerascanEngine().analyze(synthetic_placido(darkness=.96))
    assert acquisition["failure_stage"] == "ACQUISITION"
    assert acquisition["message"].startswith("Recapture required:")
    segmentation = KerascanEngine(EngineConfig(radial=RadialConfig(min_peak_prominence=1e6))).analyze(synthetic_placido())
    assert segmentation["failure_stage"] == "SEGMENTATION"
    assert segmentation["classification_skipped"] is True
    assert segmentation["message"].startswith("Automated ring segmentation failed")
    tracking = KerascanEngine(
        EngineConfig(radial=RadialConfig(expected_ring_count=8, meridians=180), tracking=TrackingConfig(min_direct_coverage=.999))
    ).analyze(synthetic_placido(rings=8, noise=15))
    assert tracking["failure_stage"] == "TRACKING"
    assert tracking["classification_skipped"] is True
    assert tracking["message"].startswith("Reliable ring identities could not be reconstructed")


@pytest.mark.parametrize("name", ["aleft.png", "aright.png"])
def test_real_sample_regression_has_no_invalid_geometry_or_classification(tmp_path, name):
    source = Path(__file__).resolve().parents[2] / "sample_images" / name
    assert source.exists()
    result = KerascanEngine().analyze(source, tmp_path / name[:-4])
    assert result["roi"]["method"] != "fallback_image_center"
    assert result["classification_skipped"] is True
    assert result["acquisition_quality"]["status"] in {"ACCEPTABLE", "FAIL"}
    assert result["segmentation"]["status"] in {"PASS", "FAIL", "NOT_RUN"}
    assert result["tracking"]["status"] in {"PASS", "FAIL", "NOT_RUN"}
    tracked = result["_artifacts"].get("tracking_result")
    if tracked is not None:
        _assert_valid_order(tracked)
    assert (tmp_path / name[:-4] / "result.json").exists()
