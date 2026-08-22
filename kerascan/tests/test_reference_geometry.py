"""Deterministic tests for self-fitted Placido reference geometry.

These image-space tests define no disease labels or clinical thresholds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from kerascan.config import EngineConfig
from kerascan.inference import KerascanEngine
from kerascan.reference_geometry import analyse_reference_geometry
from kerascan.synthetic import synthetic_placido
from kerascan.visualization import MULTIRING_AGREEMENT_Y_LABEL


class _ReferenceConfig:
    reference_deviation_magnitude_fraction = 0.08
    min_coherent_ring_run = 2
    min_reference_sector_angular_samples = 3


def _angles(count: int = 72) -> np.ndarray:
    return np.linspace(0.0, 360.0, count, endpoint=False)


def _base(ring_count: int = 7, meridians: int = 72) -> np.ndarray:
    pitch = np.arange(7.0, 7.0 + ring_count - 1, dtype=float)
    radii = [np.full(meridians, 18.0)]
    for gap in pitch:
        radii.append(radii[-1] + gap)
    return np.asarray(radii)


def _analyse(
    radii: np.ndarray,
    *,
    observed: np.ndarray | None = None,
    interpolated: np.ndarray | None = None,
    rejected: np.ndarray | None = None,
    expected_ring_count: int | None | str = "detected",
    min_direct_coverage: float = 0.55,
) -> dict:
    radii = np.asarray(radii, dtype=float)
    finite = np.isfinite(radii)
    observed = finite if observed is None else np.asarray(observed, dtype=bool)
    interpolated = np.zeros_like(finite) if interpolated is None else np.asarray(interpolated, dtype=bool)
    rejected = np.zeros_like(finite) if rejected is None else np.asarray(rejected, dtype=bool)
    expected = radii.shape[0] if expected_ring_count == "detected" else expected_ring_count
    return analyse_reference_geometry(
        radii,
        _angles(radii.shape[1]),
        observed,
        interpolated=interpolated,
        rejected=rejected,
        centre=(101.5, 99.25),
        expected_ring_count=expected,
        min_direct_coverage=min_direct_coverage,
        config=_ReferenceConfig(),
    )


def _matrix(values) -> np.ndarray:
    return np.asarray(
        [[np.nan if value is None else value for value in row] for row in values],
        dtype=float,
    )


def test_perfect_concentric_stack_has_near_zero_reference_deviation():
    result = _analyse(_base())
    reference = result["reference_geometry"]
    assert reference["concentric_reference"]["valid"] is True
    assert reference["smooth_reference"]["valid"] is True
    assert np.nanmax(np.abs(_matrix(reference["circle_deviation"]["signed_pixels"]))) < 1e-10
    assert np.nanmax(np.abs(_matrix(reference["smooth_deviation"]["signed_pixels"]))) < 1e-10


def test_nonuniform_radial_pitch_is_preserved_not_forced_equal():
    radii = _base()
    result = _analyse(radii)["reference_geometry"]
    reference_gaps = np.diff(result["concentric_reference"]["radii_pixels"])
    assert np.allclose(reference_gaps, np.arange(7.0, 13.0))
    assert not np.allclose(reference_gaps, reference_gaps[0])
    assert result["concentric_reference"]["valid"] is True


def test_smooth_ellipse_has_smaller_smooth_than_circle_residual():
    theta = np.deg2rad(_angles())
    radii = _base() * (1.0 + 0.09 * np.cos(2.0 * theta))[None, :]
    reference = _analyse(radii)["reference_geometry"]
    circle = np.nanmedian(np.abs(_matrix(reference["circle_deviation"]["signed_pixels"])))
    smooth = np.nanmedian(np.abs(_matrix(reference["smooth_deviation"]["signed_pixels"])))
    assert circle > 1.0
    assert smooth < circle * 0.05


def test_refined_centre_is_used_for_reference_stack():
    reference = _analyse(_base())["reference_geometry"]
    assert reference["centre"] == {"x": 101.5, "y": 99.25}
    assert np.nanmax(np.abs(_matrix(reference["smooth_deviation"]["signed_pixels"]))) < 1e-10


def test_one_isolated_outlier_does_not_move_robust_reference():
    radii = _base()
    original = radii[3, 17]
    radii[3, 17] += 18.0
    reference = _analyse(radii)["reference_geometry"]
    assert np.isclose(reference["concentric_reference"]["radii_pixels"][3], _base()[3, 0])
    assert reference["smooth_reference"]["number_of_rejected_outliers_by_ring"][3] >= 1
    residual = _matrix(reference["smooth_deviation"]["signed_pixels"])
    assert residual[3, 17] > original * 0.1


def test_local_inward_deformation_in_one_ring_is_not_multiring_coherence():
    radii = _base()
    radii[3, 12:20] -= 3.0
    reference = _analyse(radii)["reference_geometry"]
    assert reference["cross_ring_coherent_sectors"] == []
    residual = _matrix(reference["smooth_deviation"]["signed_pixels"])
    assert np.nanmedian(residual[3, 12:20]) < 0.0


def test_neighbouring_inward_deformations_create_supported_sector():
    radii = _base()
    radii[2:5, 12:21] -= 3.0
    sectors = _analyse(radii)["reference_geometry"]["cross_ring_coherent_sectors"]
    inward = [sector for sector in sectors if sector["direction"] == "INWARD"]
    assert len(inward) == 1
    assert inward[0]["affected_ring_count"] >= 3
    assert inward[0]["direct_observation_fraction"] == 1.0


def test_neighbouring_outward_deformations_create_supported_sector():
    radii = _base()
    radii[2:5, 30:39] += 3.0
    sectors = _analyse(radii)["reference_geometry"]["cross_ring_coherent_sectors"]
    outward = [sector for sector in sectors if sector["direction"] == "OUTWARD"]
    assert len(outward) == 1
    assert outward[0]["affected_ring_count"] >= 3


def test_progressive_inner_to_outer_deviation_is_retained():
    radii = _base()
    radii[:, 20:29] += np.arange(radii.shape[0])[:, None] * 1.2
    reference = _analyse(radii)["reference_geometry"]
    residual = _matrix(reference["circle_deviation"]["signed_pixels"])
    assert np.nanmedian(residual[-1, 20:29]) > np.nanmedian(residual[1, 20:29])
    assert max(reference["cross_ring_coherence_by_meridian"]["longest_outward_ring_run"]) >= 3


def test_missing_inner_ring_sector_remains_missing():
    radii = _base(); radii[0, 5:12] = np.nan
    reference = _analyse(radii)["reference_geometry"]
    assert np.isnan(_matrix(reference["circle_deviation"]["signed_pixels"])[0, 5:12]).all()
    assert reference["circle_deviation"]["observation_state"][0][5] == "MISSING"


def test_missing_middle_ring_sector_remains_missing():
    radii = _base(); radii[3, 5:12] = np.nan
    reference = _analyse(radii)["reference_geometry"]
    assert np.isnan(_matrix(reference["smooth_deviation"]["signed_pixels"])[3, 5:12]).all()


def test_missing_outer_ring_sector_remains_missing():
    radii = _base(); radii[-1, 5:12] = np.nan
    reference = _analyse(radii)["reference_geometry"]
    assert np.isnan(_matrix(reference["circle_deviation"]["normalized"])[-1, 5:12]).all()


def test_cross_zero_degree_sector_is_one_sector():
    radii = _base()
    indices = np.r_[0:5, 68:72]
    radii[2:5, indices] -= 3.0
    sectors = _analyse(radii)["reference_geometry"]["cross_ring_coherent_sectors"]
    inward = [sector for sector in sectors if sector["direction"] == "INWARD"]
    assert len(inward) == 1
    assert inward[0]["crosses_zero_degree_seam"] is True
    assert inward[0]["start_degrees"] > inward[0]["end_degrees"]


def test_global_scaling_preserves_normalized_deviations():
    radii = _base(); radii[2:5, 14:22] -= 2.5
    base = _analyse(radii)["reference_geometry"]
    scaled = _analyse(radii * 3.4)["reference_geometry"]
    assert np.allclose(
        _matrix(base["smooth_deviation"]["normalized"]),
        _matrix(scaled["smooth_deviation"]["normalized"]),
        equal_nan=True,
    )


def test_rotation_moves_coherent_sector_without_changing_magnitude():
    radii = _base(); radii[2:5, 10:18] += 3.0
    base = _analyse(radii)["reference_geometry"]
    rotated = _analyse(np.roll(radii, 9, axis=1))["reference_geometry"]
    base_sector = [s for s in base["cross_ring_coherent_sectors"] if s["direction"] == "OUTWARD"][0]
    rotated_sector = [s for s in rotated["cross_ring_coherent_sectors"] if s["direction"] == "OUTWARD"][0]
    assert np.isclose(rotated_sector["start_degrees"], (base_sector["start_degrees"] + 45.0) % 360.0)
    assert np.isclose(rotated_sector["median_normalized_deviation"], base_sector["median_normalized_deviation"])


def test_reference_ring_ordering_failure_is_invalid_not_sorted():
    radii = _base(); radii[3] = radii[2] - 1.0
    reference = _analyse(radii)["reference_geometry"]
    assert reference["concentric_reference"]["valid"] is False
    assert "non_positive_observed_spacing" in reference["invalid_reason_codes"]
    assert reference["concentric_reference"]["radii_pixels"][3] < reference["concentric_reference"]["radii_pixels"][2]


def test_smooth_reference_crossing_is_invalid():
    theta = np.deg2rad(_angles())
    radii = np.vstack((
        np.full(72, 15.0),
        30.0 + 14.0 * np.cos(theta),
        40.0 - 14.0 * np.cos(theta),
        np.full(72, 60.0),
    ))
    observed = np.ones_like(radii, dtype=bool)
    crossing = radii[2] <= radii[1]
    observed[1:3, crossing] = False
    radii[1:3, crossing] = np.nan
    reference = _analyse(radii, observed=observed, min_direct_coverage=0.50)["reference_geometry"]
    assert reference["smooth_reference"]["valid"] is False
    assert "smooth_reference_crossing" in reference["invalid_reason_codes"]


def test_insufficient_ring_coverage_marks_ring_invalid_without_circle():
    radii = _base(); radii[3, 20:] = np.nan
    reference = _analyse(radii)["reference_geometry"]
    assert reference["concentric_reference"]["valid_by_ring"][3] is False
    assert reference["concentric_reference"]["radii_pixels"][3] is None
    assert reference["smooth_reference"]["coefficients_by_ring"][3] is None


def test_interpolated_only_sector_never_becomes_observed_deviation():
    radii = _base()
    observed = np.ones_like(radii, dtype=bool)
    interpolated = np.zeros_like(radii, dtype=bool)
    observed[2, 8:15] = False
    interpolated[2, 8:15] = True
    reference = _analyse(radii, observed=observed, interpolated=interpolated)["reference_geometry"]
    assert np.isnan(_matrix(reference["circle_deviation"]["signed_pixels"])[2, 8:15]).all()
    assert reference["circle_deviation"]["observation_state"][2][8] == "INTERPOLATED"


def test_duplicate_ring_assignment_invalidates_reference_geometry():
    radii = _base(); radii[3, 9] = radii[2, 9]
    reference = _analyse(radii)["reference_geometry"]
    assert reference["valid"] is False
    assert "non_positive_observed_spacing" in reference["invalid_reason_codes"]


def test_reference_outputs_do_not_bypass_hardware_or_threshold_gates(tmp_path: Path):
    result = KerascanEngine().analyze(synthetic_placido(rings=8), tmp_path)
    assert result["screening_result"] == "ANALYSIS_BLOCKED"
    assert result["classification_performed"] is False
    assert result["reference_geometry"]["validated_normal_reference"] is False
    for filename in (
        "observed_vs_concentric_reference.png",
        "observed_vs_smooth_reference.png",
        "radial_deviation_vectors.png",
        "circle_deviation_heatmap.png",
        "smooth_residual_heatmap.png",
        "reference_spacing_residual_heatmap.png",
        "full_stack_reference_comparison.png",
    ):
        assert (tmp_path / filename).is_file()


def test_legacy_multiring_plot_is_not_labelled_standard_deviation():
    assert "std" not in MULTIRING_AGREEMENT_Y_LABEL.lower()
    assert "coherent" in MULTIRING_AGREEMENT_Y_LABEL.lower()


def test_unverified_reference_is_provisional_not_clinical():
    reference = _analyse(_base(), expected_ring_count=None)["reference_geometry"]
    assert reference["ring_count_verified"] is False
    assert reference["validated_normal_reference"] is False
    assert reference["classification_performed"] is False
    assert reference["reference_type"] == "SELF_FITTED_ENGINEERING_REFERENCE"


def test_verified_stack_without_approved_thresholds_remains_not_calibrated():
    from kerascan.geometry import compute_geometry

    radii = _base()
    result = compute_geometry(
        radii,
        _angles(),
        np.ones_like(radii, dtype=bool),
        0.55,
        _ReferenceConfig(),
        expected_ring_count=radii.shape[0],
        interpolated=np.zeros_like(radii, dtype=bool),
        rejected=np.zeros_like(radii, dtype=bool),
    )
    assert result["geometry_status"] == "NOT_CALIBRATED"
    assert result["reference_geometry"]["valid"] is True
    assert result["reference_geometry"]["classification_performed"] is False

