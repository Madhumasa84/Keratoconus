"""Deterministic oracles for complete Placido-ring stack geometry.

These tests use synthetic image-space radii only.  They carry no disease label
and do not define or imply clinical screening thresholds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kerascan.inference import KerascanEngine
from kerascan.geometry import compute_geometry
from kerascan.synthetic import synthetic_placido


class _EngineeringConfig:
    thresholds = None
    compression_magnitude_fraction = 0.08
    expansion_magnitude_fraction = 0.08
    min_coherent_pair_run = 3
    min_sector_angular_samples = 3
    max_missing_sector_fraction = 0.20


def _angles(count: int = 72) -> np.ndarray:
    return np.linspace(0.0, 360.0, count, endpoint=False)


def _radii_from_spacing(spacing: np.ndarray, inner_radius: float = 20.0) -> np.ndarray:
    spacing = np.asarray(spacing, dtype=float)
    inner = np.full((1, spacing.shape[1]), inner_radius, dtype=float)
    return np.vstack((inner, inner_radius + np.cumsum(spacing, axis=0)))


def _base_radii(ring_count: int = 7, meridians: int = 72) -> np.ndarray:
    pitch = np.arange(8.0, 8.0 + ring_count - 1, dtype=float)
    return _radii_from_spacing(np.repeat(pitch[:, None], meridians, axis=1))


def _analyse(
    radii: np.ndarray,
    *,
    expected_ring_count: int | None = None,
    observed: np.ndarray | None = None,
    interpolated: np.ndarray | None = None,
    rejected: np.ndarray | None = None,
) -> dict:
    radii = np.asarray(radii, dtype=float)
    expected = radii.shape[0] if expected_ring_count is None else expected_ring_count
    observed = np.isfinite(radii) if observed is None else np.asarray(observed, dtype=bool)
    interpolated = np.zeros_like(observed) if interpolated is None else np.asarray(interpolated, dtype=bool)
    rejected = np.zeros_like(observed) if rejected is None else np.asarray(rejected, dtype=bool)
    return compute_geometry(
        radii,
        _angles(radii.shape[1]),
        observed,
        0.55,
        _EngineeringConfig(),
        expected_ring_count=expected,
        interpolated=interpolated,
        rejected=rejected,
    )


def _array(values) -> np.ndarray:
    return np.asarray([[np.nan if value is None else value for value in row] for row in values], dtype=float)


def _vector(values) -> np.ndarray:
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def test_every_expected_ring_and_adjacent_pair_is_represented():
    result = _analyse(_base_radii(ring_count=8))
    stack = result["full_stack_analysis"]
    assert stack["analysed_ring_indices"] == list(range(8))
    assert stack["analysed_ring_pair_count"] == 7
    assert stack["spacing_matrix_shape"] == [7, 72]
    assert len(stack["ring_pair_completeness"]) == 7


def test_naturally_nonuniform_radial_pitch_has_zero_angular_irregularity():
    pitch = np.asarray([6.0, 8.0, 11.0, 15.0, 20.0, 27.0])
    radii = _radii_from_spacing(np.repeat(pitch[:, None], 72, axis=1))
    stack = _analyse(radii)["full_stack_analysis"]
    assert np.allclose(stack["baseline_spacing_by_pair"], pitch)
    assert np.allclose(stack["angular_spacing_variation_by_pair"], 0.0)
    assert np.allclose(_array(stack["normalized_inter_ring_spacing_matrix"]), 1.0)
    assert stack["compression_sectors"] == []
    assert stack["expansion_sectors"] == []


def test_smooth_elliptical_stack_remains_ordered_and_complete():
    angles = np.deg2rad(_angles())
    factor = 1.0 + 0.10 * np.cos(2.0 * angles)
    radii = _base_radii() * factor[None, :]
    result = _analyse(radii)
    stack = result["full_stack_analysis"]
    assert result["geometry_status"] == "NOT_CALIBRATED"
    assert stack["complete_stack_available"] is True
    assert np.all(np.diff(radii, axis=0) > 0)
    assert np.all(_vector(stack["angular_spacing_variation_by_pair"]) > 0)


def test_global_scaling_does_not_change_normalized_features():
    radii = _base_radii()
    spacing = np.diff(radii, axis=0)
    spacing[1:4, 9:18] *= 0.75
    radii = _radii_from_spacing(spacing)
    base = _analyse(radii)["full_stack_analysis"]
    scaled = _analyse(radii * 3.7)["full_stack_analysis"]
    assert np.allclose(
        _array(base["normalized_inter_ring_spacing_matrix"]),
        _array(scaled["normalized_inter_ring_spacing_matrix"]),
        equal_nan=True,
    )
    assert np.allclose(
        _vector(base["radial_stack_deviation_by_meridian"]),
        _vector(scaled["radial_stack_deviation_by_meridian"]),
        equal_nan=True,
    )


def test_rotation_moves_sector_but_preserves_aggregate_magnitude():
    spacing = np.diff(_base_radii(), axis=0)
    spacing[1:4, 6:14] *= 0.70
    radii = _radii_from_spacing(spacing)
    base = _analyse(radii)["full_stack_analysis"]
    rotated = _analyse(np.roll(radii, 11, axis=1))["full_stack_analysis"]
    assert np.isclose(
        base["full_stack_summary"]["robust_angular_variation"],
        rotated["full_stack_summary"]["robust_angular_variation"],
    )
    assert len(base["compression_sectors"]) == len(rotated["compression_sectors"]) == 1
    shift = 11 * (360.0 / 72)
    assert np.isclose(
        rotated["compression_sectors"][0]["start_degrees"],
        (base["compression_sectors"][0]["start_degrees"] + shift) % 360.0,
    )


def test_one_isolated_abnormal_pair_has_low_coherence_and_no_stack_sector():
    spacing = np.diff(_base_radii(), axis=0)
    spacing[2, 10:18] *= 0.60
    stack = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]
    coherence = stack["neighbouring_ring_coherence"]
    assert max(coherence["longest_compressed_pair_run"]) == 1
    assert stack["compression_sectors"] == []


def test_three_neighbouring_compressed_pairs_form_one_supported_sector():
    spacing = np.diff(_base_radii(), axis=0)
    spacing[1:4, 10:20] *= 0.70
    stack = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]
    sector = stack["compression_sectors"][0]
    assert max(stack["neighbouring_ring_coherence"]["longest_compressed_pair_run"]) >= 3
    assert sector["first_affected_ring_pair"] == 1
    assert sector["last_affected_ring_pair"] == 3
    assert sector["direct_observation_fraction"] == 1.0


def test_three_neighbouring_expanded_pairs_form_one_supported_sector():
    spacing = np.diff(_base_radii(), axis=0)
    spacing[2:5, 22:31] *= 1.30
    stack = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]
    sector = stack["expansion_sectors"][0]
    assert max(stack["neighbouring_ring_coherence"]["longest_expanded_pair_run"]) >= 3
    assert sector["first_affected_ring_pair"] == 2
    assert sector["last_affected_ring_pair"] == 4


def test_inner_only_compression_identifies_inner_pairs():
    spacing = np.diff(_base_radii(), axis=0)
    spacing[:3, 15:24] *= 0.70
    sector = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]["compression_sectors"][0]
    assert sector["first_affected_ring_pair"] == 0
    assert sector["last_affected_ring_pair"] == 2


def test_outer_only_compression_identifies_outer_pairs():
    spacing = np.diff(_base_radii(), axis=0)
    spacing[-3:, 15:24] *= 0.70
    sector = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]["compression_sectors"][0]
    assert sector["first_affected_ring_pair"] == spacing.shape[0] - 3
    assert sector["last_affected_ring_pair"] == spacing.shape[0] - 1


def test_progressive_compression_accumulates_toward_outer_ring():
    spacing = np.diff(_base_radii(), axis=0)
    factors = np.asarray([0.98, 0.94, 0.90, 0.86, 0.82, 0.78])
    spacing[:, 12:20] *= factors[:, None]
    stack = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]
    residual = _array(stack["normalized_cumulative_residuals"])
    assert abs(residual[-1, 15]) > abs(residual[2, 15])
    assert residual[-1, 15] < 0


def test_opposite_compression_and_expansion_can_coexist_at_same_meridian():
    spacing = np.diff(_base_radii(ring_count=8), axis=0)
    spacing[:3, 20:28] *= 0.70
    spacing[4:7, 20:28] *= 1.30
    stack = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]
    assert len(stack["compression_sectors"]) == 1
    assert len(stack["expansion_sectors"]) == 1
    assert stack["compression_sectors"][0]["start_degrees"] == stack["expansion_sectors"][0]["start_degrees"]


def test_missing_inner_ring_invalidates_first_dependent_pair():
    radii = _base_radii()
    radii[0] = np.nan
    stack = _analyse(radii)["full_stack_analysis"]
    assert stack["ring_pair_completeness"][0]["observed_fraction"] == 0.0
    assert stack["complete_stack_available"] is False


def test_missing_middle_ring_invalidates_both_dependent_pairs():
    radii = _base_radii()
    radii[3] = np.nan
    stack = _analyse(radii)["full_stack_analysis"]
    assert stack["ring_pair_completeness"][2]["observed_fraction"] == 0.0
    assert stack["ring_pair_completeness"][3]["observed_fraction"] == 0.0
    assert stack["analysed_ring_pair_count"] == radii.shape[0] - 1


def test_missing_outer_ring_reduces_outer_and_full_stack_coverage():
    radii = _base_radii()
    radii[-1] = np.nan
    stack = _analyse(radii)["full_stack_analysis"]
    assert stack["outer_region_summary"]["ring_observed_fraction"] < 1.0
    assert stack["full_stack_summary"]["observed_fraction"] < 1.0
    assert stack["complete_stack_available"] is False


def test_large_outer_missing_sector_cannot_be_hidden_by_inner_rings():
    radii = _base_radii()
    radii[-1, 10:30] = np.nan
    stack = _analyse(radii)["full_stack_analysis"]
    assert stack["inner_region_summary"]["ring_observed_fraction"] == 1.0
    assert stack["outer_region_summary"]["ring_observed_fraction"] < 1.0
    assert stack["complete_stack_available"] is False
    assert "major_angular_sector_missing" in stack["completeness_reason_codes"]


def test_sector_crossing_zero_degree_seam_remains_one_sector():
    spacing = np.diff(_base_radii(), axis=0)
    sector_indices = np.r_[0:4, 68:72]
    spacing[1:4, sector_indices] *= 0.65
    sectors = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]["compression_sectors"]
    assert len(sectors) == 1
    assert sectors[0]["crosses_zero_degree_seam"] is True
    assert sectors[0]["start_degrees"] > sectors[0]["end_degrees"]


def test_one_extreme_outlier_does_not_become_full_stack_sector():
    spacing = np.diff(_base_radii(), axis=0)
    spacing[2, 8] *= 0.10
    stack = _analyse(_radii_from_spacing(spacing))["full_stack_analysis"]
    assert stack["compression_sectors"] == []
    assert max(stack["neighbouring_ring_coherence"]["longest_compressed_pair_run"]) == 1


def test_duplicate_ring_assignment_is_ungradable():
    radii = _base_radii()
    radii[3, 7] = radii[2, 7]
    result = _analyse(radii)
    assert result["geometry_status"] == "UNGRADABLE"
    assert "non_positive_spacing" in result["reason_codes"]


def test_non_positive_spacing_is_ungradable_without_absolute_value_or_clipping():
    radii = _base_radii()
    radii[4, 9] = radii[3, 9] - 1.0
    result = _analyse(radii)
    assert result["geometry_status"] == "UNGRADABLE"
    assert "non_positive_spacing" in result["reason_codes"]


def test_ring_order_violation_across_a_missing_middle_identity_is_ungradable():
    radii = _base_radii()
    radii[2, 5] = np.nan
    radii[3, 5] = radii[1, 5] - 1.0
    result = _analyse(radii)
    assert result["geometry_status"] == "UNGRADABLE"
    assert "non_monotonic_ring_order" in result["reason_codes"]


def test_missing_verified_expected_ring_count_blocks_but_retains_provisional_features():
    radii = _base_radii()
    result = _analyse(radii, expected_ring_count=None)
    # _analyse normally substitutes the detected count; call the public API
    # explicitly here to exercise the hardware-count gate.
    result = compute_geometry(
        radii,
        _angles(),
        np.ones_like(radii, dtype=bool),
        0.55,
        _EngineeringConfig(),
        expected_ring_count=None,
        interpolated=np.zeros_like(radii, dtype=bool),
        rejected=np.zeros_like(radii, dtype=bool),
    )
    assert result["geometry_status"] == "ANALYSIS_BLOCKED"
    assert result["full_stack_analysis"]["ring_count_verified"] is False
    assert result["full_stack_analysis"]["classification_performed"] is False
    assert result["full_stack_analysis"]["analysed_ring_pair_count"] == radii.shape[0] - 1


def test_missing_approved_thresholds_is_not_calibrated_after_geometry_gates_pass():
    result = _analyse(_base_radii())
    assert result["geometry_status"] == "NOT_CALIBRATED"
    assert result["full_stack_analysis"]["ring_count_verified"] is True
    assert result["full_stack_analysis"]["complete_stack_available"] is True
    assert result["reason_codes"] == ["missing_clinical_thresholds"]


def test_ring_and_pair_states_are_mutually_exclusive_and_sum_to_one():
    radii = _base_radii()
    observed = np.ones_like(radii, dtype=bool)
    interpolated = np.zeros_like(radii, dtype=bool)
    rejected = np.zeros_like(radii, dtype=bool)
    observed[1, 3] = False
    interpolated[1, 3] = True
    observed[2, 4] = False
    radii[2, 4] = np.nan
    observed[3, 5] = False
    radii[3, 5] = np.nan
    rejected[3, 5] = True
    stack = _analyse(radii, observed=observed, interpolated=interpolated, rejected=rejected)["full_stack_analysis"]
    assert stack["state_matrix"][1][3] == "INTERPOLATED"
    assert stack["state_matrix"][2][4] == "MISSING"
    assert stack["state_matrix"][3][5] == "REJECTED"
    for item in stack["ring_completeness"] + stack["ring_pair_completeness"]:
        total = sum(item[name] for name in ("observed_fraction", "interpolated_fraction", "missing_fraction", "rejected_fraction"))
        assert np.isclose(total, 1.0)


def test_interpolated_spacing_does_not_change_directly_observed_baseline():
    radii = _base_radii(ring_count=3)
    observed = np.ones_like(radii, dtype=bool)
    interpolated = np.zeros_like(radii, dtype=bool)
    observed[1, 0] = False
    interpolated[1, 0] = True
    radii[1, 0] = radii[0, 0] + 40.0
    radii[2, 0] = radii[1, 0] + 10.0
    stack = _analyse(radii, observed=observed, interpolated=interpolated)["full_stack_analysis"]
    assert np.isclose(stack["baseline_spacing_by_pair"][0], 8.0)
    assert stack["ring_pair_completeness"][0]["observed_fraction"] < 1.0


def test_cumulative_residual_is_missing_when_an_intermediate_identity_is_missing():
    radii = _base_radii()
    radii[2, 6] = np.nan
    stack = _analyse(radii)["full_stack_analysis"]
    cumulative = _array(stack["cumulative_radial_residuals"])
    assert np.isnan(cumulative[2:, 6]).all()


def test_cumulative_residual_is_missing_across_a_rejected_numeric_identity():
    radii = _base_radii()
    observed = np.ones_like(radii, dtype=bool)
    rejected = np.zeros_like(radii, dtype=bool)
    observed[2, 6] = False
    rejected[2, 6] = True
    stack = _analyse(radii, observed=observed, rejected=rejected)["full_stack_analysis"]
    cumulative = _array(stack["cumulative_radial_residuals"])
    assert np.isnan(cumulative[2:, 6]).all()


def test_one_pair_maximum_cannot_be_suspicious_without_multiring_corroboration():
    class _TestOnlyThresholds:
        suspicious_bounds = {"LOCAL_COMPRESSION": 0.50}
        indeterminate_bounds = {"LOCAL_COMPRESSION": 0.30}

    class _Config(_EngineeringConfig):
        thresholds = _TestOnlyThresholds()

    spacing = np.diff(_base_radii(), axis=0)
    spacing[2, 10:18] *= 0.30
    radii = _radii_from_spacing(spacing)
    result = compute_geometry(
        radii,
        _angles(),
        np.ones_like(radii, dtype=bool),
        0.55,
        _Config(),
        expected_ring_count=radii.shape[0],
        interpolated=np.zeros_like(radii, dtype=bool),
        rejected=np.zeros_like(radii, dtype=bool),
    )
    assert result["geometry_status"] == "INDETERMINATE"
    assert "uncorroborated_single_pair_maximum" in result["reason_codes"]


def test_all_required_explainable_feature_families_have_provenance_fields():
    result = _analyse(_base_radii())
    required = {
        "INTER_RING_SPACING_MATRIX",
        "ANGULAR_INTER_RING_SPACING_VARIATION",
        "RADIAL_STACK_SPACING_CONSISTENCY",
        "NEIGHBOURING_RING_COHERENCE",
        "CUMULATIVE_RADIAL_RESIDUAL",
        "FULL_STACK_DIRECT_COVERAGE",
        "FULL_STACK_COMPRESSION",
        "FULL_STACK_EXPANSION",
        "INNER_MIDDLE_OUTER_CONSISTENCY",
    }
    families = result["explainable_features"]
    assert required <= set(families)
    for name in required:
        assert {"value", "units", "validity", "direct_observation_fraction", "affected_rings", "affected_angles", "reason_if_invalid"} <= set(families[name])


def test_unverified_engine_run_saves_provisional_full_stack_json_and_all_plots(tmp_path):
    result = KerascanEngine().analyze(synthetic_placido(rings=8), tmp_path)
    assert result["screening_result"] == "ANALYSIS_BLOCKED"
    assert result["classification_performed"] is False
    assert result["full_stack_analysis"]["ring_count_verified"] is False
    assert (tmp_path / "full_stack_analysis.json").exists()
    required_plots = {
        "full_stack_tracked_rings.png",
        "inter_ring_spacing_matrix.png",
        "normalized_inter_ring_spacing_matrix.png",
        "angular_variation_by_ring_pair.png",
        "radial_stack_deviation_by_meridian.png",
        "neighbouring_ring_coherence.png",
        "cumulative_radial_residual.png",
        "ring_and_pair_completeness.png",
        "inner_middle_outer_comparison.png",
        "full_stack_sector_map.png",
    }
    assert required_plots <= {path.name for path in tmp_path.iterdir()}


@pytest.mark.parametrize("filename", ["aleft.png", "aright.png"])
def test_original_real_samples_report_full_stack_engineering_data_without_a_conclusion(tmp_path, filename):
    source = Path(__file__).resolve().parents[2] / "sample_images" / filename
    output = tmp_path / filename.removesuffix(".png")
    result = KerascanEngine().analyze(source, output)
    stack = result["full_stack_analysis"]
    assert result["screening_result"] == "ANALYSIS_BLOCKED"
    assert result["geometry_status"] == "ANALYSIS_BLOCKED"
    assert result["classification_performed"] is False
    assert result["gates"]["verified_hardware_ring_count"] == "BLOCKED"
    assert result["gates"]["approved_geometry_thresholds"] == "MISSING"
    assert stack["expected_ring_count"] is None
    assert stack["ring_count_verified"] is False
    assert stack["complete_stack_available"] is False
    assert stack["analysed_ring_pair_count"] == len(stack["analysed_ring_indices"]) - 1
    assert len(stack["ring_completeness"]) == len(stack["analysed_ring_indices"])
    assert len(stack["ring_pair_completeness"]) == stack["analysed_ring_pair_count"]
    assert len(stack["angular_spacing_variation_by_pair"]) == stack["analysed_ring_pair_count"]
    assert len(stack["radial_stack_deviation_by_meridian"]) == 240
    assert (output / "full_stack_analysis.json").exists()
    assert (output / "full_stack_sector_map.png").exists()
