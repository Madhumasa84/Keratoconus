"""Explainable mathematical Placido-ring geometry assessment.

The complete tracked ring stack is evaluated as image-space geometry. No
learned image model, physical topography, clinical probability, or diagnosis is
implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .full_stack import analyse_full_stack
from .reference_geometry import analyse_reference_geometry


def _thresholds(config: Any) -> Any:
    if hasattr(config, "geometry"):
        return getattr(config.geometry, "thresholds", None)
    return getattr(config, "thresholds", None)


def _geometry_config(config: Any) -> Any:
    return getattr(config, "geometry", config)


def _ring_shape_irregularity(radii: np.ndarray, angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-ring robust Fourier residuals as an image-space proxy."""
    theta = np.deg2rad(angles_deg)
    design = np.column_stack(
        [np.ones_like(theta), np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta)]
    )
    residuals = np.full_like(radii, np.nan, dtype=float)
    irregularity = np.full(radii.shape[0], np.nan, dtype=float)
    for ring, values in enumerate(radii):
        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 6:
            continue
        beta, *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
        preliminary = values[valid] - design[valid] @ beta
        median = np.median(preliminary)
        mad = np.median(np.abs(preliminary - median))
        inliers = np.abs(preliminary - median) <= max(4.0 * mad, 1.0)
        if np.count_nonzero(inliers) >= 6:
            beta, *_ = np.linalg.lstsq(design[valid][inliers], values[valid][inliers], rcond=None)
        ring_residual = values - design @ beta
        residuals[ring] = ring_residual
        scale = max(float(np.nanmean(values[valid])), 1e-6)
        irregularity[ring] = float(np.sqrt(np.nanmean(ring_residual[valid] ** 2)) / scale)
    return irregularity, residuals


def compute_geometry(
    radii: np.ndarray,
    angles_deg: np.ndarray,
    observed: np.ndarray | None,
    min_direct_coverage: float,
    config: Any,
    expected_ring_count: int | None = None,
    *,
    ring_count_source: str | None = None,
    interpolated: np.ndarray | None = None,
    rejected: np.ndarray | None = None,
    centre: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Evaluate all adjacent ring pairs over all sampled meridians.

    ``baseline_spacing[k]`` is based only on directly observed radii for pair
    ``k``. Missing data stays missing, and non-positive spacing is never made
    positive by absolute value or clipping.
    """
    radii = np.asarray(radii, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    full = analyse_full_stack(
        radii,
        angles_deg,
        observed,
        interpolated=interpolated,
        rejected=rejected,
        expected_ring_count=expected_ring_count,
        min_direct_coverage=min_direct_coverage,
        config=_geometry_config(config),
    )
    stack = full["full_stack_analysis"]
    grids = full["_grids"]
    reference_output = analyse_reference_geometry(
        radii,
        angles_deg,
        observed,
        interpolated=interpolated,
        rejected=rejected,
        centre=centre,
        expected_ring_count=expected_ring_count,
        min_direct_coverage=min_direct_coverage,
        config=_geometry_config(config),
    )
    reference_geometry = reference_output["reference_geometry"]
    spacing = grids["spacing_matrix"]
    normalized = grids["normalized_spacing"]
    angular_variation = grids["angular_variation"]

    half = spacing.shape[1] // 2 if spacing.ndim == 2 else 0
    opposite_grid = np.abs(spacing - np.roll(spacing, half, axis=1)) if half else np.full_like(spacing, np.nan)
    baseline = np.asarray(
        [np.nan if value is None else value for value in stack["baseline_spacing_by_pair"]], dtype=float
    )
    opposite_grid = np.divide(
        opposite_grid,
        baseline[:, None],
        out=np.full_like(opposite_grid, np.nan),
        where=np.isfinite(baseline[:, None]) & (baseline[:, None] > 0.0),
    )
    compression_grid = np.maximum(0.0, 1.0 - normalized)
    expansion_grid = np.maximum(0.0, normalized - 1.0)
    irregularity, shape_residuals = _ring_shape_irregularity(radii, angles_deg)

    pair_count = max(radii.shape[0] - 1, 0)
    longest_compressed = grids["longest_compressed_pair_run"]
    longest_expanded = grids["longest_expanded_pair_run"]
    magnitude = np.zeros(len(angles_deg), dtype=float)
    combined_magnitude = np.maximum(compression_grid, expansion_grid)
    for angle in range(len(angles_deg)):
        column = combined_magnitude[:, angle]
        if np.any(np.isfinite(column)):
            magnitude[angle] = float(np.nanmax(column))
    coherence_run = np.maximum(longest_compressed, longest_expanded)
    multiring_agreement_grid = coherence_run.astype(float) / max(pair_count, 1) * np.nan_to_num(magnitude, nan=0.0)

    expected_points = max((expected_ring_count or radii.shape[0]) * max(len(angles_deg), 1), 1)
    if observed is None:
        direct_count = int(np.count_nonzero(np.isfinite(radii)))
    else:
        observed_array = np.asarray(observed, dtype=bool)
        direct_count = int(np.count_nonzero(observed_array & np.isfinite(radii)))
    direct_coverage = float(direct_count / expected_points)

    def finite_max(values: np.ndarray, default: float = 0.0) -> float:
        values = np.asarray(values, dtype=float)
        return float(np.nanmax(values)) if np.any(np.isfinite(values)) else default

    legacy_features = {
        "SPACING_VARIATION": finite_max(angular_variation),
        "OPPOSITE_ASYMMETRY": finite_max(opposite_grid),
        "LOCAL_COMPRESSION": finite_max(compression_grid),
        "LOCAL_EXPANSION": finite_max(expansion_grid),
        "RING_SHAPE_IRREGULARITY": finite_max(irregularity),
        "MULTIRING_AGREEMENT": finite_max(multiring_agreement_grid),
        "DIRECT_OBSERVATION_COVERAGE": direct_coverage,
        "TRACKING_RELIABILITY": direct_coverage,
    }

    reason_codes: list[str] = []
    thresholds = _thresholds(config)
    invalid_reasons = list(full["invalid_reason_codes"])
    if radii.ndim != 2 or radii.shape[0] < 2:
        invalid_reasons.append("insufficient_rings")

    classification_performed = False
    if invalid_reasons:
        geometry_status = "UNGRADABLE"
        reason_codes = sorted(set(invalid_reasons))
    elif expected_ring_count is None:
        geometry_status = "ANALYSIS_BLOCKED"
        reason_codes = ["missing_verified_ring_count"]
    elif not stack["complete_stack_available"]:
        geometry_status = "UNGRADABLE"
        reason_codes = list(stack["completeness_reason_codes"] or ["incomplete_full_stack"])
    elif thresholds is None:
        geometry_status = "NOT_CALIBRATED"
        reason_codes = ["missing_clinical_thresholds"]
    else:
        suspicious = False
        indeterminate = False
        uncorroborated_pair_maximum = False
        uncorroborated_only_reasons: list[str] = []
        pair_maximum_features = {
            "SPACING_VARIATION",
            "OPPOSITE_ASYMMETRY",
            "LOCAL_COMPRESSION",
            "LOCAL_EXPANSION",
        }
        minimum_run = max(2, int(getattr(_geometry_config(config), "min_coherent_pair_run", 2)))
        summary = stack["full_stack_summary"]
        multiring_supported = max(
            int(summary["maximum_compressed_pair_run"]),
            int(summary["maximum_expanded_pair_run"]),
        ) >= minimum_run
        # Requiring a coherent multi-pair run only discriminates when the stack
        # has appreciably more pairs than that run. On a short stack every pair
        # is significant and the test cannot be satisfied at all, so applying it
        # there would silently downgrade a real borderline finding to normal --
        # the one direction a screening aid must never fail in.
        corroboration_discriminates = (radii.shape[0] - 1) > minimum_run
        for feature_name, bound in getattr(thresholds, "suspicious_bounds", {}).items():
            if legacy_features.get(feature_name, float("-inf")) >= bound:
                if feature_name in pair_maximum_features and not multiring_supported:
                    uncorroborated_pair_maximum = True
                else:
                    suspicious = True
                    reason_codes.append(f"suspicious_{feature_name.lower()}")
        if not suspicious:
            for feature_name, bound in getattr(thresholds, "indeterminate_bounds", {}).items():
                if legacy_features.get(feature_name, float("-inf")) >= bound:
                    # Pair-maximum features are set by the single worst adjacent
                    # ring pair. Coverage falls off at both ends of the mire
                    # pattern -- the innermost rings subtend few pixels and the
                    # outermost fade into the limbus -- so one patchy edge pair
                    # can raise the maximum on an otherwise regular eye. Ectasia
                    # steepens a coherent run of neighbouring pairs, so the same
                    # corroboration already required for a suspicious call is
                    # applied here rather than reporting a borderline result on
                    # one isolated pair.
                    if (
                        feature_name in pair_maximum_features
                        and corroboration_discriminates
                        and not multiring_supported
                    ):
                        # Recorded for transparency but deliberately does not
                        # escalate: unlike an uncorroborated suspicious-level
                        # value, which is downgraded to borderline below, a
                        # borderline-level value on one isolated pair is not
                        # evidence of anything.
                        uncorroborated_only_reasons.append(
                            f"uncorroborated_{feature_name.lower()}"
                        )
                        continue
                    indeterminate = True
                    reason_codes.append(f"indeterminate_{feature_name.lower()}")
        if uncorroborated_pair_maximum and not suspicious:
            indeterminate = True
            reason_codes.append("uncorroborated_single_pair_maximum")
        if not suspicious and not indeterminate:
            # The capture was regular once isolated single-pair maxima were
            # discounted; keep why, so the record shows what was set aside.
            reason_codes.extend(uncorroborated_only_reasons)
        geometry_status = "SUSPICIOUS" if suspicious else "INDETERMINATE" if indeterminate else "NORMAL-LIKE"
        classification_performed = True

    gates = {
        "verified_hardware_ring_count": "PASS" if expected_ring_count is not None else "BLOCKED",
        "complete_full_stack_geometry": "PASS" if stack["complete_stack_available"] else "FAIL",
        "approved_geometry_thresholds": "PASS" if thresholds is not None else "MISSING",
    }
    grid_output = {
        **grids,
        **{f"reference_{key}": value for key, value in reference_output["_grids"].items()},
        "opp_asym_grid": opposite_grid,
        "compression_grid": compression_grid,
        "expansion_grid": expansion_grid,
        "multiring_agreement_grid": multiring_agreement_grid,
        "residuals": shape_residuals,
        "robust_var": angular_variation,
    }
    return {
        "geometry_method": "full_stack_inter_ring_spacing_regularity",
        "geometry_method_version": "2.1-reference-comparison",
        "learned_image_model_used": False,
        "classification_performed": classification_performed,
        "units": "pixels and dimensionless image-space proxy",
        "physical_calibration_status": "NOT_CALIBRATED",
        "hardware_configuration_version": getattr(config, "hardware_version", "unknown"),
        "expected_ring_count": expected_ring_count,
        "ring_count_source": ring_count_source
        or ("verified_device_config" if expected_ring_count is not None else "provisional_polar_profile"),
        "reference_profile_version": "self-fitted-concentric-and-low-order-v1",
        "threshold_configuration_version": getattr(thresholds, "version", "none") if thresholds is not None else "none",
        "feature_validity": full["valid_geometry"],
        "limitations": [
            "experimental—not clinically calibrated",
            "image-space geometry only; no physical corneal calibration",
            "equal normalized spacing is a mathematical ring-pair reference, not proof of a normal cornea",
            "spacing irregularity does not prove keratoconus",
            "self-fitted reference is not a validated normal template and may absorb diffuse deformation",
        ],
        "geometry_status": geometry_status,
        "geometry_confidence": direct_coverage if full["valid_geometry"] else 0.0,
        "gates": gates,
        "reason_codes": reason_codes,
        "features": legacy_features,
        "explainable_features": full["explainable_features"],
        "full_stack_analysis": stack,
        "reference_geometry": reference_geometry,
        "_grids": grid_output,
    }


@dataclass(frozen=True)
class GeometryValidation:
    valid: bool
    flags: list[str]
    direct_coverage: float


def validate_geometry(
    radii: np.ndarray,
    observed: np.ndarray | None = None,
    min_direct_coverage: float = 0.0,
) -> GeometryValidation:
    """Verify basic ordering before full-stack feature extraction."""
    radii = np.asarray(radii, dtype=float)
    flags: list[str] = []
    if radii.ndim != 2 or radii.shape[0] < 2 or radii.shape[1] < 1:
        return GeometryValidation(False, ["insufficient_tracked_geometry"], 0.0)
    finite = np.isfinite(radii)
    if np.any(radii[finite] <= 0):
        flags.append("non_positive_radius")
    spacing = np.diff(radii, axis=0)
    if np.any(np.isfinite(spacing) & (spacing <= 0.0)):
        flags.append("non_monotonic_ring_order")
    for angle in range(radii.shape[1]):
        ordered_values = radii[:, angle][finite[:, angle]]
        if len(ordered_values) > 1 and np.any(np.diff(ordered_values) <= 0.0):
            flags.append("non_monotonic_ring_order")
            break
    if observed is None:
        direct = float(np.mean(finite))
    else:
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != radii.shape:
            flags.append("observation_shape_mismatch")
            direct = 0.0
        else:
            direct = float(np.mean(observed & finite))
    if direct < min_direct_coverage:
        flags.append("insufficient_direct_observation")
    if not np.any(np.isfinite(spacing)):
        flags.append("no_valid_ring_spacing")
    return GeometryValidation(not flags, sorted(set(flags)), direct)
