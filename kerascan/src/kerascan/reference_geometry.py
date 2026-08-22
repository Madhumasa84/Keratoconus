"""Self-fitted engineering references for tracked Placido-ring geometry.

References in this module are fitted from the same eye.  They are image-space
comparison baselines, not validated normal templates, disease models, or
clinical classifiers.
"""
from __future__ import annotations

from typing import Any

import numpy as np


STATE_NAMES = np.asarray(("OBSERVED", "INTERPOLATED", "MISSING", "REJECTED"), dtype=object)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _serial_vector(values: np.ndarray) -> list[float | None]:
    return [_finite(value) for value in np.asarray(values).reshape(-1)]


def _serial_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [[_finite(value) for value in row] for row in np.asarray(values)]


def _design(theta: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.ones_like(theta),
            np.cos(theta),
            np.sin(theta),
            np.cos(2.0 * theta),
            np.sin(2.0 * theta),
        )
    )


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def _robust_low_order_fit(
    values: np.ndarray,
    theta: np.ndarray,
    direct: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int, int, float | None]:
    """Fit the five-term angular reference while rejecting isolated spikes only.

    Contiguous residual support is retained so a persistent sector is not
    discarded as a collection of independent outliers.  Huber weighting limits
    leverage without converting any retained observation into missing data.
    """
    direct_ids = np.flatnonzero(direct & np.isfinite(values))
    predicted = np.full_like(values, np.nan, dtype=float)
    residual = np.full_like(values, np.nan, dtype=float)
    if direct_ids.size < 6:
        return None, predicted, residual, int(direct_ids.size), 0, None

    design = _design(theta)
    fit_design = design[direct_ids]
    fit_values = values[direct_ids]
    coefficients, *_ = np.linalg.lstsq(fit_design, fit_values, rcond=None)
    preliminary = fit_values - fit_design @ coefficients
    preliminary_median = float(np.median(preliminary))
    threshold = max(4.0 * _robust_scale(preliminary), 1.0)
    candidates = np.abs(preliminary - preliminary_median) > threshold

    # Only an angularly isolated spike is removed. Persistent neighbouring
    # points remain available to the robust fit and remain visible as residuals.
    candidate_by_angle = np.zeros(len(values), dtype=bool)
    candidate_by_angle[direct_ids] = candidates
    isolated = np.zeros(len(values), dtype=bool)
    for angle in direct_ids[candidates]:
        previous = (angle - 1) % len(values)
        following = (angle + 1) % len(values)
        if not candidate_by_angle[previous] and not candidate_by_angle[following]:
            isolated[angle] = True

    used = direct & np.isfinite(values) & ~isolated
    used_ids = np.flatnonzero(used)
    if used_ids.size < 6:
        return None, predicted, residual, int(used_ids.size), int(np.count_nonzero(isolated)), None

    fit_design = design[used_ids]
    fit_values = values[used_ids]
    coefficients, *_ = np.linalg.lstsq(fit_design, fit_values, rcond=None)
    for _ in range(6):
        fit_residual = fit_values - fit_design @ coefficients
        centre = float(np.median(fit_residual))
        scale = _robust_scale(fit_residual)
        if scale <= 1e-9:
            break
        huber_limit = 2.5 * scale
        distance = np.abs(fit_residual - centre)
        weights = np.ones_like(distance)
        large = distance > huber_limit
        weights[large] = huber_limit / np.maximum(distance[large], 1e-12)
        root_weight = np.sqrt(weights)
        updated, *_ = np.linalg.lstsq(
            fit_design * root_weight[:, None],
            fit_values * root_weight,
            rcond=None,
        )
        if np.allclose(updated, coefficients, rtol=1e-9, atol=1e-9):
            coefficients = updated
            break
        coefficients = updated

    predicted = design @ coefficients
    residual[direct] = values[direct] - predicted[direct]
    used_residual = values[used] - predicted[used]
    rms = float(np.sqrt(np.mean(used_residual**2))) if used_residual.size else None
    return (
        coefficients,
        predicted,
        residual,
        int(used_ids.size),
        int(np.count_nonzero(isolated)),
        rms,
    )


def _state_matrices(
    radii: np.ndarray,
    observed: np.ndarray | None,
    interpolated: np.ndarray | None,
    rejected: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    finite = np.isfinite(radii)
    observed_input = finite.copy() if observed is None else np.asarray(observed, dtype=bool)
    interpolated_input = np.zeros_like(finite) if interpolated is None else np.asarray(interpolated, dtype=bool)
    rejected_input = np.zeros_like(finite) if rejected is None else np.asarray(rejected, dtype=bool)
    for name, state in (
        ("observed", observed_input),
        ("interpolated", interpolated_input),
        ("rejected", rejected_input),
    ):
        if state.shape != radii.shape:
            raise ValueError(f"{name} state shape must match radii")

    reasons: list[str] = []
    if np.any(observed_input & interpolated_input) or np.any(observed_input & rejected_input) or np.any(interpolated_input & rejected_input):
        reasons.append("overlapping_observation_states")
    if np.any((observed_input | interpolated_input) & ~finite):
        reasons.append("finite_radius_required_for_observed_or_interpolated_state")

    observed_mask = observed_input & finite & ~rejected_input
    interpolated_mask = interpolated_input & finite & ~observed_mask & ~rejected_input
    rejected_mask = rejected_input & ~observed_mask & ~interpolated_mask
    missing_mask = ~(observed_mask | interpolated_mask | rejected_mask)
    states = np.full(radii.shape, 2, dtype=np.uint8)
    states[observed_mask] = 0
    states[interpolated_mask] = 1
    states[rejected_mask] = 3
    return observed_mask, interpolated_mask, missing_mask, rejected_mask, states, reasons


def _pair_states(
    observed: np.ndarray,
    interpolated: np.ndarray,
    rejected: np.ndarray,
    spacing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.isfinite(spacing) & (spacing > 0.0)
    pair_observed = observed[:-1] & observed[1:] & positive
    pair_rejected = rejected[:-1] | rejected[1:] | (np.isfinite(spacing) & ~positive)
    pair_interpolated = (
        ~pair_observed
        & ~pair_rejected
        & positive
        & (observed[:-1] | interpolated[:-1])
        & (observed[1:] | interpolated[1:])
    )
    states = np.full(spacing.shape, 2, dtype=np.uint8)
    states[pair_observed] = 0
    states[pair_interpolated] = 1
    states[pair_rejected] = 3
    return pair_observed, states


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _coherent_cells(significant: np.ndarray, minimum_run: int) -> tuple[np.ndarray, np.ndarray]:
    significant = np.asarray(significant, dtype=bool)
    coherent = np.zeros_like(significant)
    longest = np.zeros(significant.shape[1], dtype=int)
    for angle in range(significant.shape[1]):
        row = significant[:, angle]
        start = 0
        while start < len(row):
            if not row[start]:
                start += 1
                continue
            stop = start + 1
            while stop < len(row) and row[stop]:
                stop += 1
            run = stop - start
            longest[angle] = max(longest[angle], run)
            if run >= minimum_run:
                coherent[start:stop, angle] = True
            start = stop
    return coherent, longest


def _circular_runs(mask: np.ndarray, minimum_samples: int) -> list[tuple[np.ndarray, bool]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.size or not np.any(mask):
        return []
    if np.all(mask):
        return [(np.arange(mask.size), False)] if mask.size >= minimum_samples else []
    runs: list[np.ndarray] = []
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(mask) and mask[stop]:
            stop += 1
        runs.append(np.arange(start, stop))
        start = stop
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(mask) - 1:
        runs = [np.concatenate((runs[-1], runs[0]))] + runs[1:-1]
        seam = True
    else:
        seam = False
    return [
        (run, bool(seam and index == 0))
        for index, run in enumerate(runs)
        if len(run) >= minimum_samples
    ]


def _coherent_sector_records(
    normalized_deviation: np.ndarray,
    direct: np.ndarray,
    angles_deg: np.ndarray,
    coherent: np.ndarray,
    direction: str,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    if len(angles_deg) > 1:
        wrapped = np.sort(np.mod(angles_deg, 360.0))
        angular_step = float(np.median(np.diff(np.r_[wrapped, wrapped[0] + 360.0])))
    else:
        angular_step = 360.0
    records: list[dict[str, Any]] = []
    for indices, seam in _circular_runs(np.any(coherent, axis=0), minimum_samples):
        cells = coherent[:, indices]
        affected = np.flatnonzero(np.any(cells, axis=1))
        values = normalized_deviation[:, indices][cells]
        values = values[np.isfinite(values)]
        if not affected.size or not values.size:
            continue
        support = direct[np.ix_(affected, indices)]
        records.append(
            {
                "direction": direction,
                "start_degrees": float(angles_deg[indices[0]] % 360.0),
                "end_degrees": float(angles_deg[indices[-1]] % 360.0),
                "angular_width_degrees": float(min(360.0, len(indices) * angular_step)),
                "first_affected_ring": int(affected[0]),
                "last_affected_ring": int(affected[-1]),
                "affected_ring_count": int(len(affected)),
                "affected_stack_fraction": float(len(affected) / max(normalized_deviation.shape[0], 1)),
                "median_normalized_deviation": float(np.median(values)),
                "maximum_absolute_normalized_deviation": float(np.max(np.abs(values))),
                "direct_observation_fraction": float(np.mean(support)),
                "crosses_zero_degree_seam": seam,
            }
        )
    return records


def _deviation_summary(
    values: np.ndarray,
    normalized: np.ndarray,
    coverage: np.ndarray,
    valid_by_ring: np.ndarray,
) -> dict[str, Any]:
    by_ring: list[dict[str, Any]] = []
    for ring, row in enumerate(values):
        finite = row[np.isfinite(row)]
        normalized_finite = normalized[ring, np.isfinite(normalized[ring])]
        by_ring.append(
            {
                "ring_index": ring,
                "valid": bool(valid_by_ring[ring] and finite.size),
                "direct_observation_fraction": float(coverage[ring]),
                "signed_median_pixels": _finite(np.median(finite)) if finite.size else None,
                "median_absolute_pixels": _finite(np.median(np.abs(finite))) if finite.size else None,
                "rms_pixels": _finite(np.sqrt(np.mean(finite**2))) if finite.size else None,
                "maximum_absolute_pixels": _finite(np.max(np.abs(finite))) if finite.size else None,
                "median_absolute_normalized": (
                    _finite(np.median(np.abs(normalized_finite))) if normalized_finite.size else None
                ),
                "technical_reliability": "SUPPORTED" if valid_by_ring[ring] and finite.size else "INVALID_REFERENCE",
            }
        )
    all_values = values[np.isfinite(values)]
    all_normalized = normalized[np.isfinite(normalized)]
    return {
        "by_ring": by_ring,
        "full_stack_median_absolute_pixels": _finite(np.median(np.abs(all_values))) if all_values.size else None,
        "full_stack_rms_pixels": _finite(np.sqrt(np.mean(all_values**2))) if all_values.size else None,
        "full_stack_median_absolute_normalized": (
            _finite(np.median(np.abs(all_normalized))) if all_normalized.size else None
        ),
    }


def _spacing_summary(values: np.ndarray, normalized: np.ndarray, pair_observed: np.ndarray) -> dict[str, Any]:
    by_pair: list[dict[str, Any]] = []
    for pair, row in enumerate(values):
        finite = row[np.isfinite(row)]
        normalized_finite = normalized[pair, np.isfinite(normalized[pair])]
        by_pair.append(
            {
                "ring_pair_index": pair,
                "rings": [pair, pair + 1],
                "direct_observation_fraction": float(np.mean(pair_observed[pair])),
                "signed_median_pixels": _finite(np.median(finite)) if finite.size else None,
                "median_absolute_pixels": _finite(np.median(np.abs(finite))) if finite.size else None,
                "rms_pixels": _finite(np.sqrt(np.mean(finite**2))) if finite.size else None,
                "median_absolute_normalized": (
                    _finite(np.median(np.abs(normalized_finite))) if normalized_finite.size else None
                ),
            }
        )
    return {"by_ring_pair": by_pair}


def analyse_reference_geometry(
    radii: np.ndarray,
    angles_deg: np.ndarray,
    observed: np.ndarray | None,
    *,
    interpolated: np.ndarray | None = None,
    rejected: np.ndarray | None = None,
    centre: tuple[float, float] = (0.0, 0.0),
    expected_ring_count: int | None = None,
    min_direct_coverage: float = 0.55,
    config: Any = None,
) -> dict[str, Any]:
    """Construct concentric and smooth references from direct observations."""
    radii = np.asarray(radii, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if radii.ndim != 2 or radii.shape[0] < 1:
        raise ValueError("radii must have shape (ring, meridian)")
    ring_count, meridians = radii.shape
    if angles_deg.shape != (meridians,):
        raise ValueError("angles_deg length must match radii meridians")

    observed_mask, interpolated_mask, _, rejected_mask, ring_states, reasons = _state_matrices(
        radii, observed, interpolated, rejected
    )
    usable = (observed_mask | interpolated_mask) & np.isfinite(radii)
    if np.any(usable & (radii <= 0.0)):
        reasons.append("non_positive_radius")
    for angle in range(meridians):
        ids = np.flatnonzero(usable[:, angle])
        if ids.size > 1 and np.any(np.diff(radii[ids, angle]) <= 0.0):
            reasons.append("non_positive_observed_spacing")
            break
    if expected_ring_count is not None and int(expected_ring_count) != ring_count:
        reasons.append("expected_ring_count_mismatch")

    direct_counts = np.sum(observed_mask, axis=1)
    coverage = direct_counts / max(meridians, 1)
    valid_by_ring = (coverage >= min_direct_coverage) & (direct_counts >= 6)
    reference_radii = np.full(ring_count, np.nan, dtype=float)
    for ring in range(ring_count):
        if valid_by_ring[ring]:
            reference_radii[ring] = float(np.median(radii[ring, observed_mask[ring]]))
    concentric_ordered = bool(
        np.all(valid_by_ring)
        and np.all(np.isfinite(reference_radii))
        and (ring_count < 2 or np.all(np.diff(reference_radii) > 0.0))
    )
    if np.all(valid_by_ring) and not concentric_ordered:
        reasons.append("concentric_reference_ordering_failure")
    if not np.all(valid_by_ring):
        reasons.append("insufficient_reference_ring_coverage")
    circle_reference = np.repeat(reference_radii[:, None], meridians, axis=1)

    theta = np.deg2rad(angles_deg)
    smooth_reference = np.full_like(radii, np.nan)
    coefficients: list[list[float] | None] = []
    fit_valid_by_ring = np.zeros(ring_count, dtype=bool)
    fit_points: list[int] = []
    rejected_outliers: list[int] = []
    fit_rms: list[float | None] = []
    for ring in range(ring_count):
        if not valid_by_ring[ring]:
            coefficients.append(None)
            fit_points.append(int(direct_counts[ring]))
            rejected_outliers.append(0)
            fit_rms.append(None)
            continue
        beta, predicted, _, points, outliers, rms = _robust_low_order_fit(
            radii[ring], theta, observed_mask[ring]
        )
        if beta is None or np.any(predicted <= 0.0):
            coefficients.append(None)
            fit_points.append(points)
            rejected_outliers.append(outliers)
            fit_rms.append(rms)
            continue
        coefficients.append([float(value) for value in beta])
        smooth_reference[ring] = predicted
        fit_valid_by_ring[ring] = True
        fit_points.append(points)
        rejected_outliers.append(outliers)
        fit_rms.append(rms)

    smooth_ordered = bool(
        np.all(fit_valid_by_ring)
        and (ring_count < 2 or np.all(np.diff(smooth_reference, axis=0) > 0.0))
    )
    if np.all(fit_valid_by_ring) and not smooth_ordered:
        reasons.append("smooth_reference_crossing")
    if not np.all(fit_valid_by_ring):
        reasons.append("invalid_smooth_reference_ring")

    observed_spacing = np.diff(radii, axis=0) if ring_count > 1 else np.empty((0, meridians))
    pair_observed, pair_states = _pair_states(
        observed_mask, interpolated_mask, rejected_mask, observed_spacing
    ) if ring_count > 1 else (np.empty((0, meridians), dtype=bool), np.empty((0, meridians), dtype=np.uint8))
    direct_spacing = observed_spacing.copy()
    direct_spacing[~pair_observed | (direct_spacing <= 0.0)] = np.nan
    pair_baseline = np.full(max(ring_count - 1, 0), np.nan)
    for pair in range(len(pair_baseline)):
        values = direct_spacing[pair, np.isfinite(direct_spacing[pair])]
        if values.size:
            pair_baseline[pair] = float(np.median(values))
    ring_denominator = np.full(ring_count, np.nan)
    if ring_count > 1:
        ring_denominator[:-1] = pair_baseline
        # The outermost ring has no outward neighbour; use its directly
        # observed inward adjacent-pair median explicitly.
        ring_denominator[-1] = pair_baseline[-1]

    circle_deviation = np.full_like(radii, np.nan)
    smooth_deviation = np.full_like(radii, np.nan)
    for ring in range(ring_count):
        if valid_by_ring[ring] and np.isfinite(reference_radii[ring]):
            circle_deviation[ring, observed_mask[ring]] = (
                radii[ring, observed_mask[ring]] - reference_radii[ring]
            )
        if fit_valid_by_ring[ring]:
            smooth_deviation[ring, observed_mask[ring]] = (
                radii[ring, observed_mask[ring]] - smooth_reference[ring, observed_mask[ring]]
            )
    normalized_circle = np.divide(
        circle_deviation,
        ring_denominator[:, None],
        out=np.full_like(circle_deviation, np.nan),
        where=np.isfinite(ring_denominator[:, None]) & (ring_denominator[:, None] > 0.0),
    )
    normalized_smooth = np.divide(
        smooth_deviation,
        ring_denominator[:, None],
        out=np.full_like(smooth_deviation, np.nan),
        where=np.isfinite(ring_denominator[:, None]) & (ring_denominator[:, None] > 0.0),
    )

    circle_reference_spacing = np.diff(reference_radii) if ring_count > 1 else np.empty(0)
    smooth_reference_spacing = np.diff(smooth_reference, axis=0) if ring_count > 1 else np.empty((0, meridians))
    circle_spacing_residual = np.full_like(observed_spacing, np.nan)
    smooth_spacing_residual = np.full_like(observed_spacing, np.nan)
    for pair in range(ring_count - 1):
        if np.isfinite(circle_reference_spacing[pair]) and circle_reference_spacing[pair] > 0.0:
            circle_spacing_residual[pair, pair_observed[pair]] = (
                observed_spacing[pair, pair_observed[pair]] - circle_reference_spacing[pair]
            )
        valid_smooth = pair_observed[pair] & np.isfinite(smooth_reference_spacing[pair]) & (smooth_reference_spacing[pair] > 0.0)
        smooth_spacing_residual[pair, valid_smooth] = (
            observed_spacing[pair, valid_smooth] - smooth_reference_spacing[pair, valid_smooth]
        )
    normalized_circle_spacing = np.divide(
        circle_spacing_residual,
        circle_reference_spacing[:, None],
        out=np.full_like(circle_spacing_residual, np.nan),
        where=np.isfinite(circle_reference_spacing[:, None]) & (circle_reference_spacing[:, None] > 0.0),
    ) if ring_count > 1 else circle_spacing_residual.copy()
    normalized_smooth_spacing = np.divide(
        smooth_spacing_residual,
        smooth_reference_spacing,
        out=np.full_like(smooth_spacing_residual, np.nan),
        where=np.isfinite(smooth_reference_spacing) & (smooth_reference_spacing > 0.0),
    ) if ring_count > 1 else smooth_spacing_residual.copy()

    magnitude = float(getattr(config, "reference_deviation_magnitude_fraction", 0.08))
    minimum_ring_run = max(2, int(getattr(config, "min_coherent_ring_run", 2)))
    minimum_sector_samples = max(2, int(getattr(config, "min_reference_sector_angular_samples", 3)))
    significant_inward = observed_mask & np.isfinite(normalized_smooth) & (normalized_smooth <= -magnitude)
    significant_outward = observed_mask & np.isfinite(normalized_smooth) & (normalized_smooth >= magnitude)
    coherent_inward, longest_inward = _coherent_cells(significant_inward, minimum_ring_run)
    coherent_outward, longest_outward = _coherent_cells(significant_outward, minimum_ring_run)
    coherent_sectors = _coherent_sector_records(
        normalized_smooth, observed_mask, angles_deg, coherent_inward, "INWARD", minimum_sector_samples
    ) + _coherent_sector_records(
        normalized_smooth, observed_mask, angles_deg, coherent_outward, "OUTWARD", minimum_sector_samples
    )
    coherent_sectors.sort(key=lambda item: (item["start_degrees"], item["direction"]))

    reliability = np.full(radii.shape, "MISSING_NOT_EVALUATED", dtype=object)
    reliability[interpolated_mask] = "INTERPOLATED_NOT_USED"
    reliability[rejected_mask] = "REJECTED_NOT_USED"
    reliability[observed_mask] = "REFERENCE_INVALID"
    for ring in range(ring_count):
        if valid_by_ring[ring]:
            reliability[ring, observed_mask[ring]] = "DIRECT_OBSERVED_REFERENCE_VALID"

    state_labels = STATE_NAMES[ring_states].tolist()
    ring_indices = list(range(ring_count))
    angle_values = [float(value) for value in angles_deg]
    circle_summary = _deviation_summary(circle_deviation, normalized_circle, coverage, valid_by_ring)
    smooth_summary = _deviation_summary(smooth_deviation, normalized_smooth, coverage, fit_valid_by_ring)
    spacing_summary = {
        "concentric_reference": _spacing_summary(
            circle_spacing_residual, normalized_circle_spacing, pair_observed
        ),
        "smooth_reference": _spacing_summary(
            smooth_spacing_residual, normalized_smooth_spacing, pair_observed
        ),
    }
    reasons = sorted(set(reasons))
    concentric_valid = bool(concentric_ordered and not any(reason in reasons for reason in (
        "non_positive_radius", "non_positive_observed_spacing", "overlapping_observation_states"
    )))
    smooth_valid = bool(smooth_ordered and not any(reason in reasons for reason in (
        "non_positive_radius", "non_positive_observed_spacing", "overlapping_observation_states"
    )))
    reference = {
        "reference_type": "SELF_FITTED_ENGINEERING_REFERENCE",
        "validated_normal_reference": False,
        "classification_performed": False,
        "valid": bool(concentric_valid and smooth_valid),
        "invalid_reason_codes": reasons,
        "centre": {"x": float(centre[0]), "y": float(centre[1])},
        "expected_ring_count": int(expected_ring_count) if expected_ring_count is not None else None,
        "detected_ring_count": int(np.count_nonzero(np.any(np.isfinite(radii), axis=1))),
        "ring_count_verified": expected_ring_count is not None,
        "analysed_ring_indices": ring_indices,
        "minimum_direct_coverage_required": float(min_direct_coverage),
        "normalization_denominator_by_ring_pixels": _serial_vector(ring_denominator),
        "outer_ring_normalization": "directly observed median spacing of the outermost inward adjacent ring pair",
        "concentric_reference": {
            "valid": concentric_valid,
            "radii_pixels": _serial_vector(reference_radii),
            "coverage_by_ring": [float(value) for value in coverage],
            "valid_by_ring": [bool(value) for value in valid_by_ring],
            "points_used_by_ring": [int(value) for value in direct_counts],
            "ordering_verified": concentric_ordered,
            "equal_pitch_assumed": False,
        },
        "smooth_reference": {
            "valid": smooth_valid,
            "coefficients_by_ring": coefficients,
            "coverage_by_ring": [float(value) for value in coverage],
            "valid_by_ring": [bool(value) for value in fit_valid_by_ring],
            "fit_residual_by_ring": fit_rms,
            "number_of_points_used_by_ring": fit_points,
            "number_of_rejected_outliers_by_ring": rejected_outliers,
            "ordering_verified_at_every_meridian": smooth_ordered,
            "basis": "a0 + a1*cos(theta) + b1*sin(theta) + a2*cos(2theta) + b2*sin(2theta)",
        },
        "circle_deviation": {
            "ring_indices": ring_indices,
            "meridian_angles_degrees": angle_values,
            "signed_pixels": _serial_matrix(circle_deviation),
            "absolute_pixels": _serial_matrix(np.abs(circle_deviation)),
            "normalized": _serial_matrix(normalized_circle),
            "observation_state": state_labels,
            "technical_reliability": reliability.tolist(),
        },
        "smooth_deviation": {
            "ring_indices": ring_indices,
            "meridian_angles_degrees": angle_values,
            "signed_pixels": _serial_matrix(smooth_deviation),
            "absolute_pixels": _serial_matrix(np.abs(smooth_deviation)),
            "normalized": _serial_matrix(normalized_smooth),
            "observation_state": state_labels,
            "technical_reliability": reliability.tolist(),
        },
        "spacing_residuals": {
            "ring_pair_indices": list(range(max(ring_count - 1, 0))),
            "meridian_angles_degrees": angle_values,
            "observation_state": STATE_NAMES[pair_states].tolist(),
            "observed_spacing_pixels": _serial_matrix(observed_spacing),
            "circle_reference_spacing_pixels": _serial_vector(circle_reference_spacing),
            "smooth_reference_spacing_pixels": _serial_matrix(smooth_reference_spacing),
            "circle_signed_residual_pixels": _serial_matrix(circle_spacing_residual),
            "smooth_signed_residual_pixels": _serial_matrix(smooth_spacing_residual),
            "circle_normalized_residual": _serial_matrix(normalized_circle_spacing),
            "smooth_normalized_residual": _serial_matrix(normalized_smooth_spacing),
        },
        "circle_deviation_summary": circle_summary,
        "smooth_deviation_summary": smooth_summary,
        "spacing_residual_summary": spacing_summary,
        "cross_ring_coherence_by_meridian": {
            "angles_degrees": angle_values,
            "inward_ring_count": [int(value) for value in np.sum(significant_inward, axis=0)],
            "outward_ring_count": [int(value) for value in np.sum(significant_outward, axis=0)],
            "longest_inward_ring_run": [int(value) for value in longest_inward],
            "longest_outward_ring_run": [int(value) for value in longest_outward],
            "inward_stack_fraction": _serial_vector(np.mean(significant_inward, axis=0)),
            "outward_stack_fraction": _serial_vector(np.mean(significant_outward, axis=0)),
            "direct_observation_fraction": _serial_vector(np.mean(observed_mask, axis=0)),
            "engineering_magnitude_fraction": magnitude,
            "minimum_neighbouring_ring_run": minimum_ring_run,
            "minimum_angular_samples": minimum_sector_samples,
        },
        "cross_ring_coherent_sectors": coherent_sectors,
        "limitations": [
            "Reference fitted from the same eye",
            "Not a validated normal template",
            "Image-space measurements only",
            "Diffuse or global deformation may be absorbed by a self-fitted reference",
        ],
        "units": "pixels and dimensionless image-space proxy",
    }
    return {
        "reference_geometry": reference,
        "valid_reference_geometry": reference["valid"],
        "invalid_reason_codes": reasons,
        "_grids": {
            "circle_reference": circle_reference,
            "smooth_reference": smooth_reference,
            "circle_deviation": circle_deviation,
            "smooth_deviation": smooth_deviation,
            "normalized_circle_deviation": normalized_circle,
            "normalized_smooth_deviation": normalized_smooth,
            "observed_spacing": observed_spacing,
            "circle_spacing_residual": circle_spacing_residual,
            "smooth_spacing_residual": smooth_spacing_residual,
            "normalized_circle_spacing_residual": normalized_circle_spacing,
            "normalized_smooth_spacing_residual": normalized_smooth_spacing,
            "ring_states": ring_states,
            "pair_states": pair_states,
            "ring_observed": observed_mask,
            "pair_observed": pair_observed,
            "coherent_inward": coherent_inward,
            "coherent_outward": coherent_outward,
            "valid_concentric_by_ring": valid_by_ring,
            "valid_smooth_by_ring": fit_valid_by_ring,
        },
    }
