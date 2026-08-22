"""Deterministic full-stack Placido inter-ring spacing geometry.

All quantities are image-space proxies.  This module does not contain a
learned model, clinical cut-offs, a disease probability, or a diagnosis.
"""
from __future__ import annotations

from typing import Any

import numpy as np


STATE_NAMES = ("OBSERVED", "INTERPOLATED", "MISSING", "REJECTED")
FRACTION_FIELDS = (
    "observed_fraction",
    "interpolated_fraction",
    "missing_fraction",
    "rejected_fraction",
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _serial_vector(values: np.ndarray) -> list[float | None]:
    return [_finite_float(value) for value in np.asarray(values).reshape(-1)]


def _serial_matrix(values: np.ndarray) -> list[list[float | None]]:
    array = np.asarray(values)
    return [[_finite_float(value) for value in row] for row in array]


def _safe_nanmedian(values: np.ndarray, axis=None) -> np.ndarray | float:
    values = np.asarray(values, dtype=float)
    if axis is None:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float("nan")
    result_shape = tuple(size for index, size in enumerate(values.shape) if index != axis)
    result = np.full(result_shape, np.nan, dtype=float)
    moved = np.moveaxis(values, axis, 0)
    for index in np.ndindex(result_shape):
        column = moved[(slice(None),) + index]
        finite = column[np.isfinite(column)]
        if finite.size:
            result[index] = np.median(finite)
    return result


def _state_fractions(states: np.ndarray) -> dict[str, float]:
    states = np.asarray(states, dtype=np.uint8)
    total = max(states.size, 1)
    return {
        "observed_fraction": float(np.count_nonzero(states == 0) / total),
        "interpolated_fraction": float(np.count_nonzero(states == 1) / total),
        "missing_fraction": float(np.count_nonzero(states == 2) / total),
        "rejected_fraction": float(np.count_nonzero(states == 3) / total),
    }


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _longest_circular_true_run(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=bool)
    if not values.size or not np.any(values):
        return 0
    if np.all(values):
        return int(values.size)
    doubled = np.concatenate((values, values))
    return min(_longest_true_run(doubled), int(values.size))


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
            length = stop - start
            longest[angle] = max(longest[angle], length)
            if length >= minimum_run:
                coherent[start:stop, angle] = True
            start = stop
    return coherent, longest


def _circular_angular_runs(mask: np.ndarray, minimum_samples: int) -> list[tuple[np.ndarray, bool]]:
    """Return contiguous angular bins, merging the 359/0 seam."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.size or not np.any(mask):
        return []
    if np.all(mask):
        return [(np.arange(mask.size, dtype=int), False)] if mask.size >= minimum_samples else []
    runs: list[np.ndarray] = []
    start = 0
    while start < mask.size:
        if not mask[start]:
            start += 1
            continue
        stop = start + 1
        while stop < mask.size and mask[stop]:
            stop += 1
        runs.append(np.arange(start, stop, dtype=int))
        start = stop
    crosses = False
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == mask.size - 1:
        runs = [np.concatenate((runs[-1], runs[0]))] + runs[1:-1]
        crosses = True
    result: list[tuple[np.ndarray, bool]] = []
    for index, run in enumerate(runs):
        if len(run) >= minimum_samples:
            result.append((run, crosses and index == 0))
    return result


def _sector_records(
    normalized: np.ndarray,
    pair_observed: np.ndarray,
    angles_deg: np.ndarray,
    coherent_cells: np.ndarray,
    direction: str,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    angular_mask = np.any(coherent_cells, axis=0)
    if len(angles_deg) > 1:
        sorted_angles = np.sort(np.mod(angles_deg, 360.0))
        angular_step = float(np.median(np.diff(np.r_[sorted_angles, sorted_angles[0] + 360.0])))
    else:
        angular_step = 360.0
    for indices, crosses_seam in _circular_angular_runs(angular_mask, minimum_samples):
        sector_cells = coherent_cells[:, indices]
        affected_pairs = np.flatnonzero(np.any(sector_cells, axis=1))
        if not len(affected_pairs):
            continue
        values = normalized[:, indices][sector_cells]
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        observation_window = pair_observed[np.ix_(affected_pairs, indices)]
        compression = np.maximum(0.0, 1.0 - values)
        expansion = np.maximum(0.0, values - 1.0)
        records.append(
            {
                "start_degrees": float(angles_deg[indices[0]] % 360.0),
                "end_degrees": float(angles_deg[indices[-1]] % 360.0),
                "angular_width_degrees": float(min(360.0, len(indices) * angular_step)),
                "first_affected_ring_pair": int(affected_pairs[0]),
                "last_affected_ring_pair": int(affected_pairs[-1]),
                "affected_pair_count": int(len(affected_pairs)),
                "affected_stack_fraction": float(len(affected_pairs) / max(normalized.shape[0], 1)),
                "median_normalized_spacing": float(np.median(values)),
                "maximum_compression": float(np.max(compression)) if direction == "compression" else 0.0,
                "maximum_expansion": float(np.max(expansion)) if direction == "expansion" else 0.0,
                "direct_observation_fraction": float(np.mean(observation_window)),
                "crosses_zero_degree_seam": bool(crosses_seam),
            }
        )
    return records


def _region_summary(
    name: str,
    ring_indices: np.ndarray,
    pair_indices: np.ndarray,
    ring_states: np.ndarray,
    pair_states: np.ndarray,
    angular_variation: np.ndarray,
    angular_range: np.ndarray,
) -> dict[str, Any]:
    ring_slice = ring_states[ring_indices] if len(ring_indices) else np.empty((0, ring_states.shape[1]), dtype=np.uint8)
    pair_slice = pair_states[pair_indices] if len(pair_indices) else np.empty((0, pair_states.shape[1]), dtype=np.uint8)
    ring_fractions = _state_fractions(ring_slice)
    pair_fractions = _state_fractions(pair_slice)
    pair_variation = angular_variation[pair_indices] if len(pair_indices) else np.empty(0)
    pair_range = angular_range[pair_indices] if len(pair_indices) else np.empty(0)
    return {
        "region": name,
        "ring_indices": [int(value) for value in ring_indices],
        "ring_pair_indices": [int(value) for value in pair_indices],
        "ring_observed_fraction": ring_fractions["observed_fraction"],
        "pair_direct_observation_fraction": pair_fractions["observed_fraction"],
        **ring_fractions,
        "pair_observed_fraction": pair_fractions["observed_fraction"],
        "pair_interpolated_fraction": pair_fractions["interpolated_fraction"],
        "pair_missing_fraction": pair_fractions["missing_fraction"],
        "pair_rejected_fraction": pair_fractions["rejected_fraction"],
        "robust_angular_variation": _finite_float(_safe_nanmedian(pair_variation)),
        "robust_angular_range": _finite_float(_safe_nanmedian(pair_range)),
    }


def _feature(
    value: Any,
    units: str,
    validity: str,
    direct_fraction: float,
    affected_rings: list[int],
    affected_angles: list[float],
    reason: str | None,
) -> dict[str, Any]:
    return {
        "value": value,
        "units": units,
        "validity": validity,
        "direct_observation_fraction": float(direct_fraction),
        "affected_rings": affected_rings,
        "affected_angles": affected_angles,
        "reason_if_invalid": reason,
    }


def analyse_full_stack(
    radii: np.ndarray,
    angles_deg: np.ndarray,
    observed: np.ndarray | None,
    *,
    interpolated: np.ndarray | None = None,
    rejected: np.ndarray | None = None,
    expected_ring_count: int | None = None,
    min_direct_coverage: float = 0.55,
    config: Any = None,
) -> dict[str, Any]:
    """Analyse every tracked ring and every adjacent pair over all meridians."""
    radii = np.asarray(radii, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if radii.ndim != 2:
        raise ValueError("radii must have shape (ring, meridian)")
    ring_count, meridians = radii.shape
    if angles_deg.shape != (meridians,):
        raise ValueError("angles_deg length must match the meridian dimension")

    finite = np.isfinite(radii)
    observed_input = finite.copy() if observed is None else np.asarray(observed, dtype=bool)
    interpolated_input = np.zeros_like(finite) if interpolated is None else np.asarray(interpolated, dtype=bool)
    rejected_input = np.zeros_like(finite) if rejected is None else np.asarray(rejected, dtype=bool)
    for name, mask in (
        ("observed", observed_input),
        ("interpolated", interpolated_input),
        ("rejected", rejected_input),
    ):
        if mask.shape != radii.shape:
            raise ValueError(f"{name} state shape must match radii")

    input_flags: list[str] = []
    if np.any(observed_input & interpolated_input) or np.any(observed_input & rejected_input) or np.any(interpolated_input & rejected_input):
        input_flags.append("overlapping_ring_states")
    if np.any(observed_input & ~finite) or np.any(interpolated_input & ~finite):
        input_flags.append("finite_radius_required_for_observed_or_interpolated_state")

    observed_mask = observed_input & finite & ~rejected_input
    interpolated_mask = interpolated_input & finite & ~observed_mask & ~rejected_input
    rejected_mask = rejected_input & ~observed_mask & ~interpolated_mask
    missing_mask = ~(observed_mask | interpolated_mask | rejected_mask)
    usable_identity = observed_mask | interpolated_mask
    if np.any(usable_identity & (radii <= 0.0)):
        input_flags.append("non_positive_radius")
    for angle in range(meridians):
        ordered_values = radii[:, angle][usable_identity[:, angle]]
        if len(ordered_values) > 1 and np.any(np.diff(ordered_values) <= 0.0):
            input_flags.append("non_monotonic_ring_order")
            break
    ring_states = np.full(radii.shape, 2, dtype=np.uint8)
    ring_states[observed_mask] = 0
    ring_states[interpolated_mask] = 1
    ring_states[rejected_mask] = 3

    pair_count = max(ring_count - 1, 0)
    spacing_raw = np.diff(radii, axis=0) if pair_count else np.empty((0, meridians), dtype=float)
    non_positive = np.isfinite(spacing_raw) & (spacing_raw <= 0.0)
    valid_spacing = spacing_raw.copy()
    valid_spacing[non_positive] = np.nan

    pair_observed = observed_mask[:-1] & observed_mask[1:] & np.isfinite(valid_spacing)
    pair_rejected = (rejected_mask[:-1] | rejected_mask[1:] | non_positive) if pair_count else np.empty((0, meridians), dtype=bool)
    pair_interpolated = (
        ~pair_observed
        & ~pair_rejected
        & np.isfinite(valid_spacing)
        & (observed_mask[:-1] | interpolated_mask[:-1])
        & (observed_mask[1:] | interpolated_mask[1:])
    ) if pair_count else np.empty((0, meridians), dtype=bool)
    pair_missing = ~(pair_observed | pair_interpolated | pair_rejected)
    pair_states = np.full((pair_count, meridians), 2, dtype=np.uint8)
    pair_states[pair_observed] = 0
    pair_states[pair_interpolated] = 1
    pair_states[pair_rejected] = 3

    direct_spacing = valid_spacing.copy()
    direct_spacing[~pair_observed] = np.nan
    baseline = np.asarray(_safe_nanmedian(direct_spacing, axis=1), dtype=float) if pair_count else np.empty(0)
    normalized = np.divide(
        valid_spacing,
        baseline[:, None],
        out=np.full_like(valid_spacing, np.nan),
        where=np.isfinite(baseline[:, None]) & (baseline[:, None] > 0.0),
    )

    absolute_deviation = np.abs(direct_spacing - baseline[:, None])
    mad = np.asarray(_safe_nanmedian(absolute_deviation, axis=1), dtype=float) if pair_count else np.empty(0)
    angular_variation = np.divide(mad, baseline, out=np.full_like(mad, np.nan), where=baseline > 0.0)
    angular_range = np.full(pair_count, np.nan, dtype=float)
    for pair in range(pair_count):
        values = direct_spacing[pair, np.isfinite(direct_spacing[pair])]
        if values.size and np.isfinite(baseline[pair]) and baseline[pair] > 0.0:
            angular_range[pair] = (np.percentile(values, 90) - np.percentile(values, 10)) / baseline[pair]

    radial_deviation = np.full(meridians, np.nan, dtype=float)
    compressed_fraction = np.zeros(meridians, dtype=float)
    expanded_fraction = np.zeros(meridians, dtype=float)
    valid_pair_fraction = np.zeros(meridians, dtype=float)
    direct_pair_fraction = np.zeros(meridians, dtype=float)
    for angle in range(meridians):
        values = normalized[:, angle]
        valid = np.isfinite(values)
        direct = valid & pair_observed[:, angle]
        if np.any(valid):
            radial_deviation[angle] = float(np.median(np.abs(values[valid] - 1.0)))
        if pair_count:
            valid_pair_fraction[angle] = float(np.count_nonzero(valid) / pair_count)
            direct_pair_fraction[angle] = float(np.count_nonzero(direct) / pair_count)
            compressed_fraction[angle] = float(np.count_nonzero(direct & (values < 1.0)) / pair_count)
            expanded_fraction[angle] = float(np.count_nonzero(direct & (values > 1.0)) / pair_count)

    compression_magnitude = float(getattr(config, "compression_magnitude_fraction", 0.05))
    expansion_magnitude = float(getattr(config, "expansion_magnitude_fraction", 0.05))
    minimum_coherent_run = max(2, int(getattr(config, "min_coherent_pair_run", 2)))
    minimum_sector_samples = max(2, int(getattr(config, "min_sector_angular_samples", 3)))
    significant_compression = pair_observed & (normalized <= (1.0 - compression_magnitude))
    significant_expansion = pair_observed & (normalized >= (1.0 + expansion_magnitude))
    coherent_compression, longest_compressed = _coherent_cells(significant_compression, minimum_coherent_run)
    coherent_expansion, longest_expanded = _coherent_cells(significant_expansion, minimum_coherent_run)
    compression_stack_fraction = np.mean(significant_compression, axis=0) if pair_count else np.zeros(meridians)
    expansion_stack_fraction = np.mean(significant_expansion, axis=0) if pair_count else np.zeros(meridians)
    compression_sectors = _sector_records(
        normalized, pair_observed, angles_deg, coherent_compression, "compression", minimum_sector_samples
    )
    expansion_sectors = _sector_records(
        normalized, pair_observed, angles_deg, coherent_expansion, "expansion", minimum_sector_samples
    )

    expected_cumulative = np.full(ring_count, np.nan, dtype=float)
    cumulative_residual = np.full_like(radii, np.nan)
    normalized_cumulative_residual = np.full_like(radii, np.nan)
    if ring_count:
        expected_cumulative[0] = 0.0
        cumulative_residual[0, finite[0]] = 0.0
        normalized_cumulative_residual[0, finite[0]] = 0.0
    for ring in range(1, ring_count):
        if np.all(np.isfinite(baseline[:ring])):
            expected_cumulative[ring] = float(np.sum(baseline[:ring]))
        if not np.isfinite(expected_cumulative[ring]) or expected_cumulative[ring] <= 0.0:
            continue
        complete_path = (
            np.all(usable_identity[: ring + 1], axis=0)
            & np.all(~rejected_mask[: ring + 1], axis=0)
            & np.all(~non_positive[:ring], axis=0)
        )
        cumulative_residual[ring, complete_path] = (
            radii[ring, complete_path] - radii[0, complete_path] - expected_cumulative[ring]
        )
        normalized_cumulative_residual[ring, complete_path] = (
            cumulative_residual[ring, complete_path] / expected_cumulative[ring]
        )

    ring_completeness = []
    for ring in range(ring_count):
        ring_completeness.append({"ring_index": ring, **_state_fractions(ring_states[ring])})
    pair_completeness = []
    for pair in range(pair_count):
        pair_completeness.append(
            {"ring_pair_index": pair, "rings": [pair, pair + 1], **_state_fractions(pair_states[pair])}
        )
    meridian_completeness = []
    for angle in range(meridians):
        meridian_completeness.append(
            {"angle_degrees": float(angles_deg[angle]), **_state_fractions(ring_states[:, angle])}
        )

    ring_regions = np.array_split(np.arange(ring_count, dtype=int), 3)
    pair_regions = np.array_split(np.arange(pair_count, dtype=int), 3)
    inner = _region_summary("inner", ring_regions[0], pair_regions[0], ring_states, pair_states, angular_variation, angular_range)
    middle = _region_summary("middle", ring_regions[1], pair_regions[1], ring_states, pair_states, angular_variation, angular_range)
    outer = _region_summary("outer", ring_regions[2], pair_regions[2], ring_states, pair_states, angular_variation, angular_range)

    full_ring_fractions = _state_fractions(ring_states)
    full_pair_fractions = _state_fractions(pair_states)
    maximum_pair = int(np.nanargmax(angular_variation)) if np.any(np.isfinite(angular_variation)) else None
    robust_variation = _finite_float(_safe_nanmedian(angular_variation))
    max_variation = _finite_float(np.nanmax(angular_variation)) if maximum_pair is not None else None
    max_cumulative = (
        float(np.nanmax(np.abs(normalized_cumulative_residual)))
        if np.any(np.isfinite(normalized_cumulative_residual))
        else None
    )

    ring_count_verified = expected_ring_count is not None
    count_matches = ring_count_verified and int(expected_ring_count) == ring_count
    max_missing_fraction = float(getattr(config, "max_missing_sector_fraction", 0.20))
    longest_missing_by_ring = [
        _longest_circular_true_run(~observed_mask[ring]) / max(meridians, 1) for ring in range(ring_count)
    ]
    major_sector_missing = any(value > max_missing_fraction for value in longest_missing_by_ring)
    completeness_reasons: list[str] = []
    if not ring_count_verified:
        completeness_reasons.append("expected_ring_count_unverified")
    if ring_count_verified and not count_matches:
        completeness_reasons.append("detected_ring_count_mismatch")
    if any(item["observed_fraction"] < min_direct_coverage for item in ring_completeness):
        completeness_reasons.append("insufficient_per_ring_direct_coverage")
    if any(item["observed_fraction"] < min_direct_coverage for item in pair_completeness):
        completeness_reasons.append("insufficient_per_pair_direct_coverage")
    if major_sector_missing:
        completeness_reasons.append("major_angular_sector_missing")
    if np.any(~np.isfinite(baseline)):
        completeness_reasons.append("missing_direct_pair_baseline")
    if input_flags:
        completeness_reasons.extend(input_flags)
    if np.any(non_positive):
        completeness_reasons.append("non_positive_spacing")
    completeness_reasons = sorted(set(completeness_reasons))
    complete_stack_available = bool(
        ring_count >= 2
        and ring_count_verified
        and count_matches
        and not completeness_reasons
    )

    full_summary = {
        **full_ring_fractions,
        "pair_observed_fraction": full_pair_fractions["observed_fraction"],
        "pair_interpolated_fraction": full_pair_fractions["interpolated_fraction"],
        "pair_missing_fraction": full_pair_fractions["missing_fraction"],
        "pair_rejected_fraction": full_pair_fractions["rejected_fraction"],
        "robust_angular_variation": robust_variation,
        "maximum_valid_ring_pair_variation": max_variation,
        "maximum_variation_ring_pair": maximum_pair,
        "robust_angular_range": _finite_float(_safe_nanmedian(angular_range)),
        "maximum_radial_stack_deviation": _finite_float(np.nanmax(radial_deviation)) if np.any(np.isfinite(radial_deviation)) else None,
        "maximum_normalized_cumulative_residual_magnitude": max_cumulative,
        "maximum_compressed_pair_run": int(np.max(longest_compressed)) if longest_compressed.size else 0,
        "maximum_expanded_pair_run": int(np.max(longest_expanded)) if longest_expanded.size else 0,
    }

    state_labels = np.asarray(STATE_NAMES, dtype=object)[ring_states].tolist()
    detected_ring_count = int(np.count_nonzero(np.any(finite, axis=1)))
    stack = {
        "expected_ring_count": int(expected_ring_count) if expected_ring_count is not None else None,
        "detected_ring_count": detected_ring_count,
        "ring_count_verified": ring_count_verified,
        "classification_performed": False,
        "analysed_ring_indices": list(range(ring_count)),
        "analysed_ring_pair_count": pair_count,
        "complete_stack_available": complete_stack_available,
        "spacing_matrix_shape": [pair_count, meridians],
        "state_matrix": state_labels,
        "inter_ring_spacing_matrix": _serial_matrix(spacing_raw),
        "normalized_inter_ring_spacing_matrix": _serial_matrix(normalized),
        "baseline_spacing_by_pair": _serial_vector(baseline),
        "ring_completeness": ring_completeness,
        "ring_pair_completeness": pair_completeness,
        "meridian_completeness": meridian_completeness,
        "angular_spacing_variation_by_pair": _serial_vector(angular_variation),
        "angular_spacing_range_by_pair": _serial_vector(angular_range),
        "radial_stack_deviation_by_meridian": _serial_vector(radial_deviation),
        "compressed_pair_fraction_by_meridian": _serial_vector(compressed_fraction),
        "expanded_pair_fraction_by_meridian": _serial_vector(expanded_fraction),
        "valid_pair_fraction_by_meridian": _serial_vector(valid_pair_fraction),
        "direct_pair_fraction_by_meridian": _serial_vector(direct_pair_fraction),
        "neighbouring_ring_coherence": {
            "longest_compressed_pair_run": [int(value) for value in longest_compressed],
            "longest_expanded_pair_run": [int(value) for value in longest_expanded],
            "compressed_stack_fraction": _serial_vector(compression_stack_fraction),
            "expanded_stack_fraction": _serial_vector(expansion_stack_fraction),
        },
        "expected_cumulative_position": _serial_vector(expected_cumulative),
        "cumulative_radial_residuals": _serial_matrix(cumulative_residual),
        "normalized_cumulative_residuals": _serial_matrix(normalized_cumulative_residual),
        "compression_sectors": compression_sectors,
        "expansion_sectors": expansion_sectors,
        "inner_region_summary": inner,
        "middle_region_summary": middle,
        "outer_region_summary": outer,
        "full_stack_summary": full_summary,
        "longest_missing_sector_fraction_by_ring": [float(value) for value in longest_missing_by_ring],
        "completeness_reason_codes": completeness_reasons,
        "units": "pixels and dimensionless image-space proxy",
    }

    validity = "VALID" if not input_flags and not np.any(non_positive) else "INVALID"
    reason = None if validity == "VALID" else ", ".join(sorted(set(input_flags + (["non_positive_spacing"] if np.any(non_positive) else []))))
    direct_fraction = float(np.sum(observed_mask) / max((expected_ring_count or ring_count) * meridians, 1))
    compression_angles = sorted(
        {float(angles_deg[index] % 360.0) for index in np.flatnonzero(np.any(coherent_compression, axis=0))}
    )
    expansion_angles = sorted(
        {float(angles_deg[index] % 360.0) for index in np.flatnonzero(np.any(coherent_expansion, axis=0))}
    )
    explainable_features = {
        "INTER_RING_SPACING_MATRIX": _feature(
            stack["inter_ring_spacing_matrix"], "pixels", validity, full_pair_fractions["observed_fraction"],
            list(range(ring_count)), list(map(float, angles_deg)), reason,
        ),
        "ANGULAR_INTER_RING_SPACING_VARIATION": _feature(
            stack["angular_spacing_variation_by_pair"], "dimensionless image-space proxy", validity,
            full_pair_fractions["observed_fraction"], list(range(ring_count)), list(map(float, angles_deg)), reason,
        ),
        "RADIAL_STACK_SPACING_CONSISTENCY": _feature(
            stack["radial_stack_deviation_by_meridian"], "dimensionless image-space proxy", validity,
            full_pair_fractions["observed_fraction"], list(range(ring_count)), list(map(float, angles_deg)), reason,
        ),
        "NEIGHBOURING_RING_COHERENCE": _feature(
            stack["neighbouring_ring_coherence"], "dimensionless image-space proxy", validity,
            full_pair_fractions["observed_fraction"], list(range(ring_count)), sorted(set(compression_angles + expansion_angles)), reason,
        ),
        "CUMULATIVE_RADIAL_RESIDUAL": _feature(
            stack["cumulative_radial_residuals"], "pixels", validity, direct_fraction,
            list(range(ring_count)), list(map(float, angles_deg)), reason,
        ),
        "FULL_STACK_DIRECT_COVERAGE": _feature(
            direct_fraction, "dimensionless image-space proxy", validity, direct_fraction,
            list(range(ring_count)), list(map(float, angles_deg)), reason,
        ),
        "FULL_STACK_COMPRESSION": _feature(
            compression_sectors, "dimensionless image-space proxy", validity,
            full_pair_fractions["observed_fraction"], sorted({p for s in compression_sectors for p in range(s["first_affected_ring_pair"], s["last_affected_ring_pair"] + 2)}), compression_angles, reason,
        ),
        "FULL_STACK_EXPANSION": _feature(
            expansion_sectors, "dimensionless image-space proxy", validity,
            full_pair_fractions["observed_fraction"], sorted({p for s in expansion_sectors for p in range(s["first_affected_ring_pair"], s["last_affected_ring_pair"] + 2)}), expansion_angles, reason,
        ),
        "INNER_MIDDLE_OUTER_CONSISTENCY": _feature(
            {"inner": inner, "middle": middle, "outer": outer}, "dimensionless image-space proxy", validity,
            direct_fraction, list(range(ring_count)), list(map(float, angles_deg)), reason,
        ),
    }

    return {
        "full_stack_analysis": stack,
        "explainable_features": explainable_features,
        "valid_geometry": validity == "VALID",
        "invalid_reason_codes": sorted(set(input_flags + (["non_positive_spacing"] if np.any(non_positive) else []))),
        "_grids": {
            "spacing_matrix": spacing_raw,
            "normalized_spacing": normalized,
            "ring_states": ring_states,
            "pair_states": pair_states,
            "angular_variation": angular_variation,
            "angular_range": angular_range,
            "radial_stack_deviation": radial_deviation,
            "longest_compressed_pair_run": longest_compressed,
            "longest_expanded_pair_run": longest_expanded,
            "compression_stack_fraction": compression_stack_fraction,
            "expansion_stack_fraction": expansion_stack_fraction,
            "coherent_compression": coherent_compression,
            "coherent_expansion": coherent_expansion,
            "cumulative_residual": cumulative_residual,
            "normalized_cumulative_residual": normalized_cumulative_residual,
            "ring_observed": observed_mask,
            "pair_observed": pair_observed,
        },
    }
