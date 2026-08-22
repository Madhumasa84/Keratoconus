"""Ordered polar peak tracking with explicit missing and interpolated states."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RadialConfig, TrackingConfig
from .radial_scan import RadialResult


@dataclass
class TrackingResult:
    radii: np.ndarray  # ring x angle; observed/interpolated values only
    observed: np.ndarray  # direct peak observations
    interpolated: np.ndarray  # short-gap, non-observed estimates
    rejected: np.ndarray  # candidates rejected as order/outlier violations
    confidence: float
    missing_fraction: float
    direct_observation_fraction: float
    duplicate_removals: int
    unassigned_extra_peak_count: int
    identity_shift_fraction: float
    ring_completeness: np.ndarray
    order_change_fraction: float
    status: str
    flags: list[str]


def _align_ordered(
    expected: np.ndarray,
    candidates: np.ndarray,
    strengths: np.ndarray,
    radial: RadialConfig,
    tracking: TrackingConfig,
) -> tuple[np.ndarray, int]:
    """Monotone dynamic alignment of one meridian's candidates to ring IDs.

    Sequence alignment means an individual candidate can appear at most once and
    assignments retain strictly increasing radial order by construction.
    """
    n, m = len(expected), len(candidates)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    action = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 skip ring, 2 skip peak, 3 match
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + tracking.missing_penalty
        action[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + tracking.extra_peak_penalty
        action[0, j] = 2
    strength_scale = max(float(np.nanmedian(strengths)) if len(strengths) else 1.0, 1.0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (dp[i - 1, j] + tracking.missing_penalty, 1),
                (dp[i, j - 1] + tracking.extra_peak_penalty, 2),
            ]
            delta = abs(float(candidates[j - 1] - expected[i - 1]))
            if delta <= radial.max_radial_jump_px:
                match_cost = delta / max(radial.max_radial_jump_px, 1e-6)
                match_cost -= min(float(strengths[j - 1]) / strength_scale, 2.0) * 0.08
                choices.append((dp[i - 1, j - 1] + match_cost, 3))
            dp[i, j], action[i, j] = min(choices, key=lambda value: value[0])
    assigned = np.full(n, np.nan, dtype=float)
    skipped_peaks = 0
    i, j = n, m
    while i > 0 or j > 0:
        step = int(action[i, j])
        if step == 3:
            assigned[i - 1] = candidates[j - 1]
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        elif step == 2:
            skipped_peaks += 1
            j -= 1
        else:  # defensive fallback for a numerical corner case
            if i:
                i -= 1
            elif j:
                skipped_peaks += 1
                j -= 1
    return assigned, skipped_peaks


def _interpolate_short_gaps(values: np.ndarray, observed: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate only bounded internal gaps; never join across the 0/360 seam."""
    interpolated = np.zeros_like(observed, dtype=bool)
    for ring in range(values.shape[0]):
        missing = ~np.isfinite(values[ring])
        start = 0
        while start < values.shape[1]:
            if not missing[start]:
                start += 1
                continue
            stop = start
            while stop < values.shape[1] and missing[stop]:
                stop += 1
            length = stop - start
            # A gap touching an image-cycle edge is deliberately left missing.
            if 0 < start and stop < values.shape[1] and length <= max_gap:
                left, right = values[ring, start - 1], values[ring, stop]
                if np.isfinite(left) and np.isfinite(right):
                    values[ring, start:stop] = np.linspace(left, right, length + 2)[1:-1]
                    interpolated[ring, start:stop] = True
            start = stop
    return interpolated


def _reject_invalid_order(values: np.ndarray, rejected: np.ndarray) -> None:
    """Final invariant: a valid meridian is finite, positive and strictly ordered."""
    for angle in range(values.shape[1]):
        previous = 0.0
        for ring in range(values.shape[0]):
            value = values[ring, angle]
            if not np.isfinite(value):
                continue
            if value <= 0.0 or value <= previous:
                rejected[ring, angle] = True
                values[ring, angle] = np.nan
            else:
                previous = value


def track_rings(
    scan: RadialResult,
    tracking: TrackingConfig = TrackingConfig(),
    radial: RadialConfig = RadialConfig(),
) -> TrackingResult:
    """Track polar peaks while making invalid spacing impossible by construction."""
    references = np.asarray(scan.reference_radii, dtype=float)
    angles = len(scan.angles_deg)
    if not len(references):
        empty = np.empty((0, angles), dtype=float)
        return TrackingResult(empty, empty.astype(bool), empty.astype(bool), empty.astype(bool), 0.0, 1.0, 0.0, 0, 0, 1.0, empty,
                              scan.order_change_fraction, "FAIL", ["no_reliable_polar_ring_peaks"])
    # Reference radii are aggregate peaks, already sorted.  They may not be
    # silently collapsed: if this invariant cannot hold, analysis must fail.
    if np.any(~np.isfinite(references)) or np.any(np.diff(references) <= 0):
        raise ValueError("polar reference rings are not strictly ordered")
    out = np.full((len(references), angles), np.nan, dtype=float)
    rejected = np.zeros_like(out, dtype=bool)
    duplicate_removals = 0
    unassigned_extra_peaks = 0
    # Start at the richest meridian and walk around the full cycle.  Therefore
    # the 359°→0° transition is treated as a normal neighbouring transition,
    # rather than a special unconnected edge of an array.
    start = int(np.argmax([len(values) for values in scan.candidate_radii])) if angles else 0
    previous = np.full(len(references), np.nan, dtype=float)
    for offset in range(angles):
        angle = (start + offset) % angles
        candidates = scan.candidate_radii[angle]
        strengths = scan.candidate_strengths[angle]
        candidates = np.asarray(candidates, dtype=float)
        strengths = np.asarray(strengths, dtype=float)
        valid = np.isfinite(candidates) & (candidates > 0)
        candidates, strengths = candidates[valid], strengths[valid]
        # Candidates originate from find_peaks but are sorted defensively before
        # alignment. Duplicate candidate radii are removed, never reused.
        if len(candidates):
            order = np.argsort(candidates)
            candidates, strengths = candidates[order], strengths[order]
            unique = np.r_[True, np.diff(candidates) > 1e-6]
            duplicate_removals += int(np.count_nonzero(~unique))
            candidates, strengths = candidates[unique], strengths[unique]
        expected = references.copy()
        usable_previous = np.isfinite(previous)
        expected[usable_previous] = (
            (1.0 - tracking.continuity_weight) * references[usable_previous]
            + tracking.continuity_weight * previous[usable_previous]
        )
        assigned, skipped = _align_ordered(expected, candidates, strengths, radial, tracking)
        unassigned_extra_peaks += skipped
        out[:, angle] = assigned
        previous = assigned
    observed = np.isfinite(out)
    interpolated = _interpolate_short_gaps(out, observed, tracking.max_short_gap_meridians)
    _reject_invalid_order(out, rejected)
    observed &= ~rejected
    interpolated &= ~rejected

    direct_fraction = float(np.mean(observed)) if out.size else 0.0
    missing_fraction = float(np.mean(~np.isfinite(out))) if out.size else 1.0
    completeness = np.mean(np.isfinite(out), axis=1) if len(out) else np.empty(0, dtype=float)
    order_changes = float(np.mean(rejected)) if out.size else 1.0
    # Difference from each aggregate reference is an identity-stability signal;
    # it does not permit reassignment that could create a ring crossing.
    shifts = np.abs(out - references[:, None]) > radial.max_radial_jump_px * 0.75
    identity_shift = float(np.mean(shifts[np.isfinite(out)])) if np.any(np.isfinite(out)) else 1.0
    cyclic_penalty = 0.0
    if angles > 1:
        first, last = out[:, 0], out[:, -1]
        cyclic = np.isfinite(first) & np.isfinite(last)
        if np.any(cyclic):
            cyclic_penalty = float(np.mean(np.abs(first[cyclic] - last[cyclic]) > radial.max_radial_jump_px))
    confidence = float(np.clip(direct_fraction * (1.0 - order_changes) * (1.0 - identity_shift) * (1.0 - cyclic_penalty), 0.0, 1.0))

    flags: list[str] = []
    if len(references) < tracking.min_tracked_rings:
        flags.append("too_few_tracked_rings")
    if direct_fraction < tracking.min_direct_coverage:
        flags.append("insufficient_direct_ring_observation")
    if confidence < tracking.min_tracking_confidence:
        flags.append("low_tracking_confidence")
    if order_changes > tracking.max_rejected_fraction:
        flags.append("ring_order_violation_rejected")
    return TrackingResult(
        radii=out,
        observed=observed,
        interpolated=interpolated,
        rejected=rejected,
        confidence=confidence,
        missing_fraction=missing_fraction,
        direct_observation_fraction=direct_fraction,
        duplicate_removals=duplicate_removals,
        unassigned_extra_peak_count=unassigned_extra_peaks,
        identity_shift_fraction=identity_shift,
        ring_completeness=completeness,
        order_change_fraction=order_changes,
        status="PASS" if not flags else "FAIL",
        flags=flags,
    )
