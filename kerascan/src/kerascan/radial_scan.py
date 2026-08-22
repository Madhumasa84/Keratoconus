"""Polar intensity unwrapping and per-meridian bright-ring centreline peaks.

This module deliberately works from the contrast-enhanced intensity image, not
from connected components in a binary edge mask.  A connected binary mask is a
useful debug artefact, but it is not a reliable representation of Placido-ring
identity when adjacent white bands are bridged by glare, lashes, or threshold
noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .config import RadialConfig
from .image_io import to_gray


@dataclass
class RadialResult:
    """Observed candidate peaks, indexed by meridian before identity tracking."""

    angles_deg: np.ndarray
    radial_positions_px: np.ndarray
    polar_image: np.ndarray  # angle x radius
    polar_smoothed: np.ndarray  # angle x radius
    candidate_radii: list[np.ndarray]
    candidate_strengths: list[np.ndarray]
    reference_radii: np.ndarray
    ring_count_source: str
    peak_coverage: float
    order_change_fraction: float


def sample_polar(
    image: np.ndarray,
    center: tuple[float, float],
    outer_radius: float,
    config: RadialConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a centred annulus without resizing the ROI.

    Rows are uniformly sampled meridians and columns are image-space radii.
    Values outside the ROI are represented by ``NaN`` rather than a fabricated
    zero intensity.
    """
    gray = to_gray(image).astype(float)
    start = max(3.0, float(outer_radius) * config.min_radius_fraction)
    stop = max(start + config.radial_sample_step, float(outer_radius) * 0.95)
    radii = np.arange(start, stop, config.radial_sample_step, dtype=float)
    angles = np.linspace(0.0, 360.0, config.meridians, endpoint=False, dtype=float)
    theta = np.deg2rad(angles)
    xx = np.rint(center[0] + np.cos(theta)[:, None] * radii[None, :]).astype(int)
    yy = np.rint(center[1] + np.sin(theta)[:, None] * radii[None, :]).astype(int)
    valid = (xx >= 0) & (xx < gray.shape[1]) & (yy >= 0) & (yy < gray.shape[0])
    polar = np.full(xx.shape, np.nan, dtype=float)
    polar[valid] = gray[yy[valid], xx[valid]]
    return angles, radii, polar


def _detect_peaks(profile: np.ndarray, config: RadialConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return bright-band centreline candidates and their local prominence."""
    if np.count_nonzero(np.isfinite(profile)) < 6:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    values = np.asarray(profile, dtype=float).copy()
    finite = np.isfinite(values)
    values[~finite] = np.nanmedian(values[finite])
    prominence = max(
        config.min_peak_prominence,
        float(np.nanstd(values) * config.peak_prominence_std_fraction),
    )
    distance = max(1, int(round(config.min_peak_separation_px / config.radial_sample_step)))
    peaks, props = find_peaks(values, prominence=prominence, distance=distance)
    return peaks.astype(int), np.asarray(props.get("prominences", []), dtype=float)


def median_count_of(counts: list[int]) -> float:
    """Typical number of mires a single meridian resolves."""
    return float(np.median(counts)) if counts else 0.0


def _merge_split_reference_peaks(
    reference: np.ndarray,
    min_pitch_fraction: float = 0.60,
) -> np.ndarray:
    """Collapse reference peaks that sit far closer together than the ring pitch.

    The reference profile is a median across meridians. A decentred or slightly
    elliptical pattern places each mire at a different radius on each meridian,
    so its aggregate peak broadens and can split into two. Those phantom slots
    are the damaging kind of error: individual meridians only ever resolve one
    of the pair, so the two ring identities end up observed on complementary
    meridians, adjacent-pair co-observation collapses towards zero, and the
    full-stack completeness check rejects an otherwise good capture.

    Real mire pitch varies smoothly with radius, so the median gap is a robust
    scale: any gap far below it indicates one mire counted twice rather than two
    genuinely adjacent mires.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.size < 3:
        return reference
    gaps = np.diff(reference)
    finite = gaps[np.isfinite(gaps) & (gaps > 0)]
    if finite.size == 0:
        return reference
    threshold = float(np.median(finite)) * min_pitch_fraction
    if threshold <= 0:
        return reference
    merged: list[float] = [float(reference[0])]
    for value in reference[1:]:
        if value - merged[-1] < threshold:
            # One mire resolved twice: keep a single identity at their midpoint.
            merged[-1] = (merged[-1] + float(value)) / 2.0
        else:
            merged.append(float(value))
    return np.asarray(merged, dtype=float)


def radial_scan(
    image: np.ndarray,
    center: tuple[float, float],
    outer_radius: float,
    config: RadialConfig = RadialConfig(),
) -> RadialResult:
    """Find radial bright-band centrelines separately on each meridian.

    The expected device ring count is deliberately optional.  With no verified
    hardware configuration, an aggregate polar profile supplies an explicitly
    provisional count for tracking; ``max_rings`` remains only a safety cap.
    """
    angles, radii, polar = sample_polar(image, center, outer_radius, config)
    # The small one-dimensional radial smoothing avoids double peaks from the
    # two sides of a white band.  No dilation or connected-component joining is
    # involved in peak extraction.
    filled = polar.copy()
    per_radius = np.nanmedian(filled, axis=0)
    for row in range(len(filled)):
        valid = np.isfinite(filled[row])
        if not np.all(valid):
            filled[row, ~valid] = per_radius[~valid]
    smoothed = gaussian_filter1d(filled, config.peak_smoothing_sigma_px, axis=1, mode="nearest")
    candidates: list[np.ndarray] = []
    strengths: list[np.ndarray] = []
    counts = []
    for row in smoothed:
        peak_indices, prominence = _detect_peaks(row, config)
        candidates.append(radii[peak_indices])
        strengths.append(prominence)
        counts.append(len(peak_indices))
    aggregate = gaussian_filter1d(np.nanmedian(smoothed, axis=0), config.peak_smoothing_sigma_px)
    aggregate_peaks, aggregate_strengths = _detect_peaks(aggregate, config)
    if config.expected_ring_count is not None:
        expected = int(config.expected_ring_count)
        if expected < 1:
            raise ValueError("expected_ring_count must be positive when configured")
        # Median ordinal radii retain the individual horizontal polar bands when
        # an ellipse spreads their aggregate radial profile.  We use only rays
        # with exactly the verified number of candidates; no absent ring is made
        # up here.  If none qualify, normal segmentation failure handling takes
        # over below.
        ordinal_rows = [row for row in candidates if len(row) == expected]
        if ordinal_rows:
            reference = np.nanmedian(np.asarray(ordinal_rows, dtype=float), axis=0)
        else:
            if len(aggregate_peaks) > expected:
                strongest = np.argsort(aggregate_strengths)[-expected:]
                aggregate_peaks = np.sort(aggregate_peaks[strongest])
            reference = radii[aggregate_peaks]
        count_source = "verified_device_config"
    else:
        # Prefer what individual meridians actually resolve over the aggregate
        # profile. The aggregate is a median across meridians, so a decentred or
        # elliptical pattern smears each mire across a band of radii and can
        # split it into two peaks; those phantom identities are then observed on
        # complementary meridians and destroy adjacent-pair co-observation.
        # Rays that resolve the typical number of mires give a cleaner ordinal
        # reference, and no absent ring is invented here.
        provisional = int(round(median_count_of(counts)))
        ordinal_rows = [row for row in candidates if len(row) == provisional]
        reference = None
        if provisional >= 2 and len(ordinal_rows) >= max(3, int(0.05 * len(candidates))):
            reference = np.nanmedian(np.asarray(ordinal_rows, dtype=float), axis=0)
        if reference is None or not np.all(np.isfinite(reference)):
            if len(aggregate_peaks) > config.max_rings:
                strongest = np.argsort(aggregate_strengths)[-config.max_rings:]
                aggregate_peaks = np.sort(aggregate_peaks[strongest])
            reference = radii[aggregate_peaks]
        # Guard against any doublet that survives either route.
        reference = _merge_split_reference_peaks(reference)
        count_source = "provisional_polar_profile"
    median_count = float(np.median(counts)) if counts else 0.0
    order_change = float(np.mean(np.abs(np.asarray(counts, dtype=float) - median_count) > 2.0)) if counts else 1.0
    coverage = float(np.mean(np.asarray(counts) > 0)) if counts else 0.0
    return RadialResult(
        angles_deg=angles,
        radial_positions_px=radii,
        polar_image=polar,
        polar_smoothed=smoothed,
        candidate_radii=candidates,
        candidate_strengths=strengths,
        reference_radii=reference,
        ring_count_source=count_source,
        peak_coverage=coverage,
        order_change_fraction=order_change,
    )
