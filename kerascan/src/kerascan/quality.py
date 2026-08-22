"""Conservative pre-classification image-quality gate.

Three quality levels:
  ACCEPTABLE              — Placido pattern clearly analysable; proceed normally.
  ACCEPTABLE_WITH_WARNING — Mild apparent quality issue; geometry remained
                            technically analysable after all downstream checks.
  REJECTED                — Blur, exposure, centring or occlusion prevents
                            reliable ring geometry.

Quality is measured primarily inside the Placido ring band (50–90 % of
outer_radius) and not from the full facial image, where skin and sclera
dominate the sharpness metric.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import QualityConfig


def _robust_noise(gray: np.ndarray) -> float:
    residual = gray.astype(float) - cv2.GaussianBlur(gray, (0, 0), 1.2).astype(float)
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def _ring_band_mask(
    gray: np.ndarray,
    center: tuple[float, float],
    outer_radius: float,
    inner_fraction: float = 0.45,
    outer_fraction: float = 0.92,
) -> np.ndarray:
    """Return a boolean mask for the Placido ring band."""
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - center[0], yy - center[1])
    return (rr >= outer_radius * inner_fraction) & (rr <= outer_radius * outer_fraction)


def _ring_band_laplacian(gray: np.ndarray, ring_mask: np.ndarray) -> float:
    """Laplacian variance restricted to the ring band."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    pixels = lap[ring_mask]
    if not pixels.size:
        return 0.0
    return float(pixels.var())


def _ring_edge_contrast(gray: np.ndarray, ring_mask: np.ndarray) -> float:
    """Local contrast (std) within the ring band."""
    pixels = gray[ring_mask]
    if not pixels.size:
        return 0.0
    return float(pixels.std())


def _ring_band_sector_means(
    gray: np.ndarray,
    center: tuple[float, float],
    outer_radius: float,
) -> np.ndarray:
    """Mean intensity per 15-degree ring-band sector; NaN where a sector has no pixels."""
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - center[0], yy - center[1])
    th = (np.degrees(np.arctan2(yy - center[1], xx - center[0])) % 360)
    ring_band = (rr > outer_radius * 0.50) & (rr < outer_radius * 0.90)
    return np.array([
        float(np.mean(pixels)) if (pixels := gray[ring_band & (th >= a * 15) & (th < (a + 1) * 15)]).size else np.nan
        for a in range(24)
    ])


def mire_contrast_amplitude(
    gray: np.ndarray,
    center: tuple[float, float],
    outer_radius: float,
) -> float:
    """Bright-to-dark swing across the ring band.

    Placido mires are specular reflections: bright rings alternating with dark
    gaps, which produces a large robust intensity spread. Skin, eyelashes and
    eyebrow shading vary smoothly and produce a much smaller one. This is the
    signal that distinguishes "we located the ring pattern" from "we locked
    onto something else that happened to look concentric" -- a distinction the
    ring-shape and periodicity measures do NOT make, because circles fitted to
    smooth skin are near-perfect and shade periodically.
    """
    h, w = gray.shape
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    radii = np.arange(max(4.0, outer_radius * 0.15), max(outer_radius * 0.95, 6.0), 1.0)
    if radii.size == 0:
        return 0.0
    xs = np.clip(np.rint(center[0] + np.outer(np.cos(theta), radii)).astype(int), 0, w - 1)
    ys = np.clip(np.rint(center[1] + np.outer(np.sin(theta), radii)).astype(int), 0, h - 1)
    band = gray[ys, xs].astype(float)
    if band.size == 0:
        return 0.0
    return float(np.percentile(band, 95) - np.percentile(band, 5))


def _sector_obstruction(
    gray: np.ndarray,
    center: tuple[float, float],
    outer_radius: float,
) -> float:
    """Fraction of 15-degree sectors that appear dark (eyelid/lash occlusion proxy)."""
    sector_means = _ring_band_sector_means(gray, center, outer_radius)
    # A sector with no pixels in the ring band (e.g. near the image edge)
    # carries no evidence either way; exclude it rather than let it silently
    # count toward the denominator as "not dark".
    valid = np.isfinite(sector_means)
    if not np.any(valid):
        return 0.0
    threshold = max(8.0, float(np.median(sector_means[valid])) * 0.65)
    dark = sector_means[valid] < threshold
    return float(np.mean(dark))


def evaluate_quality(
    image: np.ndarray,
    center: tuple[float, float],
    outer_radius: float,
    config: QualityConfig = QualityConfig(),
    ring_mask: np.ndarray | None = None,
) -> dict:
    """Return acquisition quality at three levels; segmentation assessed separately.

    ``ring_mask`` is an ignored compatibility argument for callers from earlier
    research versions.  Mask density and broken-ring evidence must not silently
    reduce an acquisition score because those are segmentation/tracking outcomes,
    not a reason to tell an operator that every failure needs a recapture.

    Quality is measured primarily inside the Placido ring band rather than the
    full image so that facial skin and sclera do not dominate the sharpness
    metric and cause valid ring-pattern images to be falsely rejected.

    Returns:
        dict with keys:
          - ``gradable`` (bool): True for ACCEPTABLE or ACCEPTABLE_WITH_WARNING
          - ``quality_level``: "ACCEPTABLE" | "ACCEPTABLE_WITH_WARNING" | "REJECTED"
          - ``status``: same as quality_level for API consistency
          - ``quality_score``: 0-100
          - ``flags``: list of named issues
          - ``metrics``: detailed numeric measurements
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape

    flags: list[str] = []
    warning_flags: list[str] = []
    metrics: dict = {}

    metrics["roi_width_px"] = int(w)
    metrics["roi_height_px"] = int(h)
    metrics["resolution_min_side_px"] = int(min(w, h))

    if min(w, h) < config.min_roi_side_px:
        flags.append("low_resolution")

    # ------------------------------------------------------------------
    # Ring-band-aware sharpness (primary metric)
    # ------------------------------------------------------------------
    rb_mask = _ring_band_mask(gray, center, outer_radius)
    ring_band_pixels = gray[rb_mask]

    # Ring-band Laplacian variance
    ring_lap = _ring_band_laplacian(gray, rb_mask)
    metrics["ring_band_laplacian_variance"] = ring_lap

    # Full-image Laplacian (kept for back-compat, but not the primary gate)
    full_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["laplacian_variance"] = full_lap

    # Gate on ring-band sharpness; fall back to full-image for very small images
    effective_lap = ring_lap if ring_band_pixels.size > 200 else full_lap
    warning_lap_threshold = config.min_laplacian_variance * getattr(config, "warning_laplacian_fraction", 0.40)
    if effective_lap < config.min_laplacian_variance:
        # Distinguish rejection from warning based on severity
        if effective_lap < warning_lap_threshold:
            flags.append("blur")
        else:
            warning_flags.append("mild_blur")

    # ------------------------------------------------------------------
    # Exposure and contrast (ring band preferred)
    # ------------------------------------------------------------------
    if ring_band_pixels.size > 200:
        band_mean = float(ring_band_pixels.mean())
        band_contrast = float(ring_band_pixels.std())
    else:
        band_mean = float(gray.mean())
        band_contrast = float(gray.std())

    metrics["mean_intensity"] = band_mean
    metrics["contrast_std"] = band_contrast

    if band_mean < config.min_mean_intensity:
        flags.append("underexposed")
    if band_contrast < config.min_contrast:
        warning_contrast_threshold = config.min_contrast * getattr(config, "warning_contrast_fraction", 0.55)
        if band_contrast < warning_contrast_threshold:
            flags.append("low_contrast")
        else:
            warning_flags.append("mild_low_contrast")

    # Saturation
    sat = float(np.mean(gray >= 250))
    metrics["saturation_fraction"] = sat
    if sat > config.max_saturation_fraction:
        flags.append("glare_or_saturation")

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------
    noise = _robust_noise(gray)
    metrics["noise_sigma_estimate"] = noise
    if noise > config.max_noise_sigma:
        flags.append("sensor_noise")

    # ------------------------------------------------------------------
    # Centring
    # ------------------------------------------------------------------
    dist = np.hypot(center[0] - w / 2, center[1] - h / 2)
    centring = 1 - dist / max(outer_radius, 1)
    metrics["pattern_centring_ratio"] = centring
    if centring < config.min_centring_ratio:
        flags.append("pattern_off_centre")

    # Pattern size
    pattern_ratio = float(outer_radius / max(min(h, w) / 2, 1))
    metrics["pattern_radius_fraction"] = pattern_ratio
    if outer_radius < config.min_roi_side_px * 0.22:
        flags.append("placido_pattern_too_small")

    # ------------------------------------------------------------------
    # Eyelid / eyelash obstruction (ring-band sectors)
    # ------------------------------------------------------------------
    dark_fraction = _sector_obstruction(gray, center, outer_radius)
    metrics["dark_peripheral_sector_fraction"] = dark_fraction

    sector_means_24 = _ring_band_sector_means(gray, center, outer_radius)
    valid_24 = np.isfinite(sector_means_24)
    if np.any(valid_24):
        threshold_24 = max(8.0, float(np.median(sector_means_24[valid_24])) * 0.65)
        dark_24 = valid_24 & (sector_means_24 < threshold_24)
    else:
        dark_24 = np.zeros(24, dtype=bool)
    obstruction_arr = np.r_[dark_24, dark_24[:3]]
    if np.any(np.convolve(obstruction_arr, np.ones(4, dtype=int), "valid") >= 4):
        flags.append("possible_eyelid_or_eyelash_obstruction")

    # Visible ring coverage is measured from polar peaks later in the pipeline.
    metrics["visible_ring_sector_coverage"] = None

    # ------------------------------------------------------------------
    # Quality level determination
    # ------------------------------------------------------------------
    critical = {
        "low_resolution", "underexposed", "blur", "low_contrast",
        "glare_or_saturation", "sensor_noise", "pattern_off_centre",
        "possible_eyelid_or_eyelash_obstruction", "placido_pattern_too_small",
    }
    warning_only = {"mild_blur", "mild_low_contrast"}

    all_flags = sorted(set(flags + warning_flags))
    critical_present = any(f in critical for f in flags)
    warning_present = any(f in warning_only for f in warning_flags)

    if critical_present:
        quality_level = "REJECTED"
        gradable = False
    elif warning_present:
        quality_level = "ACCEPTABLE_WITH_WARNING"
        gradable = True
    else:
        quality_level = "ACCEPTABLE"
        gradable = True

    score = max(0, 100 - 12 * len(set(flags)) - 5 * len(set(warning_flags)))

    return {
        "gradable": gradable,
        "quality_level": quality_level,
        "quality_score": int(score),
        "flags": all_flags,
        "metrics": metrics,
        "status": quality_level,
        # Legacy field - maps to gradable
        "acceptable": gradable,
    }
