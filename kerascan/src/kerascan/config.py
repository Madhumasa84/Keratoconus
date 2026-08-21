"""Configuration without assumptions about camera, crop size, or mire count."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ROIConfig:
    margin: float = 0.18
    min_radius_px: float = 38.0
    max_radius_fraction: float = 0.48
    manual_center: tuple[float, float] | None = None
    manual_box: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1 in source image


@dataclass(frozen=True)
class QualityConfig:
    min_roi_side_px: int = 180
    min_laplacian_variance: float = 20.0
    min_mean_intensity: float = 24.0
    min_contrast: float = 18.0
    max_saturation_fraction: float = 0.12
    max_noise_sigma: float = 28.0
    min_centring_ratio: float = 0.18
    min_angular_coverage: float = 0.42
    min_ring_fraction: float = 0.008


@dataclass(frozen=True)
class SegmentationConfig:
    method: str = "traditional"  # "traditional" or a caller-provided UNet adapter
    clahe_clip_limit: float = 2.5
    clahe_tile_size: int = 8
    inner_exclusion_fraction: float = 0.07
    circular_mask_fraction: float = 0.98
    min_component_pixels: int = 10
    ridge_sigma_px: float = 3.0
    ridge_threshold_sigma: float = 0.75


@dataclass(frozen=True)
class CentreRefinementConfig:
    max_displacement_px: float = 6.0
    search_step_px: float = 1.0
    radial_samples: int = 72
    min_improvement_fraction: float = 0.02


@dataclass(frozen=True)
class RadialConfig:
    meridians: int = 240
    expected_ring_count: int | None = None  # Requires verified device/hardware confirmation.
    require_verified_ring_count_for_classification: bool = True
    max_rings: int = 24  # Safety cap only; never interpreted as hardware ring count.
    min_radius_fraction: float = 0.06
    radial_sample_step: float = 0.75
    peak_smoothing_sigma_px: float = 1.2
    min_peak_prominence: float = 3.0
    peak_prominence_std_fraction: float = 0.18
    min_peak_separation_px: float = 4.0
    max_radial_jump_px: float = 6.0


@dataclass(frozen=True)
class TrackingConfig:
    missing_penalty: float = 1.2
    extra_peak_penalty: float = 0.5
    continuity_weight: float = 0.35
    max_short_gap_meridians: int = 4
    min_direct_coverage: float = 0.55
    min_tracking_confidence: float = 0.50
    min_tracked_rings: int = 3
    max_rejected_fraction: float = 0.02


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 20260821
    decision_threshold: float = 0.55


@dataclass(frozen=True)
class GeometryThresholds:
    # Intentionally empty: no clinical cutoffs invented.
    pass

@dataclass(frozen=True)
class GeometryConfig:
    thresholds: GeometryThresholds | None = None

@dataclass(frozen=True)
class EngineConfig:
    roi: ROIConfig = field(default_factory=ROIConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    centre_refinement: CentreRefinementConfig = field(default_factory=CentreRefinementConfig)
    radial: RadialConfig = field(default_factory=RadialConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    pipeline_version: str = "phase1-0.2.0-polar-tracking"
