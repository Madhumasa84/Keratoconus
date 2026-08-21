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


@dataclass(frozen=True)
class RadialConfig:
    meridians: int = 240
    max_rings: int = 24
    min_radius_fraction: float = 0.06
    radial_sample_step: float = 0.75
    max_gap_px: float = 3.0


@dataclass(frozen=True)
class TrackingConfig:
    radial_tolerance_px: float = 8.0
    min_component_angles: int = 6


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 20260821
    decision_threshold: float = 0.55


@dataclass(frozen=True)
class EngineConfig:
    roi: ROIConfig = field(default_factory=ROIConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    radial: RadialConfig = field(default_factory=RadialConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline_version: str = "phase1-0.1.0"
