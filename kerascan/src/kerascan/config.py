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
    # When False, acquisition quality is measured and reported but never blocks
    # the pipeline. Intended for workflows where a clinician has already
    # visually selected the capture before upload.
    enforce_gate: bool = True
    # Minimum bright-to-dark swing across the ring band for the located region
    # to be accepted as an actual Placido mire pattern. This is NOT a quality
    # judgement and is enforced even when enforce_gate is False: below it the
    # analysis is measuring skin, eyelashes or eyebrow rather than a cornea,
    # and a result computed from those is meaningless rather than merely poor.
    # Calibrated on this project's sample photos, where confirmed skin-locked
    # detections scored 40-48 and every genuine cornea scored 80 or above.
    min_mire_contrast_amplitude: float = 65.0
    min_roi_side_px: int = 180
    # Rejection threshold: Laplacian variance below this level is REJECTED
    min_laplacian_variance: float = 20.0
    # Warning fraction: Laplacian between (min * warning_fraction) and min is WARNING
    # Laplacian below (min * warning_fraction) is REJECTED
    warning_laplacian_fraction: float = 0.40
    min_mean_intensity: float = 24.0
    min_contrast: float = 18.0
    # Warning fraction: contrast between (min * warning_fraction) and min is WARNING
    warning_contrast_fraction: float = 0.55
    max_saturation_fraction: float = 0.12
    max_noise_sigma: float = 28.0
    min_centring_ratio: float = 0.18
    min_angular_coverage: float = 0.42
    min_ring_fraction: float = 0.008
    # Minimum downstream tracking confidence for ACCEPTABLE_WITH_WARNING
    min_warning_tracking_confidence: float = 0.50
    # Minimum direct observation coverage for ACCEPTABLE_WITH_WARNING
    min_warning_direct_coverage: float = 0.55


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
    # When True, the number of rings actually reconstructed from this capture is
    # used as the analysis basis instead of a pre-configured hardware count. The
    # assessment is then self-consistent for the rings genuinely observed; it is
    # explicitly NOT a verified hardware configuration and is reported as such.
    accept_detected_ring_count: bool = False
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
    # No clinical cutoffs are defined here. A caller may supply bounds (see
    # app/services/screening_service.py for the app-layer provisional,
    # non-clinical, distribution-derived thresholds it opts into); the engine
    # itself never invents or defaults to any.
    version: str = "none"
    suspicious_bounds: dict = field(default_factory=dict)
    indeterminate_bounds: dict = field(default_factory=dict)

@dataclass(frozen=True)
class GeometryConfig:
    # Engineering persistence and completeness rules only. These are not
    # clinical disease thresholds and cannot produce a calibrated conclusion.
    compression_magnitude_fraction: float = 0.05
    expansion_magnitude_fraction: float = 0.05
    min_coherent_pair_run: int = 2
    min_sector_angular_samples: int = 3
    max_missing_sector_fraction: float = 0.20
    # Adjacent ring pairs must carry enough direct observation to be measured.
    # Pairs below that are excluded from the analysis rather than failing the
    # whole capture; this is how many must survive for a stack to be reportable.
    min_analysable_ring_pairs: int = 4
    # Engineering-only persistence controls for self-fitted reference
    # residuals. They are not clinical disease thresholds.
    reference_deviation_magnitude_fraction: float = 0.08
    min_coherent_ring_run: int = 2
    min_reference_sector_angular_samples: int = 3
    thresholds: GeometryThresholds | None = None

@dataclass(frozen=True)
class EngineConfig:
    hardware_version: str = "unknown"
    ring_count_source: str = "unknown"
    roi: ROIConfig = field(default_factory=ROIConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    centre_refinement: CentreRefinementConfig = field(default_factory=CentreRefinementConfig)
    radial: RadialConfig = field(default_factory=RadialConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    pipeline_version: str = "phase1-0.4.0-reference-geometry"
