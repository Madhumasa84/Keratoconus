"""Phase-1 orchestration; invalid geometry never reaches the classifier."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np

from .centre_refinement import refine_centre
from .config import EngineConfig
from .geometry import compute_geometry, validate_geometry
from .graph_tracking import track_rings
from .image_io import image_sha256, read_image, save_png
from .quality import evaluate_quality
from .radial_scan import radial_scan
from .roi_detection import detect_placido_roi
from .segmentation import TraditionalSegmenter

EXPERIMENTAL = "experimental—not clinically calibrated"


class KerascanEngine:
    def __init__(self, config: EngineConfig = EngineConfig()):
        self.config = config


    @staticmethod
    def _failure_message(stage: str, flags: list[str]) -> str:
        detail = ", ".join(flags) if flags else "unspecified quality failure"
        if stage == "ACQUISITION":
            return f"Recapture required: {detail}"
        if stage == "SEGMENTATION":
            return "Automated ring segmentation failed; repeat capture or manual review"
        if stage == "TRACKING":
            return "Reliable ring identities could not be reconstructed; repeat capture or manual review"
        if stage == "CONFIGURATION":
            return "Verified KeraScan ring-count configuration is required before automated screening; geometry retained for engineering review"
        return "Unsupported or non-KeraScan image"

    @staticmethod
    def _serialise_result(result: dict, output_dir: Path) -> None:
        serial = {key: value for key, value in result.items() if key != "_artifacts"}
        (output_dir / "result.json").write_text(json.dumps(serial, indent=2, default=float))

    def _complete(self, result: dict, output_dir: Path | None) -> dict:
        if output_dir is not None:
            from .visualization import save_visualizations

            save_visualizations(result, output_dir)
            self._serialise_result(result, output_dir)
        return result

    def analyze(self, source: str | Path | np.ndarray, output_dir: str | Path | None = None) -> dict:
        image = read_image(source) if isinstance(source, (str, Path)) else source.copy()
        output = Path(output_dir) if output_dir else None
        roi = detect_placido_roi(image, self.config.roi)
        refinement = refine_centre(roi.crop, roi.center_roi, roi.outer_radius_px, self.config.centre_refinement)
        refined_center = refinement.refined_center
        acquisition = evaluate_quality(roi.crop, refined_center, roi.outer_radius_px, self.config.quality)
        source_pattern_fraction = roi.outer_radius_px / max(min(image.shape[:2]) / 2, 1)
        acquisition["metrics"]["source_pattern_radius_fraction"] = float(source_pattern_fraction)
        if source_pattern_fraction < 0.15:
            acquisition["flags"] = sorted(set(acquisition["flags"] + ["placido_pattern_too_small"]))
            acquisition["gradable"] = False
            acquisition["status"] = "FAIL"
            acquisition["quality_score"] = max(0, acquisition["quality_score"] - 12)
        # ``quality`` is a backwards-compatible alias.  Its score is always the
        # authoritative acquisition score passed to feature extraction.
        quality = {
            "gradable": acquisition["gradable"],
            "quality_score": acquisition["quality_score"],
            "flags": list(acquisition["flags"]),
            "metrics": dict(acquisition["metrics"]),
            "status": acquisition["status"],
        }
        result: dict = {
            "pipeline_version": self.config.pipeline_version,
            "original_shape": list(image.shape),
            "original_sha256": image_sha256(image),
            "roi": {
                "box_xyxy": roi.box,
                "center_full": roi.center_full,
                "center_roi": refined_center,
                "initial_center_roi": roi.center_roi,
                "outer_radius_px": roi.outer_radius_px,
                "confidence": roi.confidence,
                "method": roi.method,
            },
            "centre_refinement": {
                "initial_center": refinement.initial_center,
                "refined_center": refinement.refined_center,
                "displacement_px": refinement.displacement_px,
                "confidence": refinement.confidence,
                "method": refinement.method,
                "objective_initial": refinement.objective_initial,
                "objective_refined": refinement.objective_refined,
            },
            "quality": quality,
            "acquisition_quality": {
                "status": acquisition["status"],
                "score": acquisition["quality_score"],
                "flags": list(acquisition["flags"]),
                "metrics": dict(acquisition["metrics"]),
            },
            "segmentation": {"status": "NOT_RUN", "confidence": 0.0, "flags": [], "metrics": {}},
            "tracking": {"status": "NOT_RUN", "confidence": 0.0, "flags": []},
            "analysis_status": "UNGRADABLE",
            "failure_stage": "NONE",
            "classification_performed": False,
            "classification_skipped": True,
            "experimental_status": EXPERIMENTAL,
        }
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            save_png(output / "original_full_resolution.png", image)
            save_png(output / "cropped_roi.png", roi.crop)
            from .visualization import save_roi_overlay

            save_roi_overlay(image, roi.box, output)

        artifacts = {
            "roi_crop": roi.crop,
            "center": refined_center,
            "initial_center": roi.center_roi,
            "radii": np.empty((0, 0), dtype=float),
            "angles": np.empty(0, dtype=float),
        }
        if not acquisition["gradable"]:
            result.update(
                {
                    "screening_result": "UNGRADABLE",
                    "message": self._failure_message("ACQUISITION", acquisition["flags"]),
                    "failure_stage": "ACQUISITION",
                    "_artifacts": artifacts,
                }
            )
            return self._complete(result, output)

        segmenter = TraditionalSegmenter(self.config.segmentation)
        seg = segmenter.segment(roi.crop, refined_center, roi.outer_radius_px)
        if output is not None:
            save_png(output / "ring_mask.png", seg.mask)
        scan = radial_scan(seg.enhanced, refined_center, roi.outer_radius_px, self.config.radial)
        # Peak detectability is the segmentation outcome that matters to polar
        # tracking.  Connected-component size is recorded, but not used as an
        # identity proxy or blanket failure condition.
        peak_count = int(len(scan.reference_radii))
        segmentation_confidence = float(
            np.clip(
                scan.peak_coverage * min(1.0, peak_count / max(self.config.tracking.min_tracked_rings, 1))
                * (0.5 + 0.5 * seg.confidence),
                0.0,
                1.0,
            )
        )
        seg_flags: list[str] = []
        if peak_count < self.config.tracking.min_tracked_rings:
            seg_flags.append("insufficient_polar_ring_peaks")
        if self.config.radial.expected_ring_count is not None and peak_count != self.config.radial.expected_ring_count:
            seg_flags.append("configured_ring_count_not_reconstructed")
        if scan.peak_coverage < self.config.quality.min_angular_coverage:
            seg_flags.append("insufficient_peak_sector_coverage")
        result["segmentation"] = {
            "status": "PASS" if not seg_flags else "FAIL",
            "confidence": segmentation_confidence,
            "flags": seg_flags,
            "method": seg.method,
            "metrics": {
                **seg.metrics,
                "polar_peak_coverage": scan.peak_coverage,
                "visible_ring_sector_coverage": scan.peak_coverage,
                "provisional_ring_count": peak_count,
                "ring_count_source": scan.ring_count_source,
            },
        }
        artifacts.update({"mask": seg.mask, "scan": scan})
        if seg_flags:
            result.update(
                {
                    "screening_result": "UNGRADABLE",
                    "message": self._failure_message("SEGMENTATION", seg_flags),
                    "failure_stage": "SEGMENTATION",
                    "_artifacts": artifacts,
                }
            )
            return self._complete(result, output)

        tracked = track_rings(scan, self.config.tracking, self.config.radial)
        geometry = validate_geometry(
            tracked.radii,
            tracked.observed,
            self.config.tracking.min_direct_coverage,
        )
        tracking_flags = sorted(set(tracked.flags + geometry.flags))
        result["tracking"] = {
            "status": "PASS" if not tracking_flags else "FAIL",
            "confidence": tracked.confidence,
            "flags": tracking_flags,
            "missing_point_fraction": tracked.missing_fraction,
            "direct_observation_fraction": tracked.direct_observation_fraction,
            "duplicate_removals": tracked.duplicate_removals,
            "unassigned_extra_peak_count": tracked.unassigned_extra_peak_count,
            "identity_shift_fraction": tracked.identity_shift_fraction,
            "order_change_fraction": tracked.order_change_fraction,
            "ring_completeness": tracked.ring_completeness.tolist(),
        }
        quality["metrics"].update(
            {
                "missing_ring_fraction": tracked.missing_fraction,
                "ring_order_change_fraction": tracked.order_change_fraction,
                "ring_tracking_confidence": tracked.confidence,
            }
        )
        artifacts.update(
            {
                "radii": tracked.radii,
                "angles": scan.angles_deg,
                "tracking_result": tracked,
                "segmentation_confidence": segmentation_confidence,
            }
        )
        if tracking_flags:
            result.update(
                {
                    "screening_result": "UNGRADABLE",
                    "message": self._failure_message("TRACKING", tracking_flags),
                    "failure_stage": "TRACKING",
                    "_artifacts": artifacts,
                }
            )
            return self._complete(result, output)

        # A provisional count is suitable for engineering plots and regression
        # review, not for a model vector whose meaning depends on ring identity.
        # It must be supplied by a verified device/hardware configuration rather
        # than inferred from these (or any) patient/sample images.
        if self.config.radial.require_verified_ring_count_for_classification and scan.ring_count_source != "verified_device_config":
            result.update(
                {
                    "screening_result": "UNGRADABLE",
                    "message": self._failure_message("CONFIGURATION", []),
                    "failure_stage": "CONFIGURATION",
                    "_artifacts": artifacts,
                }
            )
            return self._complete(result, output)

        geom_output = compute_geometry(
            tracked.radii,
            scan.angles_deg,
            tracked.observed,
            self.config.tracking.min_direct_coverage,
            self.config.geometry
        )
        
        quality["metrics"]["detected_ring_count"] = int(np.sum(np.mean(np.isfinite(tracked.radii), axis=1) > 0.2))

        result.update(
            {
                "screening_result": geom_output["geometry_status"],
                "analysis_status": "GRADABLE",
                "failure_stage": "NONE",
                "classification_performed": False,
                "classification_skipped": True,
                "geometry_method": geom_output["geometry_method"],
                "geometry_status": geom_output["geometry_status"],
                "geometry_confidence": geom_output["geometry_confidence"],
                "gates": geom_output["gates"],
                "reason_codes": geom_output["reason_codes"],
                "features": geom_output["features"],
                "_artifacts": artifacts,
            }
        )
        # Store grids for visualization in artifacts
        artifacts["geometry_grids"] = geom_output["_grids"]
        
        return self._complete(result, output)
