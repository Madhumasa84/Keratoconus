"""Privacy-preserving local batch verification for supplied KeraScan images.

The report contains hash-based case IDs and engineering gate results only. It
does not copy source images, store source paths or filenames, classify blocked
images, or make a clinical claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from kerascan.image_io import SUPPORTED_SUFFIXES
from kerascan.inference import KerascanEngine


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round_vector(values: list[Any], digits: int = 6) -> list[float | None]:
    return [None if _finite(value) is None else round(float(value), digits) for value in values]


def _case_summary(result: dict[str, Any], case_id: str) -> dict[str, Any]:
    quality = result.get("acquisition_quality", {})
    segmentation = result.get("segmentation", {})
    tracking = result.get("tracking", {})
    stack = result.get("full_stack_analysis") or {}
    full_summary = stack.get("full_stack_summary") or {}
    radial = [value for value in stack.get("radial_stack_deviation_by_meridian", []) if value is not None]
    return {
        "case_id": case_id,
        "screening_result": result.get("screening_result"),
        "analysis_status": result.get("analysis_status"),
        "failure_stage": result.get("failure_stage"),
        "classification_performed": bool(result.get("classification_performed", False)),
        "acquisition_quality": quality.get("status"),
        "acquisition_quality_score": _finite(quality.get("score")),
        "acquisition_flags": sorted(quality.get("flags") or []),
        "roi_method": (result.get("roi") or {}).get("method"),
        "segmentation_status": segmentation.get("status"),
        "segmentation_flags": sorted(segmentation.get("flags") or []),
        "tracking_status": tracking.get("status"),
        "tracking_flags": sorted(tracking.get("flags") or []),
        "tracking_confidence": _finite(tracking.get("confidence")),
        "hardware_gate": (result.get("gates") or {}).get("verified_hardware_ring_count", "NOT_REACHED"),
        "threshold_gate": (result.get("gates") or {}).get("approved_geometry_thresholds", "NOT_REACHED"),
        "geometry_status": result.get("geometry_status", "NOT_REACHED"),
        "expected_ring_count": stack.get("expected_ring_count"),
        "detected_ring_count": stack.get("detected_ring_count"),
        "ring_count_verified": bool(stack.get("ring_count_verified", False)),
        "analysed_ring_count": len(stack.get("analysed_ring_indices") or []),
        "analysed_adjacent_pair_count": int(stack.get("analysed_ring_pair_count") or 0),
        "complete_stack_available": bool(stack.get("complete_stack_available", False)),
        "completeness_reason_codes": sorted(stack.get("completeness_reason_codes") or []),
        "ring_direct_coverage": _round_vector(
            [item.get("observed_fraction") for item in stack.get("ring_completeness", [])]
        ),
        "pair_direct_coverage": _round_vector(
            [item.get("observed_fraction") for item in stack.get("ring_pair_completeness", [])]
        ),
        "angular_variation_by_pair": _round_vector(stack.get("angular_spacing_variation_by_pair", [])),
        "radial_stack_deviation_median": _finite(median(radial)) if radial else None,
        "radial_stack_deviation_maximum": _finite(full_summary.get("maximum_radial_stack_deviation")),
        "maximum_compressed_pair_run": int(full_summary.get("maximum_compressed_pair_run") or 0),
        "maximum_expanded_pair_run": int(full_summary.get("maximum_expanded_pair_run") or 0),
        "maximum_normalized_cumulative_residual_magnitude": _finite(
            full_summary.get("maximum_normalized_cumulative_residual_magnitude")
        ),
        "compression_sector_count": len(stack.get("compression_sectors") or []),
        "expansion_sector_count": len(stack.get("expansion_sectors") or []),
    }


def verify_folder(input_dir: Path, output_file: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve(strict=True)
    if not input_dir.is_dir():
        raise ValueError("input must be a directory")
    all_files = sorted(path for path in input_dir.rglob("*") if path.is_file() and not path.is_symlink())
    supported = [path for path in all_files if path.suffix.lower() in SUPPORTED_SUFFIXES]
    engine = KerascanEngine()
    cases: list[dict[str, Any]] = []
    processing_errors = 0
    for index, path in enumerate(supported, start=1):
        case_id = _file_hash(path)[:16]
        try:
            cases.append(_case_summary(engine.analyze(path), case_id))
        except Exception:
            processing_errors += 1
            cases.append(
                {
                    "case_id": case_id,
                    "screening_result": "UNGRADABLE",
                    "analysis_status": "UNGRADABLE",
                    "failure_stage": "DECODE_OR_PIPELINE",
                    "classification_performed": False,
                }
            )
        print(f"processed {index}/{len(supported)}", flush=True)

    report = {
        "report_type": "local_privacy_preserving_full_stack_batch_verification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": engine.config.pipeline_version,
        "supported_file_count": len(supported),
        "unsupported_file_count": len(all_files) - len(supported),
        "unique_content_hash_count": len({case["case_id"] for case in cases}),
        "processing_error_count": processing_errors,
        "screening_result_counts": dict(sorted(Counter(case.get("screening_result") for case in cases).items())),
        "failure_stage_counts": dict(sorted(Counter(case.get("failure_stage") for case in cases).items())),
        "acquisition_quality_counts": dict(sorted(Counter(case.get("acquisition_quality", "NOT_REACHED") for case in cases).items())),
        "segmentation_status_counts": dict(sorted(Counter(case.get("segmentation_status", "NOT_REACHED") for case in cases).items())),
        "tracking_status_counts": dict(sorted(Counter(case.get("tracking_status", "NOT_REACHED") for case in cases).items())),
        "geometry_status_counts": dict(sorted(Counter(case.get("geometry_status", "NOT_REACHED") for case in cases).items())),
        "classification_performed_count": sum(bool(case.get("classification_performed")) for case in cases),
        "privacy": {
            "source_filenames_included": False,
            "source_paths_included": False,
            "source_images_copied": False,
            "derived_images_written": False,
            "network_used": False,
        },
        "limitations": [
            "Experimental image-space engineering assessment only.",
            "The report does not diagnose or exclude keratoconus.",
            "Folder membership is not treated as a clinical reference label.",
            "A provisional detected ring count is not a verified hardware ring count.",
        ],
        "cases": cases,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = verify_folder(args.input, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
