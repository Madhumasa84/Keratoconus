"""Generate a non-confidential example KeraScan screen-positive referral PDF.

The example uses coloured placeholder analysis panels only. It does not call an
image classifier, train a model, access a network, or contain a face/patient
image. Run from the repository root:

    PYTHONPATH=kerascan/src python examples/generate_synthetic_referral.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from PIL import Image, ImageDraw

from app.services.report_service import ReportService, _sha256


ARTIFACT_ROOT = ROOT / "synthetic_referral_artifacts" / "OD"
OUTPUT = ROOT / "synthetic_screen_positive_referral.pdf"
SOURCE_HASH = hashlib.sha256(b"kerascan-synthetic-example-od").hexdigest()


def main() -> str:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    panels = {
        "cropped_roi.png": ("Cropped de-identified ROI", "#2F6690"),
        "cropped_roi_centres.png": ("ROI + refined centre", "#3D9970"),
        "tracked_rings_cartesian.png": ("Tracked-ring overlay", "#7F5AA2"),
        "directional_spacing.png": ("Directional spacing", "#B56576"),
    }
    for filename, (label, colour) in panels.items():
        image = Image.new("RGB", (800, 440), colour)
        draw = ImageDraw.Draw(image)
        draw.text((42, 38), f"SYNTHETIC EXAMPLE — {label}", fill="white")
        if filename != "directional_spacing.png":
            for radius in (55, 95, 135, 175):
                draw.ellipse((400 - radius, 220 - radius, 400 + radius, 220 + radius), outline="white", width=4)
        else:
            points = [(60 + step * 75, 240 + ((step % 3) - 1) * 35) for step in range(9)]
            draw.line(points, fill="white", width=4)
        path = ARTIFACT_ROOT / filename
        image.save(path)
        manifest[filename] = {
            "path": str(path), "sha256": _sha256(path) or "", "eye": "OD",
            "source_image_hash": SOURCE_HASH,
        }
    output_hashes = {name: record["sha256"] for name, record in manifest.items()}
    provenance = hashlib.sha256(json.dumps({
        "eye": "OD", "original_image_hash": SOURCE_HASH,
        "processed_output_hashes": output_hashes,
        "pipeline_version": "synthetic-example-no-model",
        "model_version": "synthetic-example-no-model",
    }, sort_keys=True).encode("utf-8")).hexdigest()
    for record in manifest.values():
        record["provenance_hash"] = provenance

    data = {
        "screening_id": "SYNTHETIC-EXAMPLE-001",
        "screening_date": "2026-08-21T09:30:00Z",
        "operator_id": "SYNTHETIC-OPERATOR",
        "device_id": "SYNTHETIC-DEVICE",
        "protocol_version": "kerascan-school-screening-provisional-1",
        "software_version": "phase4-school-screening",
        "overall_result": "SCREEN_POSITIVE",
        "overall_action": "REFER",
        "referral_priority": "PRIORITY_1",
        "affected_eyes": ["OD"],
        "eyes": [
            {
                "laterality": "OD", "eye_result": "SUSPICIOUS", "image_status": "SUSPICIOUS",
                "image_hash": SOURCE_HASH, "analysis_provenance_hash": provenance,
                "analysis_artifacts": manifest, "quality_gradable": True,
                "quality_metrics": {"ring_tracking_confidence": 0.91},
                "geometry_validation_status": "PASS", "pipeline_version": "synthetic-example-no-model",
                "model_version": "synthetic-example-no-model",
                "reason_codes": ["IMAGE_CLASSIFIER_SUSPICIOUS", "K2_ABOVE_46_8_D"],
                "measurements": [{"k1_d": 43.20, "k2_d": 47.20, "pachymetry_um": 520, "cylinder_d": -0.50}],
                "decisions": [{"final_result": "HIGH_RISK_SCREEN_POSITIVE"}],
            },
            {
                "laterality": "OS", "eye_result": "NORMAL-LIKE", "image_status": "NORMAL_LIKE",
                "quality_gradable": True, "quality_metrics": {"ring_tracking_confidence": 0.92},
                "geometry_validation_status": "PASS", "pipeline_version": "synthetic-example-no-model",
                "model_version": "synthetic-example-no-model", "reason_codes": [],
                "measurements": [{"k1_d": 42.80, "k2_d": 44.10, "pachymetry_um": 530, "cylinder_d": -0.25}],
                "decisions": [{"final_result": "SCREEN_NEGATIVE"}],
            },
        ],
    }
    result = ReportService().generate_pdf(data, str(OUTPUT))
    print(result)
    return result


if __name__ == "__main__":
    main()
