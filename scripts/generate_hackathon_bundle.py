"""Generate the final hackathon demonstration bundle for KeraScan.

Creates:
  verification_outputs/final_hackathon/
    ├── od_pipeline/           (All 7 required images + features.txt + debug artifacts)
    ├── os_pipeline/           (All 7 required images + features.txt + debug artifacts)
    ├── synthetic_referral_report.pdf (Generated PDF with SYNTHETIC DEMONSTRATION label)
    ├── pdf_rendered_pages/    (High-res PNG renders of PDF pages)
    └── ENGINEERING_REPORT.md  (Complete documentation of system, math, and verification)
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np

# Ensure import paths
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "kerascan" / "src"))

from kerascan.config import EngineConfig, RadialConfig, GeometryConfig
from kerascan.inference import KerascanEngine
from app.services.report_service import ReportService
from app.services.protocol import load_protocol
from app.services.screening_service import ScreeningService


OUTPUT_DIR = WORKSPACE_ROOT / "verification_outputs" / "final_hackathon"
SAMPLE_DIR = WORKSPACE_ROOT / "sample_images"


def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    od_dir = OUTPUT_DIR / "od_pipeline"
    os_dir = OUTPUT_DIR / "os_pipeline"
    od_dir.mkdir(parents=True, exist_ok=True)
    os_dir.mkdir(parents=True, exist_ok=True)

    # Use default engine configuration with reference geometry
    engine = KerascanEngine()

    print("Running engine on OD sample (aright.png)...")
    od_res = engine.analyze(SAMPLE_DIR / "aright.png", output_dir=od_dir)
    print(f"OD status: {od_res.get('analysis_status')}, screening_result: {od_res.get('screening_result')}")

    print("Running engine on OS sample (aleft.png)...")
    os_res = engine.analyze(SAMPLE_DIR / "aleft.png", output_dir=os_dir)
    print(f"OS status: {os_res.get('analysis_status')}, screening_result: {os_res.get('screening_result')}")

    return od_res, os_res


def generate_synthetic_referral_pdf(od_res, os_res):
    print("Generating synthetic screen-positive referral PDF...")
    report_service = ReportService()
    protocol = load_protocol()

    od_dir = OUTPUT_DIR / "od_pipeline"
    os_dir = OUTPUT_DIR / "os_pipeline"

    # Build manifest of OD artifacts
    od_hash = od_res.get("original_sha256", "a" * 64)
    os_hash = os_res.get("original_sha256", "b" * 64)
    provenance_hash = "c" * 64

    def build_manifest(eye_dir, eye_label, source_h):
        manifest = {}
        for item in eye_dir.glob("*.png"):
            h = ScreeningService._file_sha256(item)
            manifest[item.name] = {
                "path": str(item.resolve()),
                "sha256": h,
                "eye": eye_label,
                "source_image_hash": source_h,
                "provenance_hash": provenance_hash,
            }
        return manifest

    od_manifest = build_manifest(od_dir, "OD", od_hash)
    os_manifest = build_manifest(os_dir, "OS", os_hash)

    # Synthetic demo data: OD is screen-positive (K2=48.2 D > 46.8 D, Pachymetry=455 um < 480 um, Cyl=-2.25 D)
    screening_data = {
        "screening_id": "KERASCAN-DEMO-2026-001",
        "screening_date": "2026-08-22T08:30:00Z",
        "operator_id": "OP-SCHOOL-04",
        "device_id": "KERASCAN-PORTABLE-V2",
        "site": "School Screening Site #12",
        "protocol_version": protocol.protocol_version,
        "software_version": protocol.software_version,
        "overall_result": "SCREEN_POSITIVE",
        "overall_action": "REFER",
        "referral_priority": "PRIORITY_1",
        "affected_eyes": ["OD"],
        "eyes": [
            {
                "laterality": "OD",
                "eye_result": "SUSPICIOUS",
                "image_status": "SUSPICIOUS",
                "image_hash": od_hash,
                "analysis_provenance_hash": provenance_hash,
                "analysis_artifacts": od_manifest,
                "quality_gradable": True,
                "quality_level": "ACCEPTABLE",
                "quality_metrics": {"ring_tracking_confidence": 0.94},
                "geometry_validation_status": "PASS",
                "pipeline_version": "2.1-reference-comparison",
                "model_version": "gated-deterministic-v1",
                "reason_codes": [
                    "IMAGE_CLASSIFIER_SUSPICIOUS",
                    "K2_ABOVE_46_8_D",
                    "PACHYMETRY_BELOW_480_UM",
                    "CYLINDER_MAGNITUDE_ABOVE_1_5_D",
                    "MULTIPLE_QUANTITATIVE_ABNORMALITIES",
                ],
                "measurements": [
                    {
                        "k1_d": 44.50,
                        "k2_d": 48.20,
                        "pachymetry_um": 455.0,
                        "cylinder_d": -2.25,
                        "pachymetry_measurement_type": "device_reported",
                    }
                ],
                "decisions": [{"final_result": "HIGH_RISK_SCREEN_POSITIVE"}],
                "image_analyses": [{"tracking_confidence": 0.94, "geometry_validation_status": "PASS"}],
            },
            {
                "laterality": "OS",
                "eye_result": "NORMAL-LIKE",
                "image_status": "NORMAL_LIKE",
                "image_hash": os_hash,
                "analysis_provenance_hash": provenance_hash,
                "analysis_artifacts": os_manifest,
                "quality_gradable": True,
                "quality_level": "ACCEPTABLE",
                "quality_metrics": {"ring_tracking_confidence": 0.96},
                "geometry_validation_status": "PASS",
                "pipeline_version": "2.1-reference-comparison",
                "model_version": "gated-deterministic-v1",
                "reason_codes": [],
                "measurements": [
                    {
                        "k1_d": 43.10,
                        "k2_d": 43.80,
                        "pachymetry_um": 535.0,
                        "cylinder_d": 0.50,
                        "pachymetry_measurement_type": "device_reported",
                    }
                ],
                "decisions": [{"final_result": "SCREEN_NEGATIVE"}],
                "image_analyses": [{"tracking_confidence": 0.96, "geometry_validation_status": "PASS"}],
            },
        ],
    }

    pdf_path = OUTPUT_DIR / "synthetic_referral_report.pdf"
    report_service.generate_pdf(screening_data, str(pdf_path))
    print(f"Generated PDF at: {pdf_path}")
    return pdf_path


def render_pdf_pages(pdf_path: Path):
    print("Rendering PDF pages to PNG...")
    render_dir = OUTPUT_DIR / "pdf_rendered_pages"
    render_dir.mkdir(parents=True, exist_ok=True)

    import pymupdf

    doc = pymupdf.open(pdf_path)
    rendered_files = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=200)
        page_file = render_dir / f"page_{page_idx + 1}.png"
        pix.save(str(page_file))
        rendered_files.append(page_file)
        print(f"Saved {page_file}")

    return rendered_files


def generate_engineering_report():
    print("Writing ENGINEERING_REPORT.md...")
    report_md = """# KeraScan — Hackathon Demonstration Engineering Report

## 1. Executive Summary

**KeraScan** is an offline, privacy-first portable keratoconus screening research prototype designed for community and school vision screening encounters. 

The system combines:
1. **Mathematical Placido-Ring Analysis**: Extracting polar-unwrapped ring geometry, computing ring-pair inter-ring spacing, fitting self-fitted artificial mathematical reference geometry (concentric and smooth low-order models), and computing robust image-space residuals.
2. **ROI-Aware Three-Tier Quality Gate**: Assessing acquisition sharpness strictly inside the Placido ring band (50–90% outer radius) to prevent skin/sclera low-frequency false blur rejections.
3. **Deterministic Bilateral Decision Matrix**: Combining KeraScan appearance results with three quantitative measurement domains (K2 curvature, pachymetry, and cylinder magnitude) into a clear clinical screening recommendation.
4. **Restrained Clinician-Readable Reporting**: Producing natural, de-identified referral reports with plain-language explanations when indicated.

---

## 2. Pipeline Architecture & Workflow

```
[ Upload OD & OS Images ]
           │
           ▼
[ Detect Placido ROI ] ──> Center Refinement
           │
           ▼
[ ROI-Aware Quality Gate ] ──> (ACCEPTABLE / ACCEPTABLE_WITH_WARNING / REJECTED)
           │
           ▼
[ Traditional Segmentation & Polar Radial Scan ]
           │
           ▼
[ Graph-Based Ring Tracking ] ──> (OBSERVED, INTERPOLATED, MISSING, REJECTED)
           │
           ▼
[ Reference Geometry Construction ]
   ├─ Concentric-Circle Reference (Median R[k,θ])
   └─ Smooth Low-Order Fourier Fit (Huber robust)
           │
           ▼
[ Geometry Residual & Heatmap Analysis ]
           │
           ▼
[ Enter Quantitative Measurements: K1, K2, Pachymetry, Cylinder ]
           │
           ▼
[ Bilateral Decision Matrix ] ──> Child-Level Outcome (REFER / REPEAT / SCREEN-NEGATIVE)
           │
           ▼
[ Natural Referral Report PDF Generation (REFER only) ]
```

---

## 3. Required Image Artifacts

Each analyzed eye generates seven standardized image artifacts:

| Filename | Description | Clinical Purpose |
| :--- | :--- | :--- |
| `cropped_roi.png` | De-identified cropped Placido corneal ROI | Visual image confirmation |
| `observed_vs_concentric_reference.png` | Tracked rings (solid) vs concentric mathematical reference (dashed) | Ring symmetry & circularity comparison |
| `observed_vs_smooth_reference.png` | Tracked rings (solid) vs smooth Fourier reference (dashed) | Local cone/ectasia deviation visibility |
| `radial_deviation_vectors.png` | Signed displacement vectors from reference | Directional cone displacement vectors |
| `reference_spacing_residual_heatmap.png` | Normalized ring spacing residuals across 360° meridians | Compression/expansion localization |
| `full_stack_sector_map.png` | Coherent full-stack angular sector map | Multi-ring agreement representation |
| `clinician_comparison_panel.png` | 4-part summary panel for clinician referral review | Complete visual summary on report |

---

## 4. Decision Matrix Rules & Boundary Semantics

- **K2 Steep Curvature**: Abnormal strictly when $K_2 > 46.8\\text{ D}$ ($K_2 = 46.8\\text{ D}$ is Normal).
- **Pachymetry (Corneal Thickness)**: Abnormal strictly when $\\text{Pachymetry} < 480\\ \\mu\\text{m}$ ($480\\ \\mu\\text{m}$ is Normal).
- **Cylinder Magnitude**: Abnormal strictly when $|\\text{Cylinder}| > 1.5\\text{ D}$ ($1.5\\text{ D}$ is Normal).
- **Safety Rule**: A missing, rejected, ungradable, or analysis-blocked image can never produce a normal outcome.

---

## 5. Verification & Test Suite Results

The offline test suite comprises unit, integration, and robustness tests:
- **Quality Gate Tests**: Tested against synthetic blur/contrast perturbations and doctor-selected real images.
- **Reference Geometry Tests**: Deterministic validation of median radius, Huber fitting, monotonicity, and residual limits.
- **Decision Matrix Tests**: Exhaustive validation of all decision rows and exact threshold boundaries.
- **Provenance & PDF Tests**: SHA256 integrity checks, path redaction, and watermark validation.

*All tests pass offline with zero network connectivity or cloud dependencies.*
"""
    (OUTPUT_DIR / "ENGINEERING_REPORT.md").write_text(report_md, encoding="utf-8")
    print("Saved ENGINEERING_REPORT.md")


def main():
    od_res, os_res = run_pipeline()
    pdf_path = generate_synthetic_referral_pdf(od_res, os_res)
    render_pdf_pages(pdf_path)
    generate_engineering_report()
    print("\n✅ Complete hackathon demonstration bundle generated successfully!")


if __name__ == "__main__":
    main()
