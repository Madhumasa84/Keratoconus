# Attribution and modification record

KERASCAN Phase 1 draws limited implementation inspiration from Microsoft SmartKC, specifically its public discussion/code patterns for contrast/edge-based mire segmentation, ray/radial mire localisation, and spatial graph-based consistency correction.

Source: [microsoft/SmartKC-A-Smartphone-based-Corneal-Topographer](https://github.com/microsoft/SmartKC-A-Smartphone-based-Corneal-Topographer), `LICENSE-CODE` (MIT license). SmartKC copyright and MIT attribution are retained; see its repository and `LICENSE-CODE` for the complete license text.

KERASCAN modifications and non-reuse boundaries:

- This is an independent, modular rewrite, not a copied SmartKC pipeline.
- It accepts PNG/JPEG/TIFF/RGBA images and performs full-resolution, full-eye ROI detection before any ring analysis.
- It removes SmartKC-specific assumptions including 3000×4000 inputs, 500×500 crops, `(250, 250)` centres, 22/20 mire counts, zoom undoing, fixed working distance, Android workflow, pretrained weights, and camera geometry.
- It uses the estimated centre in every geometric operation and preserves missing radial points as `NaN`.
- Its traditional segmentation is default; a U-Net is only an externally supplied adapter, with no claim that SmartKC weights generalise to KERASCAN.
- It contains no Arc-Step calculation, physical curvature claim, clinical labels, or SmartKC clinical-label reuse.

No SmartKC code or weights were copied verbatim into KERASCAN. This document is maintained as the modification record required for downstream research governance.
