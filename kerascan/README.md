# KERASCAN Phase 1 image engine

Research software for early-screening workflow development. It analyses a full-eye photograph only after automatically extracting a square Placido ROI. It can return `NORMAL-LIKE`, `SUSPICIOUS`, or `UNGRADABLE`; it never confirms keratoconus. `UNGRADABLE` always bypasses the classifier, but the message distinguishes an acquisition recapture request from a segmentation, tracking, or configuration failure requiring repeat capture or manual review.

The model score is **experimental—not clinically calibrated**. The bundled Logistic Regression / Random Forest baseline is fit only on deterministic synthetic development proxies. Supplied or locked real images are inference-only and must never be included in `.fit()`.

## Setup and execution

```bash
cd /home/masa84/e1/kerascan
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
pytest -q
python - <<'PY'
from kerascan.inference import KerascanEngine
engine = KerascanEngine()
result = engine.analyze('../sample_images/aleft.png', 'outputs/aleft')
print(result['screening_result'], result['quality'])
PY
jupyter nbconvert --to notebook --execute notebooks/phase1_image_engine.ipynb --output executed_phase1.ipynb
```

`outputs/` contains a separate original full-resolution audit copy, crop, mask, overlays, directional-spacing plot, missing-sector map, feature text, and machine-readable `result.json` for a gradable image. Input files are never modified.

The Stage 1 geometry output also includes `full_stack_analysis.json` and ten
complete-stack audit plots covering every tracked ring identity and every
adjacent pair. See [FULL_STACK_GEOMETRY.md](FULL_STACK_GEOMETRY.md) for the
mathematics, state/completeness semantics, and non-clinical limitations.

## Design notes

- ROI detection is performed without resizing the full frame and returns original-coordinate `(x0, y0, x1, y1)` bounds. `ROIConfig(manual_center=...)` and `manual_box=...` are fallbacks.
- Acquisition quality is independent of segmentation and tracking quality. Its single authoritative score is passed unchanged into valid feature vectors. Ring evidence, peak detectability, mask fragmentation and bridging are reported under segmentation; coverage, order, duplicate-use rejection and direct observations are reported under tracking.
- Traditional CLAHE/bilateral/ridge segmentation is default. The binary ridge mask is a diagnostic artefact only: polar intensity profiles—not connected components—supply ring-centreline candidates. `UNetAdapter` provides the common interface only; it has no bundled weights.
- Radial scans accept 180–360 (or another configured) meridians. Peaks are tracked with ordered dynamic alignment, short-gap interpolation is marked separately, and missing points remain `NaN`. Consequently, negative ring spacing is rejected by construction.
- `RadialConfig.expected_ring_count` is `None` by default. In that state the engine records a **provisional** polar-profile count for engineering review and blocks automated classification. A verified KERASCAN hardware/clinical configuration must provide the count before model-ready geometry is enabled; it must never be inferred or tuned from sample/patient images.
- Reported rPACI-related values are **rPACI-inspired geometric proxies**. They are not the published rPACI implementation and must not be represented as such.
- Arc-Step physical topography is intentionally a placeholder: KERASCAN-specific calibrated camera intrinsics, working distance, Placido pattern geometry, and calibration validation are prerequisites.

## Package interface

```python
from kerascan import EngineConfig, KerascanEngine
engine = KerascanEngine(EngineConfig())
result = engine.analyze('full_eye.png', 'outputs/case_001')
```

`result` includes `acquisition_quality`, `segmentation`, `tracking`, `analysis_status`, an exact `failure_stage`, ROI metadata, and (only when invariants and verified device configuration pass) model-ready geometric features. Internal arrays are returned in `_artifacts` for approved research workflows; remove them before JSON serialization.

## Known Phase 1 limitations

- No clinical calibration, diagnostic validation, or physical corneal curvature.
- Classical ROI/segmentation may fail under unrepresented acquisition conditions; manual crop/centre is provided for audit/recovery. A segmentation or tracking failure is not automatically an acquisition-quality failure.
- The KERASCAN hardware ring count remains a required external confirmation. Until then, provisional geometry is for engineering review only and the experimental classifier is intentionally blocked.
- The quality gate and synthetic prototype thresholds are engineering defaults, not clinical acceptance criteria.
- Graph tracking is a lightweight spatial-consistency implementation, requiring validation against representative de-identified KERASCAN development data under governance.

See [ATTRIBUTION.md](ATTRIBUTION.md) for SmartKC MIT attribution and the modification record.
