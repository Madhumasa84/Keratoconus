# KERASCAN Phase 1 image engine

Research software for early-screening workflow development. It analyses a full-eye photograph only after automatically extracting a square Placido ROI. It can return `NORMAL-LIKE`, `SUSPICIOUS`, or `UNGRADABLE`; `UNGRADABLE` always means **recapture required** and bypasses the classifier. It never confirms keratoconus.

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

## Design notes

- ROI detection is performed without resizing the full frame and returns original-coordinate `(x0, y0, x1, y1)` bounds. `ROIConfig(manual_center=...)` and `manual_box=...` are fallbacks.
- Quality checks cover resolution, blur, glare/saturation, exposure, contrast, noise, centring, ring coverage, ring evidence, obstruction proxy, small patterns, and non-ring content. Some warnings are non-critical, but missing ring evidence and related failures prevent classification.
- Traditional CLAHE/bilateral/edge segmentation is default. `UNetAdapter` provides the common interface only; it has no bundled weights.
- Radial scans accept 180–360 (or another configured) meridians and configurable ring maxima. Missing points remain `NaN`.
- Reported rPACI-related values are **rPACI-inspired geometric proxies**. They are not the published rPACI implementation and must not be represented as such.
- Arc-Step physical topography is intentionally a placeholder: KERASCAN-specific calibrated camera intrinsics, working distance, Placido pattern geometry, and calibration validation are prerequisites.

## Package interface

```python
from kerascan import EngineConfig, KerascanEngine
engine = KerascanEngine(EngineConfig())
result = engine.analyze('full_eye.png', 'outputs/case_001')
```

`result` includes quality, ROI metadata, tracking statistics, explainable geometric features, and experimental screening metadata. Internal arrays are returned in `_artifacts` for approved research workflows; remove them before JSON serialization.

## Known Phase 1 limitations

- No clinical calibration, diagnostic validation, or physical corneal curvature.
- Classical ROI/segmentation may fail under unrepresented acquisition conditions; manual crop/centre is provided for audit/recovery.
- The quality gate and synthetic prototype thresholds are engineering defaults, not clinical acceptance criteria.
- Graph tracking is a lightweight spatial-consistency implementation, requiring validation against representative de-identified KERASCAN development data under governance.

See [ATTRIBUTION.md](ATTRIBUTION.md) for SmartKC MIT attribution and the modification record.
