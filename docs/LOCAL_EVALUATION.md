# Local confidential-dataset evaluation

KERASCAN never requests, uploads, copies, or logs confidential source images. Keep the manifest and images in a local protected location outside this repository. Use anonymous `patient_id` values only.

Required CSV columns:

```text
patient_id,eye,session_id,image_path,reference_label,site,device_id,operator_id
```

Supported labels are `NORMAL`, `SUSPICIOUS`, `UNGRADABLE`, and `EXCLUDE`. Both eyes and all sessions of a child must remain in one partition.

Audit without inference:

```bash
python -m kerascan.audit_dataset \
  --manifest /private/path/development_manifest.csv \
  --development-manifest /private/path/development_manifest.csv \
  --calibration-manifest /private/path/calibration_manifest.csv \
  --locked-test-manifest /private/path/locked_test_manifest.csv \
  --output local_results/dataset_audit.json
```

The audit detects missing records, duplicate patient/eye/session records, duplicate local image hashes, and patient leakage. It records only redacted image names, hashes, and anonymous IDs.

Locked evaluation requires a frozen, provenance-bearing local model bundle and an explicit acknowledgement:

```bash
python -m kerascan.evaluate_locked \
  --manifest /private/path/locked_test_manifest.csv \
  --model /private/path/models/frozen_model.joblib \
  --output local_results/locked_2026_08 \
  --confirm-locked-evaluation
```

The command generates the aggregate workbook, plots, failure summary, final evaluation PDF, and a read-only evaluation record. It prints no private source paths and stores no raw images. Never use locked-test results for preprocessing, threshold selection, feature selection, model selection, or retuning. A modified model needs a new untouched test set.

## Approved development and calibration only

Only after governance approval, compare local Logistic Regression, calibrated SVM, Random Forest, Extra Trees, optional locally installed gradient boosting, a normal-only anomaly baseline, and the geometry-plus-quality feature model with patient-grouped cross-validation:

```bash
python -m kerascan.train \
  --development-manifest /private/path/development_manifest.csv \
  --calibration-manifest /private/path/calibration_manifest.csv \
  --output-model /private/path/models/candidate_model.joblib \
  --approved-development
```

This command writes `frozen=false`; an approved release process must review, version, and freeze a bundle before locked evaluation. Calibration data are used only for threshold selection. Use realistic acquisition robustness tests (small rotations/translations, exposure change, mild blur/noise/compression, small colour variation, and partial obstruction) only as robustness checks. Never relabel arbitrary synthetic distortions as confirmed disease.

After governance review, freeze the exact candidate without changing its estimator, features, or threshold:

```bash
python -m kerascan.freeze_model \
  --input-model /private/path/models/candidate_model.joblib \
  --output-model /private/path/models/frozen_model.joblib \
  --model-version approved-YYYY-MM-DD \
  --confirm-approved-release
```
