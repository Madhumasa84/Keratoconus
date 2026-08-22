# KERASCAN — Phases 1–3

KERASCAN is an offline portable initial screening aid for school children. It analyses Placido images, applies a transparent referral protocol, and produces local screening/referral reports. It can output `SCREEN_NEGATIVE`, `STANDARD_REFERRAL`, `PRIORITY_REFERRAL`, `RECAPTURE_REQUIRED`, `INCOMPLETE`, or `MANUAL_REVIEW`.

> Suspicious screening result—further corneal evaluation is recommended.

KERASCAN does not diagnose keratoconus, does not replace tomography or qualified clinical assessment, and does not collect follow-up outcomes. The workflow ends after screening, local report generation, and a referral recommendation.

## Offline installation and start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links /path/to/wheelhouse -r app/requirements.txt
python -m app.manage migrate
python -m app.manage create-user --operator-id field_admin --role administrator --password 'local-only-long-passphrase'
streamlit run app/streamlit_app.py
```

For development with an approved package index, omit `--no-index --find-links ...`.

## Tests

```bash
PYTHONPATH=kerascan/src pytest -q kerascan/tests app/tests
```

## Local confidential evaluation

The repository has no confidential dataset, credential, or patient image. Place manifests and images outside the repository and run:

```bash
python -m kerascan.audit_dataset --manifest /private/path/manifest.csv --output local_results/dataset_audit.json
python -m kerascan.evaluate_locked --manifest /private/path/locked_test_manifest.csv --model /private/path/models/frozen_model.joblib --output local_results/locked --confirm-locked-evaluation
```

See [local evaluation instructions](docs/LOCAL_EVALUATION.md), [field deployment](docs/FIELD_DEPLOYMENT.md), [model card](docs/MODEL_CARD.md), [failure-mode register](docs/FAILURE_MODE_REGISTER.md), and [SmartKC attribution](kerascan/ATTRIBUTION.md).

The software has no telemetry, cloud dependency, remote logging, or unauthenticated operator access. Do not use a locked-test evaluation to retune a model; a changed model requires a new untouched test set.

## Initial image policy

The initial school-screening workflow requires good-quality KeraScan images for
both OD and OS, plus K1, K2, one protocol-defined pachymetry value, and
cylinder for each eye. A missing, rejected, failed, or hardware-blocked image
cannot yield a completed screen-negative result. Detailed local referral PDFs
are generated only for final screen-positive `REFER` outcomes.

Blur-versus-good-image comparison, synthetic blur robustness experiments, and
clinical quality-threshold determination are deferred; see
[future blur-evaluation work](docs/FUTURE_BLUR_EVALUATION.md). The present
workflow does not sharpen or otherwise repair clinically unusable images.
