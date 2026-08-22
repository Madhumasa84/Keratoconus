# KERASCAN Phase 3 model card

## Intended use

Portable initial screening aid for school-age children using Placido images and the configured referral protocol. Output is `SCREEN_NEGATIVE`, `STANDARD_REFERRAL`, `PRIORITY_REFERRAL`, `RECAPTURE_REQUIRED`, `INCOMPLETE`, or `MANUAL_REVIEW`.

## Not intended use

Not a diagnosis, not a replacement for specialist assessment, and not physical corneal topography. `NORMAL-LIKE` is an image-engine label, not a disease exclusion.

## Data and validation

The committed prototype uses synthetic development data only. Confidential datasets remain local. A frozen model bundle records feature schema, threshold, pipeline version, partition provenance, and a file hash. No clinical performance or deployment-suitability claim is authorized from synthetic development data; any future claim requires a prospectively approved validation protocol and independent evaluation on an untouched patient-level locked test set.

## Evaluation requirements

Always report accuracy with sensitivity, specificity, PPV, NPV, balanced accuracy, macro-F1, ROC-AUC, PR-AUC, calibration/Brier score, ungradable rate, referral rate, per-class counts, patient-grouped bootstrap intervals, and the confusion matrix.

## Risks

ROI errors, broken rings, occlusion, blur, glare, device shifts, wrong-eye upload, representative-data gaps, and referral-rule configuration can degrade performance. The quality gate should route poor images to recapture rather than force a normal-like classification.
