# Known-risk and failure-mode register

| Category | Control | Required action |
|---|---|---|
| ROI or centre failure | Full-resolution ROI audit, manual ROI review | Recapture or manual review |
| Blur, glare, noise, low contrast, obstruction | Quality gate flags | Recapture; do not classify as normal |
| Missing rings / identity shifts | Radial and tracking metrics | Recapture or manual review |
| Wrong eye / non-ring input | Explicit OD/OS selection and input checks | Correct upload and rerun |
| Measurement disagreement | Repeat-reading protocol | Repeat or manual review |
| Patient leakage | Manifest patient partition audit | Stop evaluation |
| Model/protocol mismatch | Frozen model/provenance checks | Stop evaluation |
| Storage exhaustion/corruption | Storage warning, SQLite backup/restore | Stop safely and restore |
| Poor locked-set outcome | Immutable report and no-retuning warning | Improve only on approved development/calibration data |
