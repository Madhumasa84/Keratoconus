# Deferred image-quality research: blur comparison

The initial KeraScan school-screening workflow is **good-quality-only**. Its
existing acquisition-quality gate remains active: blurred, poorly exposed,
incomplete, or incorrectly centred images are rejected and must be replaced.
Rejected images are not sharpened, generatively enhanced, classified, or made
normal by downstream measurement values.

No blur-versus-good-image model training or clinical comparison is implemented
in this application phase. Any later study must occur under the appropriate
governance, outside locked-test evaluation, and must not retune the deployed
image pipeline from locked or confidential data.

## Future study questions

- Compare paired good-quality and blurred KeraScan images using approved,
  de-identified development data.
- Create controlled synthetic blur levels only as robustness checks, never as
  disease labels.
- Distinguish motion blur from defocus blur.
- Measure effects on ROI detection, segmentation, centre refinement, polar
  ring tracking, geometry validation, and image classification.
- Determine clinically acceptable acquisition-quality thresholds prospectively;
  do not lower current thresholds simply to make poor images pass.

Until that work is complete and separately approved, the good-quality-only
policy in `app/config/referral_protocol.yaml` is authoritative.
