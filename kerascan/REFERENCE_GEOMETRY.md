# Self-fitted Placido reference geometry

This module is an engineering comparison of directly observed tracked Placido
rings with two artificial references fitted from the same eye. It is not a
validated normal-cornea template and cannot independently create a clinical
classification.

For tracked radius `R[k, theta]`, the concentric reference uses the directly
observed robust median for each ring:

```text
rho[k] = median_theta(R[k, theta])
R_circle[k, theta] = rho[k]
```

The radii are never sorted or forced to equal pitch. Every required ring must
meet the configured direct-observation coverage, and `rho[k] < rho[k+1]` must
hold. A missing or invalid middle ring invalidates the complete reference.

The smooth reference is fitted separately to directly observed points on each
ring:

```text
R_smooth[k, theta] = a0
  + a1 cos(theta) + b1 sin(theta)
  + a2 cos(2 theta) + b2 sin(2 theta)
```

The fit uses Huber iterative reweighting. An isolated high-residual sample may
be rejected, while a contiguous residual sector is retained so a persistent
local deformation is not erased as an outlier. The fitted rings must remain
strictly ordered at every sampled meridian.

Signed radial residuals are `R - R_reference`. Signed spacing residuals compare
each directly observed adjacent spacing with the corresponding reference
spacing. Normalization uses that ring pair's directly observed median spacing;
the outermost ring explicitly uses its inward adjacent-pair median because it
has no outward neighbour. Missing, rejected, and interpolated-only points do
not become observed deviations.

Cross-ring coherence requires the same signed residual direction in
neighbouring rings, persistence over configured adjacent angular samples, and
directly observed support. It is not standard deviation. All persistence and
coverage values are engineering configuration, not clinical disease cutoffs.

The pipeline remains `ANALYSIS_BLOCKED` when verified hardware ring count is
missing and `NOT_CALIBRATED` when hardware and tracked-geometry gates pass but
approved geometry thresholds are absent. Invalid artificial-reference rings are
marked invalid and not drawn; the ancillary self-fitted comparison does not add
or bypass a clinical decision gate. Artificial references alone never generate
a referral report or disease probability.

Limitations: a self-fitted reference may absorb diffuse or global deformation;
all values are image-space proxies; clinical use would require verified device
configuration, separately developed labelled-normal references, locked
threshold calibration, and prospective validation.
