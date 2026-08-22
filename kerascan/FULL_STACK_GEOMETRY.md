# Full-stack inter-ring spacing regularity

This Stage 1 engineering method evaluates the complete tracked Placido-ring
stack. It is experimental image-space geometry, not physical topography, a
clinical probability, or a keratoconus diagnosis.

For tracked radius `R[k, theta]`, every adjacent identity is retained:

```text
S[k, theta] = R[k + 1, theta] - R[k, theta]
```

No intermediate pair is skipped. Missing values remain missing. A directly
observed spacing requires direct observations of both adjacent radii. A
non-positive finite spacing makes the geometry ungradable; it is not converted
with an absolute value or clipped.

Each pair has its own directly observed angular baseline:

```text
B[k]        = median_theta(S[k, theta])
N[k, theta] = S[k, theta] / B[k]
```

This separates natural inner-to-outer pitch differences from angular change.
`N = 1` means only that a spacing equals its own ring-pair angular median. It
does not mean the eye or cornea is clinically normal.

The engine reports:

- pair-wise angular `MAD / median` and `(P90 - P10) / median`;
- per-meridian median `abs(N - 1)` across the radial stack;
- directly supported consecutive compression and expansion runs;
- cumulative residuals from the innermost tracked identity, calculated only
  when every intermediate identity is usable;
- mutually exclusive observed, interpolated, missing, and rejected fractions
  for rings, pairs, meridians, radial regions, and the full stack;
- seam-aware coherent angular sectors; and
- inner, middle, outer, and robust full-stack summaries.

Compression/expansion magnitude, neighbouring-pair persistence, angular
persistence, and missing-sector limits in `GeometryConfig` are engineering
rules for reproducible feature extraction and coverage gating. They are not
clinical thresholds. The separate `GeometryConfig.thresholds` remains unset by
default, so a hardware-verified, complete geometry result is `NOT_CALIBRATED`.

If `RadialConfig.expected_ring_count` is absent, provisional complete-stack
features and audit plots can be written, but `ring_count_verified` and
`classification_performed` remain false and `geometry_status` is
`ANALYSIS_BLOCKED`.

The ten full-stack plots are:

1. `full_stack_tracked_rings.png`
2. `inter_ring_spacing_matrix.png`
3. `normalized_inter_ring_spacing_matrix.png`
4. `angular_variation_by_ring_pair.png`
5. `radial_stack_deviation_by_meridian.png`
6. `neighbouring_ring_coherence.png`
7. `cumulative_radial_residual.png`
8. `ring_and_pair_completeness.png`
9. `inner_middle_outer_comparison.png`
10. `full_stack_sector_map.png`

These plots use pixels or dimensionless image-space proxies. Missing cells are
shown as gaps; interpolated and rejected support is marked separately where
applicable. None of the plots uses diagnostic wording.

