# Referral protocol v3.0.0

The application evaluates each explicitly labelled eye and escalates at child level when either eye is positive.

- Suspicious image result: standard referral.
- Suspicious image plus an abnormal quantitative domain: priority referral.
- Normal-like image with two abnormal quantitative domains: priority referral.
- Confirmed isolated abnormal K2, pachymetry, or cylinder: standard referral according to repeat-reading rules.
- Ungradable image: `RECAPTURE_REQUIRED`; it is never converted to screen-negative.
- Missing required measurements: `INCOMPLETE`.

Thresholds are versioned in [referral_protocol.yaml](../app/config/referral_protocol.yaml): K2 ≥47.0 D, pachymetry ≤480 μm, cylinder magnitude ≥2.0 D, and inter-eye K2 difference ≥1.5 D. These provisional defaults require local clinical governance approval before deployment.

Required report wording: **Suspicious screening result—further corneal evaluation is recommended.**
