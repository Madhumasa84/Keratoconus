"""Review the screening outcome for both eyes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.services.ui_security import require_authenticated

st.set_page_config(page_title="Findings — KeraScan", layout="wide")
require_authenticated(st)
st.title("Findings")

result = st.session_state.get("analysis_result")
if not result:
    st.error("No analysis result found. Run Analysis first.")
    st.stop()

# Plain-English wording for each reason code. Raw codes are never shown.
REASON_TEXT = {
    "IMAGE_CLASSIFIER_SUSPICIOUS": "The ring pattern looked irregular",
    "K2_ABOVE_46_8_D": "K2 above 46.8 D",
    "PACHYMETRY_BELOW_480_UM": "Corneal thickness below 480 µm",
    "CYLINDER_MAGNITUDE_ABOVE_1_5_D": "Cylinder above 1.5 D",
    "MULTIPLE_QUANTITATIVE_ABNORMALITIES": "More than one measurement was abnormal",
    "MEASUREMENT_MISSING": "A measurement is missing",
    "MEASUREMENT_INVALID": "A measurement needs correcting",
    "IMAGE_MISSING": "An image is missing",
    "IMAGE_REJECTED": "The image could not be used",
    "SEGMENTATION_FAILED": "The ring pattern could not be found",
    "TRACKING_FAILED": "Too much of the ring pattern was hidden",
    "ANALYSIS_BLOCKED": "The image could not be analysed",
    "IMAGE_NOT_READY": "The image is not ready",
    "IMAGE_INDETERMINATE": "The result was borderline",
}

IMAGE_SUMMARY = {
    "NORMAL_LIKE": ("success", "Ring pattern looks regular"),
    "SUSPICIOUS": ("error", "Ring pattern looks irregular"),
    "INDETERMINATE": ("warning", "Ring pattern is borderline"),
}

# Ring-spacing / comparison images, most informative first.
IMAGE_PREFERENCE = (
    "clinician_comparison_panel.png",
    "observed_vs_concentric_reference.png",
    "reference_spacing_residual_heatmap.png",
    "directional_spacing.png",
    "tracked_rings_cartesian.png",
    "cropped_roi_centres.png",
    "cropped_roi.png",
)

IMAGE_CAPTIONS = {
    "clinician_comparison_panel.png": "Ring-pattern comparison",
    "observed_vs_concentric_reference.png": "Observed rings (solid) vs even-spacing reference (dashed)",
    "reference_spacing_residual_heatmap.png": "Ring-spacing deviation across the cornea",
    "directional_spacing.png": "Ring-spacing pattern by direction",
    "tracked_rings_cartesian.png": "Detected ring pattern",
    "cropped_roi_centres.png": "Cropped ring image with centre",
    "cropped_roi.png": "Cropped corneal ring image",
}

columns = st.columns(2)
for column, eye_result, verification, eye_key, title in (
    (columns[0], result.od_eye_result, result.od_image_verification, "od", "Right eye (OD)"),
    (columns[1], result.os_eye_result, result.os_image_verification, "os", "Left eye (OS)"),
):
    with column:
        st.subheader(title)
        if eye_result is None:
            st.warning("No result available.")
            continue

        level, summary = IMAGE_SUMMARY.get(
            eye_result.image_status, ("error", "This image could not be analysed")
        )
        getattr(st, level)(summary)

        # Measurements, with the abnormal ones marked.
        measurements = st.session_state.get(f"{eye_key}_measurements") or {}
        flags = eye_result.flags
        rows = []
        if (value := measurements.get("k1_d")) is not None:
            rows.append(f"K1 {float(value):.2f} D")
        if (value := measurements.get("k2_d")) is not None:
            mark = " ⚠️" if flags and flags.keratometry == "ABNORMAL" else ""
            rows.append(f"K2 {float(value):.2f} D{mark}")
        if (value := measurements.get("pachymetry_um")) is not None:
            mark = " ⚠️" if flags and flags.pachymetry == "ABNORMAL" else ""
            rows.append(f"Thickness {float(value):.0f} µm{mark}")
        if (value := measurements.get("cylinder_d")) is not None:
            mark = " ⚠️" if flags and flags.refraction == "ABNORMAL" else ""
            rows.append(f"Cylinder {float(value):.2f} D{mark}")
        if rows:
            st.write(" · ".join(rows))

        if eye_result.reason_codes:
            for code in eye_result.reason_codes:
                st.write(f"- {REASON_TEXT.get(code, code)}")
        if eye_result.missing_or_invalid_fields:
            st.warning("Needs correcting: " + "; ".join(eye_result.missing_or_invalid_fields))

        # Ring-spacing images are shown for every eye whose analysis actually
        # completed, normal ones included, so the operator can see what the
        # result was based on rather than taking the wording on trust.
        #
        # An eye that FAILED still leaves intermediate artefacts on disk, but
        # those were discarded by the engine: the detected "rings" may be
        # tracing eyelashes or skin. Presenting them beside confident captions
        # would imply a measurement that was never accepted, so a failed eye
        # gets its diagnostics behind an explicit warning instead.
        manifest = getattr(verification, "artifact_manifest", {}) or {}
        available = [
            (name, manifest[name]["path"])
            for name in IMAGE_PREFERENCE
            if manifest.get(name) and Path(manifest[name].get("path", "")).exists()
        ]
        analysed = eye_result.image_status in IMAGE_SUMMARY
        if not available:
            st.caption("No analysis image available for this eye.")
        elif analysed:
            name, path = available[0]
            st.image(path, caption=IMAGE_CAPTIONS.get(name, name), use_container_width=True)
            st.caption("Image-space ring comparison for review. Not a corneal map.")
            if len(available) > 1:
                with st.expander(f"More images ({len(available) - 1})"):
                    for name, path in available[1:]:
                        st.image(path, caption=IMAGE_CAPTIONS.get(name, name), use_container_width=True)
        else:
            with st.expander("Why this image was not used"):
                st.warning(
                    "These are discarded working images, not a result. The ring pattern "
                    "could not be reliably located, so the detected rings below may be "
                    "following eyelashes or skin rather than the cornea. Do not read them "
                    "as measurements — upload a clearer photo of this eye."
                )
                for name, path in available[:3]:
                    st.image(path, caption=f"Rejected — {IMAGE_CAPTIONS.get(name, name)}", use_container_width=True)

st.divider()
child = result.child_result
if child:
    if child.action == "REFER":
        st.error(
            f"**Refer** — {', '.join(child.affected_eyes)}. "
            "Corneal tomography and specialist assessment are recommended."
        )
    elif child.decision == "REPEAT_REQUIRED":
        st.warning("**Repeat measurement** — one measurement was abnormal on its own. Repeat it before finishing.")
    elif child.decision == "INCOMPLETE_SCREENING":
        st.warning("**Incomplete** — both eyes need a usable image and complete measurements.")
    else:
        st.success("**No referral needed** — nothing met the referral criteria today.")

st.caption("Initial screening result — not a diagnosis.")
st.page_link("pages/06_confirm_report.py", label="Next: confirm and export →")
