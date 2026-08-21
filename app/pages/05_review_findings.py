"""Page 5 — Review per-eye findings and child-level decision."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Review Findings — KERASCAN", layout="wide")
from app.services.ui_security import require_authenticated
require_authenticated(st)
st.title("🔍 Review Findings")

result = st.session_state.get("analysis_result")
if not result:
    st.error("No analysis result found. Run Analysis first.")
    st.stop()

HUMAN_REASON = {
    "IMG_SUSPICIOUS":      "Suspicious Placido pattern",
    "IMG_UNGRADABLE":      "Image ungradable — recapture required",
    "K_HIGH":              "Elevated K2 (steep keratometry)",
    "PACHY_LOW":           "Low corneal thickness (pachymetry)",
    "CYL_HIGH":            "High cylinder (astigmatism)",
    "TWO_DOMAIN_ABNORMAL": "Two or more domains abnormal",
    "CLINICAL_SIGN":       "Clinical sign present",
    "INTER_EYE_ASYMMETRY": "Inter-eye keratometry asymmetry",
    "REPEAT_REQUIRED":     "Repeat measurement required",
    "MEASUREMENT_MISSING": "Required measurement(s) missing",
}

def badge_color(decision: str) -> str:
    if decision == "SCREEN_NEGATIVE":      return "✅"
    if decision == "PRIORITY_REFERRAL":    return "🔶"
    if decision == "STANDARD_REFERRAL":    return "🔷"
    if decision == "RECAPTURE_REQUIRED":   return "🔁"
    if decision == "INCOMPLETE":           return "⚠️"
    return "❓"

col_od, col_os = st.columns(2)

for col, eye_result, laterality in [
    (col_od, result.od_eye_result, "OD — Right Eye"),
    (col_os, result.os_eye_result, "OS — Left Eye"),
]:
    with col:
        st.subheader(laterality)
        if eye_result is None:
            st.warning("No result available.")
            continue

        dec = eye_result.decision
        icon = badge_color(dec)

        # Engine result
        engine_label = eye_result.engine_result or "—"
        st.metric("Phase 1 Engine Result", engine_label)
        st.metric(f"{icon} Referral Decision", dec)

        # Reason codes
        if eye_result.reason_codes:
            st.markdown("**Reason codes:**")
            for code in eye_result.reason_codes:
                human = HUMAN_REASON.get(code, code)
                st.markdown(f"- `{code}` — {human}")
        else:
            st.markdown("**Reason codes:** None")

        if eye_result.repeat_required:
            st.warning("⚠ Repeat measurement required before final decision.")

        # Engine raw data
        eng = result.od_engine_raw if laterality.startswith("OD") else result.os_engine_raw
        if eng:
            quality = eng.get("quality", {})
            with st.expander("Quality Metrics"):
                st.write(f"Gradable: {'Yes' if quality.get('gradable') else 'No'}")
                st.write(f"Quality score: {quality.get('quality_score', '—'):.1f}" if isinstance(quality.get('quality_score'), (int, float)) else "Quality score: —")
                flags = quality.get("flags", [])
                if flags:
                    st.write("Flags:", ", ".join(flags))
            roi = eng.get("roi", {})
            with st.expander("ROI Info"):
                st.write(f"Method: {roi.get('method', '—')}")
                st.write(f"Confidence: {roi.get('confidence', '—')}")

# Child-level
st.divider()
st.subheader("Child-Level Decision")
child = result.child_result
if child:
    icon = badge_color(child.decision)
    is_ref = child.decision not in ("SCREEN_NEGATIVE",)

    if is_ref:
        st.error(f"{icon} **{child.decision}**   |   Priority: **{child.referral_priority}**")
    else:
        st.success(f"{icon} **{child.decision}**")

    if child.inter_eye_asymmetry:
        st.warning("⚠ Inter-eye keratometry asymmetry detected (K2 difference exceeds threshold).")

    if child.reason_codes:
        st.markdown("**Child-level reason codes:** " + ", ".join(child.reason_codes))

    st.caption(f"Protocol version: {child.protocol_version}")

st.divider()
st.markdown(
    "> ⚠ *AI-assisted keratoconus screening result. This is not a confirmed diagnosis. "
    "Suspicious screening result—further corneal evaluation is recommended.*"
)

st.page_link("pages/06_confirm_report.py", label="→ Operator Confirmation & Export")
