"""Page 4 — Run Phase 1 engine + referral rules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kerascan" / "src"))

import streamlit as st

st.set_page_config(page_title="Analysis — KERASCAN", layout="wide")
from app.services.ui_security import require_authenticated
require_authenticated(st)
st.title("🔬 Run Analysis")

for k in ("current_screening", "od_measurements", "os_measurements"):
    if not st.session_state.get(k):
        st.error(f"Complete earlier steps first (missing: {k}).")
        st.stop()

screening = st.session_state["current_screening"]

# Summary
st.subheader("Screening Summary")
col1, col2 = st.columns(2)
with col1:
    st.json(screening, expanded=False)
with col2:
    od_img = st.session_state.get("od_image_path", "")
    os_img = st.session_state.get("os_image_path", "")
    st.write(f"**OD image:** {Path(od_img).name if od_img else '⚠ Not uploaded'}")
    st.write(f"**OS image:** {Path(os_img).name if os_img else '⚠ Not uploaded'}")

st.divider()

if st.button("▶ Run Analysis", type="primary", use_container_width=True):
    try:
        from app.database import SessionLocal, init_db
        from app.services.screening_service import ScreeningService

        init_db()
        progress = st.progress(0, "Initialising...")

        with SessionLocal() as session:
            svc = ScreeningService(db_session=session)
            progress.progress(10, "Validating form...")

            screening_data = {
                "form": screening,
                "od_image_path": st.session_state.get("od_image_path", ""),
                "os_image_path": st.session_state.get("os_image_path", ""),
                "od_measurements": st.session_state.get("od_measurements", {}),
                "os_measurements": st.session_state.get("os_measurements", {}),
                "od_measurements_r2": st.session_state.get("od_measurements_r2"),
                "os_measurements_r2": st.session_state.get("os_measurements_r2"),
            }

            progress.progress(20, "Analysing OD image (Phase 1 engine)...")
            with st.status("Running Phase 1 engine on OD...", expanded=True) as status:
                st.write("Loading Phase 1 KerascanEngine...")
                result = svc.conduct_screening(screening_data)
                status.update(label="Analysis complete!", state="complete")

        progress.progress(100, "Done.")

        if not result.success:
            for e in result.validation_errors:
                st.error(e)
            st.error(result.error_message)
        else:
            st.session_state["analysis_result"] = result
            st.success("✓ Analysis complete!")

            col_od, col_os = st.columns(2)
            with col_od:
                if result.od_eye_result:
                    dec = result.od_eye_result.decision
                    st.metric("OD Decision", dec)
                    if result.od_eye_result.reason_codes:
                        st.write("Reasons:", ", ".join(result.od_eye_result.reason_codes))
            with col_os:
                if result.os_eye_result:
                    dec = result.os_eye_result.decision
                    st.metric("OS Decision", dec)
                    if result.os_eye_result.reason_codes:
                        st.write("Reasons:", ", ".join(result.os_eye_result.reason_codes))

            if result.child_result:
                st.subheader("Child-Level Decision")
                is_ref = result.child_result.decision != "SCREEN_NEGATIVE"
                if is_ref:
                    st.error(f"**{result.child_result.decision}** — Priority: {result.child_result.referral_priority}")
                else:
                    st.success(f"**{result.child_result.decision}**")

            st.page_link("pages/05_review_findings.py", label="→ View Detailed Findings")

    except Exception as exc:
        st.exception(exc)
