"""Page 4 — Run Phase 1 engine + referral rules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kerascan" / "src"))

import streamlit as st

st.set_page_config(page_title="Analysis — KERASCAN", layout="wide")
from app.services.ui_security import require_authenticated
require_authenticated(st)
st.title("Run Analysis")

for k in ("current_screening", "od_measurements", "os_measurements", "od_image_path", "os_image_path"):
    if not st.session_state.get(k):
        st.error(f"Complete earlier steps first (missing: {k}).")
        st.stop()

screening = st.session_state["current_screening"]

st.caption(
    f"Child {screening.get('screening_id', '—')} · age {screening.get('age', '—')} · "
    f"{screening.get('site', '—')}"
)

if st.button("Run Analysis", type="primary", use_container_width=True):
    try:
        from app.database import SessionLocal, init_db
        from app.services.screening_service import ScreeningService

        init_db()

        with SessionLocal() as session:
            svc = ScreeningService(db_session=session)

            screening_data = {
                "form": screening,
                "od_image_path": st.session_state.get("od_image_path", ""),
                "os_image_path": st.session_state.get("os_image_path", ""),
                "od_measurements": st.session_state.get("od_measurements", {}),
                "os_measurements": st.session_state.get("os_measurements", {}),
                "od_measurements_r2": st.session_state.get("od_measurements_r2"),
                "os_measurements_r2": st.session_state.get("os_measurements_r2"),
                "analysis_output_dir": str(Path(st.session_state.get("od_image_path")).parent / "analysis" / "final"),
            }

            with st.spinner("Analysing both eyes…"):
                result = svc.conduct_screening(screening_data)

        if not result.success:
            for e in result.validation_errors:
                st.error(e)
            st.error(result.error_message)
        else:
            st.session_state["analysis_result"] = result
            child = result.child_result
            if child:
                if child.action == "REFER":
                    st.error(f"Refer — {', '.join(child.affected_eyes)}")
                elif child.decision == "REPEAT_REQUIRED":
                    st.warning("Repeat one measurement before finishing.")
                elif child.decision == "INCOMPLETE_SCREENING":
                    st.warning("Screening incomplete — both eyes need a usable image and complete measurements.")
                else:
                    st.success("No referral needed")

            st.page_link("pages/05_review_findings.py", label="See details →")

    except Exception as exc:
        st.exception(exc)
