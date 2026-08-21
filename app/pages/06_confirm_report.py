"""Page 6 — Operator confirmation, override, and export."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from datetime import datetime, timezone

st.set_page_config(page_title="Confirm & Export — KERASCAN", layout="wide")
st.title("✅ Operator Confirmation & Export")

result = st.session_state.get("analysis_result")
if not result:
    st.error("No analysis result. Complete analysis first.")
    st.stop()

child = result.child_result
st.subheader("Automated Decision Summary")
if child:
    st.info(f"**Overall:** {child.decision}  |  **Priority:** {child.referral_priority}")

st.divider()
tab_accept, tab_override = st.tabs(["Accept Decision", "Override Decision"])

with tab_accept:
    st.write("Accepting the automated decision. The result will be recorded as-is.")
    if st.button("✓ Accept Automated Decision", type="primary", use_container_width=True):
        st.session_state["decision_confirmed"] = True
        st.session_state["override_data"] = None
        st.success("Decision accepted and recorded.")

with tab_override:
    st.warning(
        "**Overriding the automated decision is permanently recorded in the audit log and cannot be undone.** "
        "The original automated decision is always preserved.",
        icon="⚠️"
    )
    with st.form("override_form"):
        confirm_op_id = st.text_input("Re-enter your Operator ID to confirm identity")
        new_decision = st.selectbox("New Decision", [
            "SCREEN_NEGATIVE", "STANDARD_REFERRAL", "PRIORITY_REFERRAL",
            "RECAPTURE_REQUIRED", "INCOMPLETE", "MANUAL_REVIEW",
        ])
        override_reason = st.text_area(
            "Mandatory override reason (min 20 characters)",
            help="Provide detailed clinical justification for overriding the automated decision."
        )
        submitted = st.form_submit_button("Apply Override", use_container_width=True)

    if submitted:
        errors = []
        if confirm_op_id.strip() != st.session_state.get("operator_id", ""):
            errors.append("Operator ID does not match. Please re-enter correctly.")
        if len(override_reason.strip()) < 20:
            errors.append("Override reason must be at least 20 characters.")
        if child and new_decision == child.decision:
            errors.append("New decision must differ from the automated decision.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.session_state["decision_confirmed"] = True
            st.session_state["override_data"] = {
                "user_identity": confirm_op_id,
                "override_new": new_decision,
                "override_reason": override_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "original_decision": child.decision if child else "",
            }
            st.success(f"Override recorded: {child.decision if child else '?'} → {new_decision}")

st.divider()
st.subheader("Export Reports")
if not st.session_state.get("decision_confirmed"):
    st.info("Confirm or override the decision above before exporting.")
else:
    export_dir = Path(__file__).parent.parent / "data" / "exports" / result.screening_id
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        from app.database import SessionLocal
        from app.database.repository import ScreeningRepository
        from app.services.report_service import ReportService

        with SessionLocal() as session:
            repo = ScreeningRepository(session)
            full_data = repo.get_screening_full(result.screening_id) or {}
    except Exception:
        full_data = {"screening_id": result.screening_id}

    report_svc = ReportService()
    col_pdf, col_json, col_excel = st.columns(3)

    with col_pdf:
        if st.button("📄 Generate PDF", use_container_width=True):
            try:
                with st.spinner("Generating PDF..."):
                    path = report_svc.generate_pdf(full_data, str(export_dir / f"{result.screening_id}.pdf"))
                st.success("PDF generated!")
                with open(path, "rb") as f:
                    st.download_button("⬇ Download PDF", f, file_name=f"{result.screening_id}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF error: {e}")

    with col_json:
        if st.button("📦 Generate JSON", use_container_width=True):
            try:
                path = report_svc.generate_json(full_data, str(export_dir / f"{result.screening_id}.json"))
                st.success("JSON generated!")
                with open(path) as f:
                    st.download_button("⬇ Download JSON", f.read(), file_name=f"{result.screening_id}.json", mime="application/json")
            except Exception as e:
                st.error(f"JSON error: {e}")

    with col_excel:
        if st.button("📊 Generate Excel", use_container_width=True):
            try:
                path = report_svc.generate_excel(full_data, str(export_dir / f"{result.screening_id}.xlsx"))
                st.success("Excel generated!")
                with open(path, "rb") as f:
                    st.download_button("⬇ Download Excel", f, file_name=f"{result.screening_id}.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Excel error: {e}")
