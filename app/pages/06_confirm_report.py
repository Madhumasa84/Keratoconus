"""Confirm outcome and expose only allowed local exports."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.services.ui_security import require_authenticated

st.set_page_config(page_title="Confirm & Export — KeraScan", layout="wide")
require_authenticated(st)
st.title("Confirm & Export")

result = st.session_state.get("analysis_result")
if not result or not result.child_result:
    st.error("No completed analysis result found. Run Analysis first.")
    st.stop()

child = result.child_result
if child.action == "REFER":
    st.error(f"**Refer** — {', '.join(child.affected_eyes)}")
elif child.decision == "REPEAT_REQUIRED":
    st.warning("**Repeat measurement** before finishing.")
elif child.decision == "INCOMPLETE_SCREENING":
    st.warning("**Incomplete** — both eyes need a usable image and complete measurements.")
else:
    st.success("**No referral needed**")

confirmed = st.checkbox("I have reviewed this result.", key="confirm_outcome")
st.session_state["decision_confirmed"] = confirmed
if not confirmed:
    st.stop()

from app.database import SessionLocal
from app.database.repository import ScreeningRepository
from app.services.report_service import ReportService
from app.services.screening_service import ScreeningService

exports_root = Path(os.environ.get("KERASCAN_LOCAL_OUTPUT_DIR", str(Path.home() / ".kerascan" / "outputs")))
output_dir = exports_root / result.screening_id
output_dir.mkdir(parents=True, exist_ok=True)
register_path = exports_root / "screening_register.xlsx"
with SessionLocal() as session:
    full_data = ScreeningRepository(session).get_screening_full(result.screening_id)

if not full_data:
    st.error("The local screening record is unavailable; no export can be generated.")
    st.stop()

report_service = ReportService()

# Every confirmed screening lands in one cumulative register for the camp.
# Re-confirming the same child updates that child's row rather than duplicating.
try:
    report_service.append_to_register(full_data, register_path)
    st.caption(f"Added to the screening register ({register_path.name}).")
except Exception as exc:  # pragma: no cover - surfaced to the operator
    st.warning(f"Could not update the screening register: {exc}")
pdf_col, data_col = st.columns(2)
with pdf_col:
    if child.action != "REFER":
        st.caption("A referral letter is produced only when a referral is needed.")
    elif st.button("Referral letter (PDF)", use_container_width=True, type="primary"):
        try:
            with SessionLocal() as session:
                service = ScreeningService(db_session=session)
                path = service.generate_referral_pdf(result, output_dir / f"{result.screening_id}-referral.pdf")
            with open(path, "rb") as handle:
                st.download_button("Download PDF", handle, file_name=Path(path).name, mime="application/pdf")
        except Exception as exc:
            st.error(f"Referral PDF error: {exc}")
with data_col:
    if st.button("Screening record (Excel)", use_container_width=True):
        try:
            path = report_service.generate_excel(full_data, str(output_dir / f"{result.screening_id}.xlsx"))
            with open(path, "rb") as handle:
                st.download_button("Download Excel", handle, file_name=Path(path).name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as exc:
            st.error(f"Excel error: {exc}")
    with st.expander("Full data (JSON)"):
        if st.button("Generate JSON", use_container_width=True):
            try:
                path = report_service.generate_json(full_data, str(output_dir / f"{result.screening_id}.json"))
                with open(path, encoding="utf-8") as handle:
                    st.download_button("Download JSON", handle.read(), file_name=Path(path).name, mime="application/json")
            except Exception as exc:
                st.error(f"JSON error: {exc}")

st.divider()
register_col, next_col = st.columns(2)
with register_col:
    if register_path.exists():
        with open(register_path, "rb") as handle:
            st.download_button(
                "Download screening register (all children)",
                handle,
                file_name=register_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
with next_col:
    if st.button("Screen another child →", use_container_width=True, type="primary"):
        # Clear this child's screening but keep the operator logged in.
        for key in (
            "current_screening", "current_step", "analysis_result", "decision_confirmed",
            "od_image_path", "os_image_path", "od_image_verification", "os_image_verification",
            "od_measurements", "os_measurements", "od_measurements_r2", "os_measurements_r2",
            "od_upload_content_hash", "os_upload_content_hash", "od_upload", "os_upload",
            "confirm_outcome",
        ):
            st.session_state.pop(key, None)
        for eye in ("OD", "OS"):
            for field in ("k1", "k2", "pachymetry", "cylinder"):
                st.session_state.pop(f"active_{field}_{eye}", None)
        st.switch_page("pages/01_new_screening.py")
