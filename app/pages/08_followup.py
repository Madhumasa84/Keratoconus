"""Page 8 — Record Pentacam / corneal tomography follow-up."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from datetime import date

st.set_page_config(page_title="Pentacam Follow-Up — KERASCAN", layout="wide")
st.title("🔭 Pentacam / Corneal Tomography Follow-Up")

try:
    from app.database import SessionLocal, init_db
    from app.database.repository import ScreeningRepository
    init_db()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

screening_search = st.text_input("Search Screening ID to link follow-up")
screening_uuid = None

if screening_search:
    with SessionLocal() as session:
        repo = ScreeningRepository(session)
        found = repo.get_screening(screening_search)
    if found:
        st.success(f"Found: {screening_search} — {found.get('overall_result', 'unknown result')}")
        screening_uuid = found.get("id")
    else:
        st.warning("Screening not found.")

if screening_uuid:
    st.divider()
    with st.form("followup_form"):
        st.subheader("Follow-Up Measurements")
        exam_date = st.date_input("Exam date", value=date.today())
        col1, col2 = st.columns(2)
        with col1:
            kmax_od = st.number_input("Kmax OD (D)", 30.0, 90.0, 44.0, 0.01)
            ba_d_od = st.number_input("Belin-Ambrósio D OD", -20.0, 50.0, 0.0, 0.1)
        with col2:
            kmax_os = st.number_input("Kmax OS (D)", 30.0, 90.0, 44.0, 0.01)
            ba_d_os = st.number_input("Belin-Ambrósio D OS", -20.0, 50.0, 0.0, 0.1)
        performed_by = st.text_input("Performed by")
        notes = st.text_area("Clinical notes")
        submitted = st.form_submit_button("Save Follow-Up", use_container_width=True)

    if submitted:
        if not performed_by.strip():
            st.error("'Performed by' is required.")
        else:
            try:
                with SessionLocal() as session:
                    repo = ScreeningRepository(session)
                    fid = repo.save_pentacam_followup({
                        "screening_id": screening_uuid,
                        "exam_date": str(exam_date),
                        "kmax_od": kmax_od,
                        "kmax_os": kmax_os,
                        "belin_ambrosio_d_od": ba_d_od,
                        "belin_ambrosio_d_os": ba_d_os,
                        "notes": notes,
                        "performed_by": performed_by,
                    })
                    session.commit()
                st.success(f"Follow-up saved (id: {fid})")
            except Exception as e:
                st.error(f"Save failed: {e}")
