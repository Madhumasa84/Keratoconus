"""Page 1 — New Screening form."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kerascan" / "src"))

import streamlit as st
from datetime import date

st.set_page_config(page_title="New Screening — KERASCAN", layout="wide")
st.title("📋 New Screening")
st.caption("Complete all required fields before proceeding.")

try:
    from app.database import SessionLocal, init_db
    from app.database.repository import ScreeningRepository
    init_db()
    _db_ok = True
except Exception as e:
    _db_ok = False
    st.error(f"Database not available: {e}")

with st.form("screening_form"):
    st.subheader("Screening Details")
    col1, col2 = st.columns(2)
    with col1:
        screening_id = st.text_input("Screening ID *", help="Alphanumeric identifier (hyphens/underscores allowed)")
        age = st.number_input("Age *", min_value=5, max_value=25, step=1, value=12)
        sex = st.selectbox("Sex *", ["Male", "Female", "Not recorded"])
    with col2:
        site = st.text_input("School / Site *")
        screening_date = st.date_input("Screening Date *", value=date.today())
        device_id = st.text_input("Device ID *")

    operator_id = st.session_state.get("operator_id", "")
    st.text_input("Operator ID", value=operator_id, disabled=True)
    consent = st.checkbox("Consent recorded ✓ *", help="Must be checked to proceed")

    submitted = st.form_submit_button("Save and Proceed →", use_container_width=True)

if submitted:
    from app.services.screening_service import ScreeningService
    svc = ScreeningService()
    form_data = {
        "screening_id": screening_id,
        "age": age,
        "sex": sex,
        "site": site,
        "screening_date": str(screening_date),
        "operator_id": operator_id,
        "device_id": device_id,
        "consent_recorded": consent,
    }
    valid, errors = svc.validate_screening_form(form_data)

    if _db_ok:
        with SessionLocal() as session:
            repo = ScreeningRepository(session)
            if repo.screening_id_exists(screening_id):
                errors.append(f"Screening ID '{screening_id}' already exists.")
                valid = False

    if not valid:
        for e in errors:
            st.error(e)
    else:
        st.session_state["current_screening"] = form_data
        st.session_state["current_step"] = 2
        st.success(f"✓ Screening {screening_id} saved. Proceed to Upload Images.")
        st.page_link("pages/02_upload_images.py", label="→ Upload Images")
