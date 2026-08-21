"""Page 7 — Search previous screenings."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Search History — KERASCAN", layout="wide")
from app.services.ui_security import require_authenticated
require_authenticated(st)
st.title("🔎 Search Screening History")

try:
    from app.database import SessionLocal, init_db
    from app.database.repository import ScreeningRepository
    init_db()
    _db_ok = True
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    query = st.text_input("Search (Screening ID or Site)", placeholder="e.g. SCR-001 or City School")
with col2:
    date_from = st.date_input("From date", value=None)
with col3:
    date_to = st.date_input("To date", value=None)

page = st.number_input("Page", min_value=1, value=1, step=1)
per_page = 50

with SessionLocal() as session:
    repo = ScreeningRepository(session)
    if query:
        rows = repo.search_screenings(query)
    else:
        rows = repo.list_screenings(
            limit=per_page,
            offset=(page - 1) * per_page,
            date_from=str(date_from) if date_from else None,
            date_to=str(date_to) if date_to else None,
        )

if not rows:
    st.info("No screenings found.")
else:
    st.write(f"**{len(rows)}** result(s)")
    cols = ["screening_id", "age", "sex", "site", "screening_date", "overall_result", "referral_priority", "created_at"]
    display = [{c: r.get(c, "") for c in cols} for r in rows]
    st.dataframe(display, use_container_width=True, hide_index=True)

    selected_id = st.selectbox("Select a screening to view", [""] + [r.get("screening_id", "") for r in rows])
    if selected_id:
        with SessionLocal() as session:
            repo = ScreeningRepository(session)
            full = repo.get_screening_full(selected_id)
        if full:
            st.subheader(f"Screening: {selected_id}")
            with st.expander("Full data"):
                st.json(full)

            export_dir = Path(__file__).parent.parent / "data" / "exports" / selected_id
            export_dir.mkdir(parents=True, exist_ok=True)

            from app.services.report_service import ReportService
            rpt = ReportService()
            col_pdf, col_json, col_xlsx = st.columns(3)
            with col_pdf:
                if st.button("Generate PDF", key="hist_pdf"):
                    try:
                        p = rpt.generate_pdf(full, str(export_dir / f"{selected_id}.pdf"))
                        with open(p, "rb") as f:
                            st.download_button("⬇ PDF", f, file_name=f"{selected_id}.pdf", mime="application/pdf")
                    except Exception as e:
                        st.error(str(e))
            with col_json:
                if st.button("Generate JSON", key="hist_json"):
                    try:
                        p = rpt.generate_json(full, str(export_dir / f"{selected_id}.json"))
                        with open(p) as f:
                            st.download_button("⬇ JSON", f.read(), file_name=f"{selected_id}.json")
                    except Exception as e:
                        st.error(str(e))
            with col_xlsx:
                if st.button("Generate Excel", key="hist_xlsx"):
                    try:
                        p = rpt.generate_excel(full, str(export_dir / f"{selected_id}.xlsx"))
                        with open(p, "rb") as f:
                            st.download_button("⬇ Excel", f, file_name=f"{selected_id}.xlsx")
                    except Exception as e:
                        st.error(str(e))
