"""
streamlit_app.py — KERASCAN Phase 2 main entry point.

Start with:
    cd /home/masa84/e1
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make app/ and kerascan/src importable
APP_DIR = Path(__file__).parent
REPO_ROOT = APP_DIR.parent
PHASE1_SRC = REPO_ROOT / "kerascan" / "src"
for p in (str(REPO_ROOT), str(APP_DIR.parent), str(PHASE1_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(
    page_title="KERASCAN — Keratoconus Screening",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise database on first run
try:
    from app.database import init_db
    init_db()
    _db_ok = True
except Exception as _e:
    _db_ok = False
    _db_err = str(_e)

# Load protocol version for display
try:
    from app.services.referral_engine import ReferralEngine
    _proto_ver = ReferralEngine().get_protocol_version()
except Exception:
    _proto_ver = "unknown"

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "operator_id":          "",
    "operator_authenticated": False,
    "current_screening":    {},
    "current_step":         1,
    "od_image_path":        "",
    "os_image_path":        "",
    "od_roi":               None,
    "os_roi":               None,
    "od_measurements":      {},
    "os_measurements":      {},
    "od_measurements_r2":   None,
    "os_measurements_r2":   None,
    "analysis_result":      None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar — operator login + navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Human_eye_diagram-sagittal_view-NEI.jpg/220px-Human_eye_diagram-sagittal_view-NEI.jpg",
             use_container_width=True, caption="KERASCAN")
    st.title("KERASCAN")
    st.caption(f"Protocol v{_proto_ver}")

    # Offline indicator
    st.success("🟢 Offline mode — no internet required", icon="💾")

    st.divider()

    # Operator login
    if not st.session_state["operator_authenticated"]:
        st.subheader("Operator Login")
        op_id = st.text_input("Operator ID", key="login_op_id")
        op_pin = st.text_input("PIN", type="password", key="login_pin")
        if st.button("Login", use_container_width=True):
            if op_id.strip() and op_pin.strip():
                st.session_state["operator_id"] = op_id.strip()
                st.session_state["operator_authenticated"] = True
                st.rerun()
            else:
                st.error("Enter both Operator ID and PIN.")
    else:
        st.success(f"✓ Logged in: {st.session_state['operator_id']}")
        if st.button("Logout", use_container_width=True):
            for k in _DEFAULTS:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()

    st.divider()
    st.caption("Navigation via pages in the sidebar above.")
    st.divider()
    st.caption(f"Protocol version: **{_proto_ver}**")
    if not _db_ok:
        st.error(f"Database error: {_db_err}")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("👁 KERASCAN — Keratoconus Screening System")
st.warning(
    "**AI-assisted screening tool.** Results require clinical validation. "
    "This is not a confirmed diagnosis. All positive, discordant, or ungradable "
    "findings must be reviewed by a qualified clinician.",
    icon="⚠️",
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Phase 2** — Offline screening application\nBuilt on Phase 1 image engine.")
with col2:
    st.info(f"**Protocol Version**\n{_proto_ver}")
with col3:
    db_status = "✅ Connected" if _db_ok else "❌ Error"
    st.info(f"**Database**\n{db_status}")

st.divider()
st.markdown("""
### Workflow
1. **New Screening** — Enter screening and patient details
2. **Upload Images** — OD and OS KERASCAN images (operator must label laterality)
3. **Measurements** — Enter keratometry, pachymetry, refraction values
4. **Analysis** — Run Phase 1 engine + referral rules
5. **Review** — Per-eye findings with reason codes
6. **Confirm & Export** — Operator sign-off, PDF/JSON/Excel
7. **Search History** — Find previous screenings
8. **Pentacam Follow-Up** — Record tomography results

Use the **sidebar** to navigate between pages.
""")

st.caption("KERASCAN Phase 2 — Research use only. Not for clinical deployment.")
