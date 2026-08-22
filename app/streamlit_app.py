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
    "operator_role":        "",
    "operator_authenticated": False,
    "current_screening":    {},
    "current_step":         1,
    "od_image_path":        "",
    "os_image_path":        "",
    "od_image_verification": None,
    "os_image_verification": None,
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
                try:
                    from app.database import SessionLocal
                    from app.services.auth_service import LocalAuthService
                    with SessionLocal() as session:
                        identity = LocalAuthService(session).authenticate(op_id, op_pin)
                except Exception:
                    identity = None
                if identity:
                    st.session_state["operator_id"] = identity["operator_id"]
                    st.session_state["operator_role"] = identity["role"]
                    st.session_state["operator_authenticated"] = True
                else:
                    st.error("Local authentication failed. Ask an administrator to provision an account.")
                    st.stop()
                st.rerun()
            else:
                st.error("Enter both Operator ID and PIN.")
    else:
        st.success(f"✓ Logged in: {st.session_state['operator_id']} ({st.session_state['operator_role']})")
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
    else:
        try:
            from app.services.operations_service import storage_health
            health = storage_health(Path.home() / ".kerascan" / "kerascan.db")
            if health["warning"]:
                st.warning("Low local storage. Complete or safely stop active work, then free space before further screening.")
            else:
                st.caption(f"Local storage free: {health['free_bytes'] // (1024**3)} GB")
        except Exception:
            st.warning("Storage health could not be checked; verify local disk space before screening.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("👁 KeraScan")
st.caption("Offline corneal screening aid for school programmes.")
st.warning(
    "Screening aid only — results are not a diagnosis and do not replace clinical assessment.",
    icon="⚠️",
)

st.markdown("""
### How it works
1. **New Screening** — child and site details
2. **Upload Images** — one photo per eye
3. **Measurements** — K1, K2, thickness, cylinder
4. **Analysis** — run the screening
5. **Findings** — review both eyes
6. **Confirm & Export** — referral letter and records

Use the sidebar to move between steps.
""")

if not _db_ok:
    st.error(f"Database error: {_db_err}")

st.caption(f"Research use only · offline, no telemetry · protocol {_proto_ver}")
