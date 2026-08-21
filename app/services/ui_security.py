"""Shared Streamlit guard: pages cannot be used without a local session login."""
from __future__ import annotations

def require_authenticated(st) -> None:
    if not st.session_state.get("operator_authenticated"):
        st.error("Local operator login is required before using the screening workflow.")
        st.stop()
