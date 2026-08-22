"""Initial-protocol simplified OD/OS measurement entry."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.services.ui_security import require_authenticated

st.set_page_config(page_title="Measurements — KeraScan", layout="wide")
require_authenticated(st)
st.title("Measurements")

if not st.session_state.get("current_screening"):
    st.error("Complete Step 1 (New Screening) first.")
    st.stop()
if not st.session_state.get("od_image_path") or not st.session_state.get("os_image_path"):
    st.error("Upload both required KeraScan images first.")
    st.stop()

from app.services.screening_service import ScreeningService

service = ScreeningService()
protocol = service.protocol
st.caption(
    f"Referral thresholds: K2 > {protocol.k2_abnormal_above_d:g} D · "
    f"thickness < {protocol.pachymetry_abnormal_below_um:g} µm · "
    f"cylinder > {protocol.cylinder_magnitude_abnormal_above_d:g} D"
)


def _invalidate_prior_result() -> None:
    st.session_state["analysis_result"] = None
    st.session_state["decision_confirmed"] = False


def _measurement_form(eye: str) -> dict:
    st.caption("All four values are required.")
    k1 = st.number_input(
        "K1 flat (D)", min_value=30.0, max_value=70.0, value=None,
        step=0.01, format="%.2f", key=f"active_k1_{eye}", on_change=_invalidate_prior_result,
    )
    k2 = st.number_input(
        "K2 steep (D)", min_value=30.0, max_value=70.0, value=None,
        step=0.01, format="%.2f", key=f"active_k2_{eye}", on_change=_invalidate_prior_result,
    )
    pachymetry = st.number_input(
        "Thickness (µm)", min_value=200.0, max_value=800.0, value=None,
        step=1.0, format="%.0f", key=f"active_pachymetry_{eye}", on_change=_invalidate_prior_result,
        help="Use the pachymetry value defined by the protocol. Do not convert between central and thinnest.",
    )
    cylinder = st.number_input(
        "Cylinder (D)", min_value=-12.0, max_value=12.0, value=None,
        step=0.01, format="%.2f", key=f"active_cylinder_{eye}", on_change=_invalidate_prior_result,
        help="Plus or minus notation both accepted; screening uses the magnitude.",
    )
    values = {"k1_d": k1, "k2_d": k2, "pachymetry_um": pachymetry, "cylinder_d": cylinder}
    valid, errors = service.validate_measurements(values)
    if not valid:
        for error in errors:
            st.caption(f"⚠ {error}")
    return values


tab_od, tab_os = st.tabs(["Right Eye (OD)", "Left Eye (OS)"])
with tab_od:
    od_measurements = _measurement_form("OD")
with tab_os:
    os_measurements = _measurement_form("OS")

st.divider()
if st.button("Save Measurements and Proceed →", use_container_width=True, type="primary"):
    _, od_errors = service.validate_measurements(od_measurements)
    _, os_errors = service.validate_measurements(os_measurements)
    errors = [f"OD: {error}" for error in od_errors] + [f"OS: {error}" for error in os_errors]
    if errors:
        for error in errors:
            st.error(error)
    else:
        st.session_state["od_measurements"] = od_measurements
        st.session_state["os_measurements"] = os_measurements
        # The deferred multi-reading fields are intentionally not part of the
        # initial workflow and cannot affect a new decision.
        st.session_state["od_measurements_r2"] = None
        st.session_state["os_measurements_r2"] = None
        st.session_state["analysis_result"] = None
        st.session_state["current_step"] = 4
        st.success("Simplified measurements saved. Run bilateral analysis to finalize the screening state.")
        st.page_link("pages/04_analysis.py", label="Run Analysis →")
