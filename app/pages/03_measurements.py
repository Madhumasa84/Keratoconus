"""Page 3 — Enter keratometry, pachymetry, refraction measurements."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Measurements — KERASCAN", layout="wide")
st.title("📐 Enter Measurements")

if not st.session_state.get("current_screening"):
    st.error("Complete Step 1 (New Screening) first.")
    st.stop()

try:
    from app.services.screening_service import ScreeningService
    svc = ScreeningService()
except Exception as e:
    st.error(f"Service load error: {e}")
    st.stop()

def measurement_form(laterality: str) -> dict:
    """Render measurement form for one eye. Returns dict of values."""
    st.subheader(f"{'Right Eye (OD)' if laterality == 'OD' else 'Left Eye (OS)'}")

    with st.expander("Keratometry", expanded=True):
        st.caption("K2 (steep K) is the primary referral metric. Kmax and Mean K are stored separately.")
        c1, c2 = st.columns(2)
        with c1:
            k1_d    = st.number_input(f"K1 flat (D) [{laterality}]", 30.0, 70.0, 43.0, 0.01, key=f"k1_{laterality}")
            k1_axis = st.number_input(f"K1 axis (°) [{laterality}]", 0, 180, 180, 1, key=f"k1ax_{laterality}")
            kmax_d  = st.number_input(f"Kmax (D) [{laterality}] — optional", 30.0, 70.0, 44.0, 0.01, key=f"kmax_{laterality}")
        with c2:
            k2_d    = st.number_input(f"K2 steep (D) [{laterality}]", 30.0, 70.0, 44.0, 0.01, key=f"k2_{laterality}")
            k2_axis = st.number_input(f"K2 axis (°) [{laterality}]", 0, 180, 90, 1, key=f"k2ax_{laterality}")
            mean_k  = st.number_input(f"Mean K (D) [{laterality}] — optional", 30.0, 70.0, 43.5, 0.01, key=f"meank_{laterality}")

        # Reading 2
        st.caption("Reading 2 (optional — enter if performed)")
        c3, c4 = st.columns(2)
        with c3:
            k2_r2 = st.number_input(f"K2 Reading 2 (D) [{laterality}]", 30.0, 70.0, 44.0, 0.01, key=f"k2r2_{laterality}")
        with c4:
            has_r2 = st.checkbox(f"Reading 2 performed [{laterality}]", key=f"r2_{laterality}")

        if has_r2:
            diff = abs(k2_d - k2_r2)
            if diff > 0.5:
                st.warning(f"⚠ K2 readings disagree by {diff:.2f}D (threshold 0.5D). A third reading is required.", icon="⚠️")

    with st.expander("Pachymetry", expanded=True):
        st.info("Record only the type measured. Do not convert between central and thinnest.", icon="ℹ️")
        c5, c6 = st.columns(2)
        with c5:
            central_pachy = st.number_input(f"Central pachymetry (µm) [{laterality}]", 200.0, 800.0, 540.0, 1.0, key=f"cpachy_{laterality}")
            has_central   = st.checkbox(f"Central pachymetry measured [{laterality}]", key=f"hc_{laterality}", value=True)
        with c6:
            thinnest_pachy = st.number_input(f"Thinnest pachymetry (µm) [{laterality}]", 200.0, 800.0, 530.0, 1.0, key=f"tpachy_{laterality}")
            has_thinnest   = st.checkbox(f"Thinnest pachymetry measured [{laterality}]", key=f"ht_{laterality}")

    with st.expander("Refraction", expanded=True):
        ref_type = st.selectbox(f"Refraction type [{laterality}]", ["autorefraction", "subjective"], key=f"reftype_{laterality}")
        c7, c8, c9 = st.columns(3)
        with c7:
            sphere = st.number_input(f"Sphere (D) [{laterality}]", -30.0, 20.0, 0.0, 0.25, key=f"sph_{laterality}")
        with c8:
            cylinder = st.number_input(f"Cylinder (D) [{laterality}]", 0.0, 12.0, 0.0, 0.25, key=f"cyl_{laterality}")
        with c9:
            cyl_axis = st.number_input(f"Cyl axis (°) [{laterality}]", 0, 180, 0, 1, key=f"cylax_{laterality}")
        va = st.number_input(f"VA (logMAR) [{laterality}] — optional", -0.3, 3.0, 0.0, 0.01, key=f"va_{laterality}")

    with st.expander("Quality & Clinical Flags"):
        quality = st.selectbox(f"Measurement quality [{laterality}]", ["Good", "Acceptable", "Poor"], key=f"qual_{laterality}")
        clinical_flags = st.multiselect(
            f"Clinical signs [{laterality}]",
            ["Vogt striae", "Fleischer ring", "Corneal scarring", "Other clinical sign"],
            key=f"flags_{laterality}",
        )

    # Build measurement dict
    meas = {
        "k1_d": k1_d, "k1_axis": k1_axis,
        "k2_d": k2_d, "k2_axis": k2_axis,
        "kmax_d": kmax_d if kmax_d != 44.0 else None,
        "mean_k_d": mean_k if mean_k != 43.5 else None,
        "sphere_d": sphere, "cylinder_d": cylinder, "cylinder_axis": cyl_axis,
        "va_logmar": va if va != 0.0 else None,
        "refraction_type": ref_type,
        "measurement_quality": quality,
        "clinical_flags": clinical_flags if clinical_flags else None,
    }

    # Pachymetry
    if has_central:
        meas["pachymetry_um"] = central_pachy
        meas["pachymetry_type"] = "central"
    elif has_thinnest:
        meas["pachymetry_um"] = thinnest_pachy
        meas["pachymetry_type"] = "thinnest"
    else:
        meas["pachymetry_um"] = None
        meas["pachymetry_type"] = None

    meas_r2 = None
    if has_r2:
        meas_r2 = dict(meas)
        meas_r2["k2_d"] = k2_r2
        meas_r2["reading_number"] = 2

    meas["reading_number"] = 1

    # Live validation feedback
    valid, errors = svc.validate_measurements(meas)
    if not valid:
        for e in errors:
            st.error(e)

    return meas, meas_r2

tab_od, tab_os = st.tabs(["Right Eye (OD)", "Left Eye (OS)"])
with tab_od:
    od_meas, od_meas_r2 = measurement_form("OD")
with tab_os:
    os_meas, os_meas_r2 = measurement_form("OS")

st.divider()
if st.button("Save Measurements and Proceed →", use_container_width=True, type="primary"):
    od_valid, od_errors = svc.validate_measurements(od_meas)
    os_valid, os_errors = svc.validate_measurements(os_meas)
    all_errors = [f"OD: {e}" for e in od_errors] + [f"OS: {e}" for e in os_errors]

    if all_errors:
        for e in all_errors:
            st.error(e)
    else:
        st.session_state["od_measurements"] = od_meas
        st.session_state["os_measurements"] = os_meas
        st.session_state["od_measurements_r2"] = od_meas_r2
        st.session_state["os_measurements_r2"] = os_meas_r2
        st.session_state["current_step"] = 4
        st.success("Measurements saved. Proceed to Analysis.")
        st.page_link("pages/04_analysis.py", label="→ Run Analysis")
