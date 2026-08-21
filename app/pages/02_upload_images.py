"""Page 2 — Upload OD/OS images and review ROI."""
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kerascan" / "src"))

import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Upload Images — KERASCAN", layout="wide")
st.title("🖼 Upload OD & OS Images")

st.warning(
    "**Do NOT rely on image appearance to determine laterality. "
    "The operator must explicitly select OD (right eye) or OS (left eye) for each image.**",
    icon="⚠️"
)

if not st.session_state.get("current_screening"):
    st.error("Complete Step 1 (New Screening) first.")
    st.stop()

screening = st.session_state["current_screening"]
screening_id = screening.get("screening_id", "unknown")
img_dir = Path(__file__).parent.parent / "data" / "images" / screening_id
img_dir.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp", "tiff"]

def save_and_analyse(uploaded_file, laterality: str):
    """Save image and run ROI detection."""
    suffix = Path(uploaded_file.name).suffix.lower()
    save_path = img_dir / f"{laterality.lower()}_original{suffix}"
    save_path.write_bytes(uploaded_file.getbuffer())

    try:
        img = Image.open(save_path).convert("RGB")
        st.image(img, caption=f"{laterality} — {uploaded_file.name} ({uploaded_file.size // 1024} KB)", width=300)
    except Exception as e:
        st.error(f"Cannot read image: {e}")
        return None

    return str(save_path)

col_od, col_os = st.columns(2)

with col_od:
    st.subheader("Right Eye (OD)")
    od_file = st.file_uploader("Upload OD image", type=ALLOWED_TYPES, key="od_upload")
    if od_file:
        path = save_and_analyse(od_file, "OD")
        if path:
            st.session_state["od_image_path"] = path
            st.success(f"OD image saved: {Path(path).name}")

            # ROI detection
            if st.button("Detect OD ROI", key="detect_od"):
                with st.spinner("Detecting OD ROI..."):
                    try:
                        from kerascan import EngineConfig, KerascanEngine
                        engine = KerascanEngine()
                        img_arr = np.array(Image.open(path).convert("RGB"))
                        result = engine.analyze(img_arr)
                        roi = result.get("roi", {})
                        st.session_state["od_roi"] = roi
                        st.success(f"ROI detected: method={roi.get('method')} confidence={roi.get('confidence', 0):.2f}")
                        st.json(roi)
                    except Exception as e:
                        st.warning(f"ROI detection failed: {e}. Use manual adjustment below.")

            # Manual ROI
            with st.expander("Manual ROI Adjustment"):
                st.caption("Adjust crop coordinates if automatic detection failed.")
                try:
                    img_pil = Image.open(path)
                    w, h = img_pil.size
                except Exception:
                    w, h = 640, 480
                x0 = st.slider("OD x0", 0, w, 0, key="od_x0")
                y0 = st.slider("OD y0", 0, h, 0, key="od_y0")
                x1 = st.slider("OD x1", 0, w, w, key="od_x1")
                y1 = st.slider("OD y1", 0, h, h, key="od_y1")
                if st.button("Apply manual OD ROI"):
                    st.session_state["od_roi"] = {"box_xyxy": [x0, y0, x1, y1], "method": "manual"}
                    st.success("Manual OD ROI applied.")

with col_os:
    st.subheader("Left Eye (OS)")
    os_file = st.file_uploader("Upload OS image", type=ALLOWED_TYPES, key="os_upload")
    if os_file:
        path = save_and_analyse(os_file, "OS")
        if path:
            st.session_state["os_image_path"] = path
            st.success(f"OS image saved: {Path(path).name}")

            if st.button("Detect OS ROI", key="detect_os"):
                with st.spinner("Detecting OS ROI..."):
                    try:
                        from kerascan import EngineConfig, KerascanEngine
                        engine = KerascanEngine()
                        img_arr = np.array(Image.open(path).convert("RGB"))
                        result = engine.analyze(img_arr)
                        roi = result.get("roi", {})
                        st.session_state["os_roi"] = roi
                        st.success(f"ROI detected: method={roi.get('method')} confidence={roi.get('confidence', 0):.2f}")
                        st.json(roi)
                    except Exception as e:
                        st.warning(f"ROI detection failed: {e}. Use manual adjustment below.")

            with st.expander("Manual ROI Adjustment"):
                try:
                    img_pil = Image.open(path)
                    w, h = img_pil.size
                except Exception:
                    w, h = 640, 480
                x0 = st.slider("OS x0", 0, w, 0, key="os_x0")
                y0 = st.slider("OS y0", 0, h, 0, key="os_y0")
                x1 = st.slider("OS x1", 0, w, w, key="os_x1")
                y1 = st.slider("OS y1", 0, h, h, key="os_y1")
                if st.button("Apply manual OS ROI"):
                    st.session_state["os_roi"] = {"box_xyxy": [x0, y0, x1, y1], "method": "manual"}
                    st.success("Manual OS ROI applied.")

st.divider()
od_ready = bool(st.session_state.get("od_image_path"))
os_ready = bool(st.session_state.get("os_image_path"))
if od_ready and os_ready:
    st.success("Both images uploaded. Proceed to measurements.")
    st.page_link("pages/03_measurements.py", label="→ Enter Measurements")
else:
    st.info(f"OD: {'✓' if od_ready else '✗'}  |  OS: {'✓' if os_ready else '✗'}  — Upload both images to continue.")
