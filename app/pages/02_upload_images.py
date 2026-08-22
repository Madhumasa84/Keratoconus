"""Mandatory bilateral KeraScan upload and image-gate page."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kerascan" / "src"))

import streamlit as st

from app.services.ui_security import require_authenticated

st.set_page_config(page_title="Upload Images — KeraScan", layout="wide")
require_authenticated(st)
st.title("Upload Images")
st.caption("Upload one image for each eye.")
with st.expander("How to take a good photo"):
    st.markdown(
        """
1. **Fill the frame with the rings** — move close; the ring pattern should be the
   largest thing in the picture, not a small circle in a face photo.
2. **Hold the device flat and square to the eye.** Tilting makes the round rings
   photograph as ovals, which looks like corneal irregularity when it is not.
3. **Hold the upper eyelid open** and ask the child to look straight at the centre.
   Eyelashes covering the rings are the most common reason a photo cannot be used.
4. **Avoid glare** from windows and overhead lights, and hold steady.
"""
    )

if not st.session_state.get("current_screening"):
    st.error("Complete Step 1 (New Screening) first.")
    st.stop()

from app.services.screening_service import ScreeningService

screening = st.session_state["current_screening"]
screening_id = screening.get("screening_id", "unknown")
image_dir = Path(os.environ.get("KERASCAN_LOCAL_IMAGE_DIR", str(Path.home() / ".kerascan" / "images"))) / screening_id
image_dir.mkdir(parents=True, exist_ok=True)
analysis_root = image_dir / "analysis"
ALLOWED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]


_READY_MESSAGE = {
    "NORMAL_LIKE": "Image analysed.",
    "SUSPICIOUS": "Image analysed.",
    "INDETERMINATE": "Image analysed.",
}

_RETAKE_MESSAGE = {
    "IMAGE_REJECTED": "This image could not be used.",
    "SEGMENTATION_FAILED": "The ring pattern could not be found in this photo.",
    "TRACKING_FAILED": "Too much of the ring pattern is hidden in this photo.",
    "ANALYSIS_BLOCKED": "This image could not be analysed.",
}

# What the operator should physically change, keyed on why the image failed.
# Ordered by how much difference the correction makes.
_RETAKE_ACTION = (
    ("no_placido_pattern_located", "Point the device at the cornea — the rings were not found in this photo."),
    ("placido_pattern_too_small", "Move the device closer so the rings fill the frame."),
    ("pattern_off_centre", "Centre the ring pattern in the frame."),
    ("possible_eyelid_or_eyelash_obstruction", "Hold the upper eyelid open and ask the child to look straight at the centre."),
    ("glare_or_saturation", "Reduce glare — angle away from windows and overhead lights."),
    ("underexposed", "More light is needed on the eye."),
    ("blur", "Hold the device steady and retake."),
    ("mild_blur", "Hold the device steady and retake."),
    ("low_contrast", "Increase lighting contrast on the ring pattern."),
    ("sensor_noise", "More light is needed; the image is noisy."),
)


def _retake_actions(verification) -> list[str]:
    """Concrete corrections for this photo, derived from why it actually failed."""
    raw = verification.raw_result or {}
    flags = set((raw.get("acquisition_quality") or {}).get("flags") or [])
    flags |= set((raw.get("segmentation") or {}).get("flags") or [])
    flags |= set((raw.get("tracking") or {}).get("flags") or [])
    actions = [text for flag, text in _RETAKE_ACTION if flag in flags]
    if not actions:
        # Partial ring coverage is the common failure: the rings are there but
        # the lid, lashes or framing hide too much of them.
        actions.append(
            "Move closer so the rings fill the frame, hold the device flat and square to the eye, "
            "and hold the upper eyelid open."
        )
    return actions


def _status_text(verification) -> None:
    """Show whether this eye is ready, without exposing internal pipeline stages."""
    status = verification.image_status
    if status in _READY_MESSAGE:
        st.success(_READY_MESSAGE[status])
        return
    st.error(_RETAKE_MESSAGE.get(status, "This image could not be used."))
    for action in _retake_actions(verification):
        st.write(f"- {action}")


def _save_and_verify(uploaded, eye: str):
    suffix = Path(uploaded.name).suffix.lower()
    content = uploaded.getvalue()
    content_hash = hashlib.sha256(content).hexdigest()
    hash_key = f"{eye.lower()}_upload_content_hash"
    path_key = f"{eye.lower()}_image_path"
    verification_key = f"{eye.lower()}_image_verification"
    destination = image_dir / f"{eye.lower()}_original{suffix}"
    if st.session_state.get(hash_key) != content_hash or not destination.exists():
        destination.write_bytes(content)
        service = ScreeningService()
        verification = service.verify_image(destination, eye, analysis_root / eye)
        st.session_state[hash_key] = content_hash
        st.session_state[path_key] = str(destination)
        st.session_state[verification_key] = verification
        # Any new image invalidates a prior child-level result and confirmation.
        st.session_state["analysis_result"] = None
        st.session_state["decision_confirmed"] = False
    return st.session_state.get(verification_key)


columns = st.columns(2)
for column, eye in ((columns[0], "OD"), (columns[1], "OS")):
    with column:
        st.subheader("Right eye (OD)" if eye == "OD" else "Left eye (OS)")
        uploaded = st.file_uploader("Choose image", type=ALLOWED_TYPES, key=f"{eye.lower()}_upload")
        if uploaded is None:
            continue
        st.image(uploaded, use_container_width=True)
        verification = _save_and_verify(uploaded, eye)
        if verification is not None:
            _status_text(verification)

st.divider()
od = st.session_state.get("od_image_verification")
os = st.session_state.get("os_image_verification")
if od is None or os is None:
    st.caption("Upload an image for each eye to continue.")
elif od.original_image_hash == os.original_image_hash:
    st.error("The same file was used for both eyes. Upload the correct image for each eye.")
else:
    unread = [
        "right" if eye is od else "left"
        for eye in (od, os)
        if eye.image_status not in _READY_MESSAGE
    ]
    if unread:
        # Never a dead end: replacing the photo is the better option, but the
        # operator can still record measurements and finish. An eye whose image
        # did not analyse keeps the encounter INCOMPLETE in the referral engine,
        # so continuing can never turn into a clean screen-negative.
        st.warning(
            f"The {' and '.join(unread)} eye image could not be analysed. "
            "Uploading a clearer photo is recommended. You can still continue, "
            "but the screening will be recorded as incomplete."
        )
        st.page_link("pages/03_measurements.py", label="Continue anyway →")
    else:
        st.page_link("pages/03_measurements.py", label="Next: enter measurements →")
