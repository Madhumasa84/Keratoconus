"""Deliberately unavailable physical-topography placeholder.

Arc-Step calculation is outside Phase 1. It requires validated KERASCAN camera
intrinsics, working distance, Placido target geometry, and calibration data.
"""

def arc_step_topography(*_args, **_kwargs):
    raise NotImplementedError(
        "Arc-Step physical topography is not implemented: calibrated KERASCAN "
        "camera parameters, working distance, and Placido geometry are required."
    )
