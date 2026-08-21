import numpy as np
import pytest
from kerascan.geometry import compute_geometry

def _mock_config():
    class DummyConfig:
        pass
    return DummyConfig()

def test_perfect_rings():
    radii = np.array([[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]], dtype=float)
    angles_deg = np.array([0, 90, 180, 270], dtype=float)
    observed = np.ones_like(radii, dtype=bool)
    
    result = compute_geometry(radii, angles_deg, observed, 0.0, _mock_config())
    assert result["geometry_status"] == "NOT_CALIBRATED"
    features = result["features"]
    
    # Perfect rings have 0 variation
    assert features["SPACING_VARIATION"] == 0.0
    assert features["OPPOSITE_ASYMMETRY"] == 0.0
    assert features["LOCAL_COMPRESSION"] == 0.0
    assert features["LOCAL_EXPANSION"] == 0.0
    assert features["MULTIRING_AGREEMENT"] == 0.0

def test_missing_sectors():
    radii = np.array([[10, 10, np.nan, 10], [20, 20, np.nan, 20], [30, 30, 30, 30]], dtype=float)
    angles_deg = np.array([0, 90, 180, 270], dtype=float)
    observed = np.isfinite(radii)
    
    result = compute_geometry(radii, angles_deg, observed, 0.0, _mock_config())
    assert result["features"]["SPACING_VARIATION"] == 0.0

def test_non_positive_spacing():
    radii = np.array([[10, 10, 10, 10], [20, 5, 20, 20]], dtype=float) # 5 < 10, so spacing is -5
    angles_deg = np.array([0, 90, 180, 270], dtype=float)
    observed = np.ones_like(radii, dtype=bool)
    
    result = compute_geometry(radii, angles_deg, observed, 0.0, _mock_config())
    assert result["geometry_status"] == "UNGRADABLE"
    assert "non_positive_spacing" in result["reason_codes"]

def test_elliptical_rings():
    angles = np.linspace(0, 360, 36, endpoint=False)
    r1 = 10 + 2 * np.cos(np.deg2rad(angles))
    r2 = 20 + 4 * np.cos(np.deg2rad(angles))
    radii = np.vstack([r1, r2])
    observed = np.ones_like(radii, dtype=bool)
    
    result = compute_geometry(radii, angles, observed, 0.0, _mock_config())
    
    # Not purely circular, so there will be some variation if not proportional,
    # but since r2 = 2 * r1, the spacing is r2 - r1 = 10 + 2*cos, so not constant.
    assert result["features"]["SPACING_VARIATION"] > 0

