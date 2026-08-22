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
    assert result["geometry_status"] == "ANALYSIS_BLOCKED"
    assert result["full_stack_analysis"]["ring_count_verified"] is False
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


def test_oracle_mathematical_features():
    # 4 rings, 4 angles
    r1 = np.array([10.0, 10.0, 10.0, 10.0])
    r2 = np.array([20.0, 20.0, 20.0, 20.0])
    r3 = np.array([30.0, 30.0, 30.0, 30.0])
    r4 = np.array([40.0, 35.0, 40.0, 45.0]) # spacing is 10, 5, 10, 15
    
    radii = np.vstack([r1, r2, r3, r4])
    angles = np.array([0, 90, 180, 270])
    obs = np.ones_like(radii, dtype=bool)
    
    res = compute_geometry(radii, angles, obs, 0.0, _mock_config(), expected_ring_count=4)
    
    # Spacings:
    # r1->r2: 10, 10, 10, 10 -> median 10, MAD 0
    # r2->r3: 10, 10, 10, 10 -> median 10, MAD 0
    # r3->r4: 10, 5, 10, 15 -> median 10
    #   MAD for r3->r4: |10-10|=0, |5-10|=5, |10-10|=0, |15-10|=5 -> median of 0, 5, 0, 5 is 2.5
    # robust_var for r3->r4 = 2.5 / 10 = 0.25
    assert np.isclose(res["features"]["SPACING_VARIATION"], 0.25)
    
    # Opposite asymmetry:
    # r3->r4 spacings: 10, 5, 10, 15
    # angle 0 (10) vs 180 (10): diff = 0
    # angle 90 (5) vs 270 (15): diff = 10
    # max opposite asymmetry = 10 / 10 = 1.0
    assert np.isclose(res["features"]["OPPOSITE_ASYMMETRY"], 1.0)
    
    # Compression: max(0, 1 - normalized_spacing)
    # min spacing is 5. normalized = 5/10 = 0.5. compression = 1 - 0.5 = 0.5
    assert np.isclose(res["features"]["LOCAL_COMPRESSION"], 0.5)
    
    # Expansion: max(0, normalized_spacing - 1)
    # max spacing is 15. normalized = 15/10 = 1.5. expansion = 1.5 - 1.0 = 0.5
    assert np.isclose(res["features"]["LOCAL_EXPANSION"], 0.5)

def test_scale_invariance():
    angles = np.linspace(0, 360, 360, endpoint=False)
    r1 = np.full(360, 10.0); r1[0] -= 1
    r2 = np.full(360, 20.0); r2[0] -= 1
    radii_base = np.vstack([r1, r2])
    obs = np.ones_like(radii_base, dtype=bool)
    
    res_base = compute_geometry(radii_base, angles, obs, 0.0, _mock_config(), expected_ring_count=2)
    
    for scale in [0.5, 2.0, 4.0]:
        res_scaled = compute_geometry(radii_base * scale, angles, obs, 0.0, _mock_config(), expected_ring_count=2)
        assert np.isclose(res_base["features"]["SPACING_VARIATION"], res_scaled["features"]["SPACING_VARIATION"])
        assert np.isclose(res_base["features"]["OPPOSITE_ASYMMETRY"], res_scaled["features"]["OPPOSITE_ASYMMETRY"])
        assert np.isclose(res_base["features"]["LOCAL_COMPRESSION"], res_scaled["features"]["LOCAL_COMPRESSION"])
        
def test_rotation_equivariance():
    angles = np.linspace(0, 360, 360, endpoint=False)
    r1 = np.full(360, 10.0); r1[10:20] -= 1
    r2 = np.full(360, 20.0); r2[10:20] -= 2
    radii_base = np.vstack([r1, r2])
    obs = np.ones_like(radii_base, dtype=bool)
    
    res_base = compute_geometry(radii_base, angles, obs, 0.0, _mock_config(), expected_ring_count=2)
    
    # shift by 90 degrees (90 indices)
    radii_rot = np.roll(radii_base, 90, axis=1)
    res_rot = compute_geometry(radii_rot, angles, obs, 0.0, _mock_config(), expected_ring_count=2)
    
    assert np.isclose(res_base["features"]["SPACING_VARIATION"], res_rot["features"]["SPACING_VARIATION"])
    assert np.isclose(res_base["features"]["MULTIRING_AGREEMENT"], res_rot["features"]["MULTIRING_AGREEMENT"])
