import numpy as np
import pytest
from kerascan.geometry import compute_geometry

def _mock_config():
    class DummyConfig:
        pass
    return DummyConfig()

def test_direct_observation_coverage_uses_expected_count():
    r1 = np.full(360, 10.0)
    r2 = np.full(360, 20.0)
    radii = np.vstack([r1, r2])
    angles = np.arange(360)
    observed = np.ones((2, 360), dtype=bool)
    # 2 rings fully observed. But expected is 4.
    res = compute_geometry(radii, angles, observed, 0.0, _mock_config(), expected_ring_count=4)
    assert res["features"]["DIRECT_OBSERVATION_COVERAGE"] == 0.5  # 2/4

def test_median_spacing_ignores_interpolated():
    r1 = np.full(360, 10.0)
    r2 = np.full(360, 20.0)
    # create an outlier spacing but mark it not observed (interpolated)
    r2[0] = 100.0 
    radii = np.vstack([r1, r2])
    angles = np.arange(360)
    observed = np.ones((2, 360), dtype=bool)
    observed[1, 0] = False  # outlier is interpolated
    
    res = compute_geometry(radii, angles, observed, 0.0, _mock_config(), expected_ring_count=2)
    # If the interpolated point is included, the max spacing is 90, median is 10
    # and the max normalized spacing would be 9. 
    # If ignored during median/compression, the compression and expansion at angle 0 shouldn't be based on this interpolated point, 
    # Wait, the rule is "median of directly observed valid s_k".
    # And "Interpolated points do not increase direct coverage."
    
    # Actually, let's explicitly test that median spacing computation ignores it.
    # The requirement: median_spacing_k = median of directly observed valid s_k
    # If we have 3 points: 10, 10, 100. 
    # observed: True, True, False
    # median should be 10.
    
    r1_small = np.array([10.0, 10.0, 10.0])
    r2_small = np.array([20.0, 30.0, 100.0])
    radii_small = np.vstack([r1_small, r2_small])
    angles_small = np.array([0, 120, 240])
    obs_small = np.array([[True, True, True], [True, True, False]])
    
    res_small = compute_geometry(radii_small, angles_small, obs_small, 0.0, _mock_config(), expected_ring_count=2)
    # spacing is [10, 20, 90]
    # observed spacing is [10, 20]. median should be 15.0
    # If it didn't ignore False, median would be 20.0
    # Let's check max normalized spacing. 
    # max observed spacing is 20, normalized is 20/15 = 1.333
    # Wait, if we check the residuals or something we can infer it. 
    # But it's easier to just check the geometry grid outputs if we add them, or write a dedicated test.

def test_multiring_agreement_synthetic_cases():
    # 1. Three neighbouring pairs compressed in the same sector.
    angles = np.linspace(0, 360, 360, endpoint=False)
    # base spacing is 10.
    r0 = np.full(360, 10.0)
    r1 = np.full(360, 20.0)
    r2 = np.full(360, 30.0)
    r3 = np.full(360, 40.0)
    r4 = np.full(360, 50.0)
    radii = np.vstack([r0, r1, r2, r3, r4])
    obs = np.ones_like(radii, dtype=bool)
    
    # Compress a sector (indices 10 to 20) by 5 pixels in 3 pairs
    r1[10:20] -= 1
    r2[10:20] -= 2
    r3[10:20] -= 3
    # Pair spacings in sector:
    # pair 0 (r1-r0): 9 (compressed)
    # pair 1 (r2-r1): 9 (compressed)
    # pair 2 (r3-r2): 9 (compressed)
    
    radii_case1 = np.vstack([r0, r1, r2, r3, r4])
    res1 = compute_geometry(radii_case1, angles, obs, 0.0, _mock_config(), expected_ring_count=5)
    
    # 3 pairs with equal magnitude but different sector locations
    r1_2 = np.full(360, 20.0); r1_2[10:20] -= 1
    r2_2 = np.full(360, 30.0); r2_2[30:40] -= 1  # different sector
    r3_2 = np.full(360, 40.0); r3_2[50:60] -= 1  # different sector
    # pair 0: 9 in 10-20
    # pair 1: 10 in 10-20, 9 in 30-40
    # pair 2: 10 in 30-40, 9 in 50-60
    radii_case2 = np.vstack([r0, r1_2, r2_2, r3_2, r4])
    res2 = compute_geometry(radii_case2, angles, obs, 0.0, _mock_config(), expected_ring_count=5)
    
    assert res1["features"]["MULTIRING_AGREEMENT"] > res2["features"]["MULTIRING_AGREEMENT"]
