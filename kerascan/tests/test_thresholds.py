import numpy as np
from kerascan.geometry import compute_geometry

class TEST_ONLY_NON_CLINICAL_THRESHOLDS:
    suspicious_bounds = {"LOCAL_COMPRESSION": 0.8}
    indeterminate_bounds = {"LOCAL_COMPRESSION": 0.5}

def _test_config():
    class DummyConfig:
        thresholds = TEST_ONLY_NON_CLINICAL_THRESHOLDS()
    return DummyConfig()

def test_threshold_state_transitions():
    angles = np.array([0, 90, 180, 270])
    obs = np.ones((2, 4), dtype=bool)
    
    # Below threshold (0.0) -> Normal
    r1 = np.array([10.0, 10.0, 10.0, 10.0])
    r2 = np.array([20.0, 20.0, 20.0, 20.0])
    res1 = compute_geometry(np.vstack([r1, r2]), angles, obs, 0.0, _test_config(), expected_ring_count=2)
    assert res1["geometry_status"] == "NORMAL-LIKE"
    
    # Equal to indeterminate (0.5) -> Indeterminate
    r2_ind = np.array([20.0, 20.0, 15.0, 20.0]) # spacing 10, 10, 5, 10 -> min spacing 5 -> norm 0.5 -> comp 0.5
    res2 = compute_geometry(np.vstack([r1, r2_ind]), angles, obs, 0.0, _test_config(), expected_ring_count=2)
    assert res2["geometry_status"] == "INDETERMINATE"
    assert res2["features"]["LOCAL_COMPRESSION"] == 0.5
    
    # Equal to the test-only suspicious maximum, but confined to one ring pair:
    # it remains indeterminate because full-stack corroboration is absent.
    r2_susp = np.array([20.0, 20.0, 12.0, 20.0]) # spacing 10, 10, 2, 10 -> norm 0.2 -> comp 0.8
    res3 = compute_geometry(np.vstack([r1, r2_susp]), angles, obs, 0.0, _test_config(), expected_ring_count=2)
    assert res3["geometry_status"] == "INDETERMINATE"
    assert "uncorroborated_single_pair_maximum" in res3["reason_codes"]
    assert np.isclose(res3["features"]["LOCAL_COMPRESSION"], 0.8)
    
