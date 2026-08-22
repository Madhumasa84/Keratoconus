"""Tests for the privacy-preserving local folder verifier."""
from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "verify_local_folder.py"
SPEC = importlib.util.spec_from_file_location("kerascan_folder_verifier", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_even_meridian_radial_deviation_uses_mathematical_median():
    result = {
        "acquisition_quality": {},
        "segmentation": {},
        "tracking": {},
        "full_stack_analysis": {
            "radial_stack_deviation_by_meridian": [0.0, 1.0, 2.0, 3.0],
            "full_stack_summary": {},
        },
    }
    summary = MODULE._case_summary(result, "hash-only-id")
    assert summary["radial_stack_deviation_median"] == 1.5

