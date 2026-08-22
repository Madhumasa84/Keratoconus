"""Regression: an unverified ring count alone can never unlock classification.

``classification_performed`` only becomes ``True`` when the engine is given
BOTH a verified ring count AND explicit ``GeometryThresholds`` (see
``geometry.py``); the engine itself defines no thresholds by default, so
supplying only a ring-count override -- without also opting into thresholds,
as ``app/services/screening_service.py`` now does for its own provisional,
non-clinical demo classifier -- must never be enough to produce a
NORMAL-LIKE/SUSPICIOUS classification. This test proves that narrower
invariant holds across every real image in ``normal/`` and ``suspicious/``,
not just synthetic unit fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kerascan.config import EngineConfig, RadialConfig
from kerascan.image_io import SUPPORTED_SUFFIXES
from kerascan.inference import KerascanEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
NORMAL_DIR = REPO_ROOT / "normal"
SUSPICIOUS_DIR = REPO_ROOT / "suspicious"


def _real_images() -> list[Path]:
    files: list[Path] = []
    for folder in (NORMAL_DIR, SUSPICIOUS_DIR):
        if folder.exists():
            files.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES))
    return files


_IMAGES = _real_images()
if not _IMAGES:
    pytest.skip("normal/ and suspicious/ folders not found", allow_module_level=True)


@pytest.mark.parametrize("path", _IMAGES, ids=lambda p: p.name)
def test_ring_count_override_alone_never_reaches_classification(path: Path) -> None:
    baseline = KerascanEngine().analyze(path)
    assert baseline["classification_performed"] is False
    assert baseline["screening_result"] not in {"NORMAL-LIKE", "SUSPICIOUS"}

    stack = baseline.get("full_stack_analysis") or {}
    detected = stack.get("detected_ring_count")
    if not detected:
        return

    override_engine = KerascanEngine(EngineConfig(radial=RadialConfig(expected_ring_count=detected, meridians=240)))
    overridden = override_engine.analyze(path)
    assert overridden["classification_performed"] is False
    assert overridden["screening_result"] not in {"NORMAL-LIKE", "SUSPICIOUS"}
    if overridden.get("gates", {}).get("verified_hardware_ring_count") == "PASS":
        # Once the ring-count gate is satisfied, the pipeline still never
        # classifies: it either reports the geometry honestly as uncalibrated
        # (no invented clinical thresholds) or, if full-stack coverage under
        # the override is insufficient, UNGRADABLE. Both are safe, non-classifying
        # outcomes; only an actual classification would violate the invariant.
        assert overridden["geometry_status"] in {"NOT_CALIBRATED", "UNGRADABLE"}
