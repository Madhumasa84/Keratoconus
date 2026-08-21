"""
Pytest conftest for KERASCAN Phase 2 tests.
All fixtures are self-contained and require no internet access or real images.
"""
import sys
import os
from pathlib import Path

# Make Phase 1 and Phase 2 importable
REPO_ROOT = Path(__file__).parent.parent.parent
PHASE1_SRC = REPO_ROOT / "kerascan" / "src"
APP_DIR = REPO_ROOT / "app"
for p in (str(REPO_ROOT), str(PHASE1_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def make_placido_image(size: int = 480, n_rings: int = 8, center_offset=(0, 0), irregular: bool = False) -> np.ndarray:
    """Draw concentric gray rings on a black background."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = size // 2 + center_offset[0]
    cy = size // 2 + center_offset[1]
    for i in range(1, n_rings + 1):
        r = int(i * (size // 2 - 20) / n_rings)
        color = (200 - i * 10, 200 - i * 10, 200 - i * 10)
        if irregular and i % 3 == 0:
            r += np.random.randint(-20, 20)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=color, width=3)
    return np.array(img)


def make_suspicious_image(size: int = 480) -> np.ndarray:
    """Irregular, off-center rings simulating a suspicious pattern."""
    return make_placido_image(size, n_rings=6, center_offset=(50, 30), irregular=True)


def make_blank_image(size: int = 480) -> np.ndarray:
    """Blank black image — will be UNGRADABLE."""
    return np.zeros((size, size, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine_config():
    from kerascan import EngineConfig
    return EngineConfig()


@pytest.fixture(scope="session")
def kerascan_engine(engine_config):
    from kerascan import KerascanEngine
    eng = KerascanEngine(engine_config)
    eng.fit_synthetic_baseline(n=200)
    return eng


@pytest.fixture
def sample_normal_image():
    return make_placido_image(480, n_rings=8)


@pytest.fixture
def sample_suspicious_image():
    return make_suspicious_image(480)


@pytest.fixture
def blank_image():
    return make_blank_image(480)


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_kerascan.db"
    os.environ["KERASCAN_DB_URL"] = f"sqlite+pysqlite:///{db_path}"
    # Reload the module so it picks up the env var
    import importlib
    import app.database
    importlib.reload(app.database)
    from app.database import init_db
    init_db()
    yield str(db_path)
    del os.environ["KERASCAN_DB_URL"]


@pytest.fixture
def db_session(tmp_db):
    from app.database import SessionLocal
    with SessionLocal() as s:
        yield s


@pytest.fixture
def sample_screening_data():
    return {
        "screening_id": "TEST-001",
        "age": 15,
        "sex": "Male",
        "site": "Test School",
        "screening_date": "2026-08-21",
        "operator_id": "OP01",
        "device_id": "DEV01",
        "consent_recorded": True,
    }


@pytest.fixture
def sample_measurements_od():
    return {
        "k1_d": 43.0, "k1_axis": 180,
        "k2_d": 44.5, "k2_axis": 90,
        "pachymetry_um": 530.0, "pachymetry_type": "central",
        "sphere_d": -1.0, "cylinder_d": 0.5, "cylinder_axis": 90,
        "refraction_type": "autorefraction",
        "measurement_quality": "Good",
        "reading_number": 1,
    }


@pytest.fixture
def sample_measurements_os():
    return {
        "k1_d": 42.5, "k1_axis": 175,
        "k2_d": 44.0, "k2_axis": 85,
        "pachymetry_um": 545.0, "pachymetry_type": "central",
        "sphere_d": -0.75, "cylinder_d": 0.25, "cylinder_axis": 85,
        "refraction_type": "autorefraction",
        "measurement_quality": "Good",
        "reading_number": 1,
    }


@pytest.fixture
def normal_engine_result():
    return {
        "screening_result": "NORMAL-LIKE",
        "prototype_score": 0.2,
        "classification_skipped": False,
        "quality": {"gradable": True, "quality_score": 72.0, "flags": [], "metrics": {}},
        "roi": {"box_xyxy": [50, 50, 430, 430], "center_full": [240, 240], "outer_radius_px": 190.0, "confidence": 0.91, "method": "hough"},
        "features": {"detected_ring_count": 8, "mean_spacing": 22.0},
        "pipeline_version": "phase1-0.1.0",
        "model": {"model_hash": "abc123", "pipeline_version": "phase1-0.1.0"},
    }


@pytest.fixture
def suspicious_engine_result():
    return {
        "screening_result": "SUSPICIOUS",
        "prototype_score": 0.82,
        "classification_skipped": False,
        "quality": {"gradable": True, "quality_score": 55.0, "flags": ["irregular_spacing"], "metrics": {}},
        "roi": {"box_xyxy": [50, 50, 430, 430], "center_full": [240, 240], "outer_radius_px": 190.0, "confidence": 0.78, "method": "hough"},
        "features": {"detected_ring_count": 6},
        "pipeline_version": "phase1-0.1.0",
        "model": {"model_hash": "def456", "pipeline_version": "phase1-0.1.0"},
    }


@pytest.fixture
def ungradable_engine_result():
    return {
        "screening_result": "UNGRADABLE",
        "message": "recapture required",
        "classification_skipped": True,
        "quality": {"gradable": False, "quality_score": 10.0, "flags": ["low_contrast", "low_ring_fraction"], "metrics": {}},
        "roi": {},
        "features": {},
        "pipeline_version": "phase1-0.1.0",
    }


@pytest.fixture
def referral_engine():
    from app.services.referral_engine import ReferralEngine
    return ReferralEngine()
