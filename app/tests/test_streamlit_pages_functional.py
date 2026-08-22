"""Functional tests for the Streamlit wizard pages, using AppTest.

``test_streamlit_pages.py`` only checks that the page scripts compile. This
file actually drives the widgets and asserts on rendered output and session
state, exercising the real ScreeningService / ReferralEngine / ReportService
stack end to end.

Only the KerascanEngine.analyze() call is substituted for a deterministic
fake (the same pattern ``test_screening_service.py`` already uses), because
classification is not yet implemented anywhere in kerascan.inference — see
``kerascan/tests/test_classification_never_performed.py`` — so a real image
can never produce NORMAL_LIKE/SUSPICIOUS and the UI branches for those
outcomes could otherwise never be exercised.

Pages are always reached via ``AppTest.from_file(MAIN_APP).switch_page(...)``
rather than ``AppTest.from_file(page_path)`` directly: a standalone page
doesn't know about its sibling pages, so any ``st.page_link`` call on it
raises (missing page registry) unless navigation started from the main app.
"""
from __future__ import annotations

import importlib
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from app.services.auth_service import hash_password
from app.services.screening_service import ScreeningService

PAGES_DIR = Path(__file__).parent.parent / "pages"
MAIN_APP = Path(__file__).parent.parent / "streamlit_app.py"


def _raw_result(
    label="NORMAL-LIKE",
    *,
    classified=True,
    quality="ACCEPTABLE",
    segmentation="PASS",
    tracking="PASS",
    geometry="PASS",
    failure="NONE",
):
    return {
        "screening_result": label,
        "classification_performed": classified,
        "classification_skipped": not classified,
        "acquisition_quality": {"status": quality, "score": 80.0, "flags": [], "metrics": {}},
        "quality": {"status": quality, "gradable": quality == "ACCEPTABLE", "quality_score": 80.0, "flags": [], "metrics": {}},
        "roi": {"box_xyxy": [10, 10, 110, 110], "center_full": [60, 60], "outer_radius_px": 50, "confidence": 0.9, "method": "hough"},
        "segmentation": {"status": segmentation, "confidence": 0.9, "flags": []},
        "tracking": {"status": tracking, "confidence": 0.9, "flags": []},
        "geometry_validation": {"status": geometry},
        "failure_stage": failure,
        "pipeline_version": "synthetic-pipeline-1",
        "model": {"model_hash": "synthetic-model-hash"},
        "features": {"mean_ring_spacing": 0.1},
    }


class _FakeEngine:
    """Deterministic stand-in for KerascanEngine.analyze, keyed by file stem."""

    def __init__(self, by_name):
        self.by_name = by_name

    def analyze(self, source, output_dir=None):
        result = deepcopy(self.by_name.get(Path(source).stem, self.by_name.get("default")))
        if output_dir:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            for name, colour in (
                ("cropped_roi.png", "#4477aa"),
                ("cropped_roi_centres.png", "#aa7744"),
                ("tracked_rings_cartesian.png", "#44aa77"),
                ("directional_spacing.png", "#aa4477"),
            ):
                image = Image.new("RGB", (120, 80), colour)
                ImageDraw.Draw(image).text((5, 5), name, fill="white")
                image.save(output / name)
        return result


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point the DB and every local-file output at tmp_path, never the real home dir."""
    db_path = tmp_path / "kerascan_test.db"
    monkeypatch.setenv("KERASCAN_DB_URL", f"sqlite+pysqlite:///{db_path}")
    monkeypatch.setenv("KERASCAN_LOCAL_IMAGE_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("KERASCAN_LOCAL_OUTPUT_DIR", str(tmp_path / "outputs"))
    import app.database

    importlib.reload(app.database)
    app.database.init_db()
    yield tmp_path


@pytest.fixture
def fake_engine(monkeypatch):
    """Patch ScreeningService so every instance (including ones the pages create) uses a fake engine."""
    holder: dict[str, _FakeEngine] = {}

    def _get_image_engine(self):
        return holder["engine"]

    monkeypatch.setattr(ScreeningService, "_get_image_engine", _get_image_engine)

    def _configure(by_name):
        holder["engine"] = _FakeEngine(by_name)

    _configure({"default": _raw_result()})
    return _configure


@pytest.fixture
def sample_images(tmp_path):
    paths = {}
    for name in ("od_image", "os_image"):
        path = tmp_path / f"{name}.png"
        image = Image.new("RGB", (120, 120), "black")
        draw = ImageDraw.Draw(image)
        draw.ellipse((15, 15, 105, 105), outline="white", width=4)
        draw.text((4, 4), name, fill="white")
        image.save(path)
        paths[name] = path
    return paths


def _screening_form(screening_id="SCR-APP-001"):
    return {
        "screening_id": screening_id, "age": 14, "sex": "Female", "site": "Synthetic School",
        "screening_date": "2026-08-21", "operator_id": "OP-TEST", "device_id": "KERASCAN-SYNTH",
        "consent_recorded": True,
    }


def _main_app_authed():
    # The first app run in a worker pays a cold-start cost (database init,
    # engine import, protocol load) inside the timed window, which on a loaded
    # machine has exceeded smaller ceilings. The timeout only bounds a hang; it
    # costs nothing when the run is fast.
    at = AppTest.from_file(str(MAIN_APP), default_timeout=180)
    at.session_state["operator_authenticated"] = True
    at.session_state["operator_id"] = "OP-TEST"
    at.session_state["operator_role"] = "administrator"
    at.run()
    assert not at.exception
    return at


# ---------------------------------------------------------------------------
# Cross-cutting: auth gate on every page (checked standalone; the page stops
# before ever reaching a page_link call, so no navigation context is needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "page",
    [
        "01_new_screening.py", "02_upload_images.py", "03_measurements.py",
        "04_analysis.py", "05_review_findings.py", "06_confirm_report.py",
        "07_search_history.py",
    ],
)
def test_every_page_blocks_unauthenticated_operator(isolated_env, page):
    at = AppTest.from_file(str(PAGES_DIR / page), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Local operator login is required" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# Page 1 — New Screening
# ---------------------------------------------------------------------------

def _submit_new_screening(at, screening_id):
    at.text_input[0].set_value(screening_id)  # Screening ID
    at.number_input[0].set_value(14)  # Age
    at.selectbox[0].set_value("Female")  # Sex
    at.text_input[1].set_value("Synthetic School")  # Site
    at.date_input[0].set_value(date(2026, 8, 21))  # Screening date
    at.text_input[2].set_value("KERASCAN-SYNTH")  # Device ID
    at.checkbox[0].set_value(True)  # Consent
    at.button[0].click().run()


def test_new_screening_saves_valid_form_and_advances(isolated_env):
    at = _main_app_authed()
    at.switch_page("pages/01_new_screening.py").run()
    assert not at.exception

    _submit_new_screening(at, "SCR-APP-001")

    assert not at.exception
    assert any("saved" in s.value for s in at.success)
    assert at.session_state["current_screening"]["screening_id"] == "SCR-APP-001"
    assert at.session_state["current_step"] == 2


def test_new_screening_rejects_duplicate_id(isolated_env):
    from app.database import SessionLocal

    with SessionLocal() as session:
        svc = ScreeningService(db_session=session)
        result = svc.conduct_screening({
            "form": _screening_form("SCR-DUP-001"),
            "od_measurements": {}, "os_measurements": {},
        })
    assert result.success is True  # persisted, even though incomplete (no images)

    at = _main_app_authed()
    at.switch_page("pages/01_new_screening.py").run()
    _submit_new_screening(at, "SCR-DUP-001")

    assert not at.exception
    assert any("already exists" in e.value for e in at.error)
    assert at.session_state["current_screening"] == {}


# ---------------------------------------------------------------------------
# Page 2 — Upload Images
# ---------------------------------------------------------------------------

def test_upload_images_requires_step_one_first(isolated_env):
    at = AppTest.from_file(str(PAGES_DIR / "02_upload_images.py"), default_timeout=60)
    at.session_state["operator_authenticated"] = True
    at.run()
    assert any("Complete Step 1" in e.value for e in at.error)


def test_upload_images_rejects_identical_file_for_both_eyes(isolated_env, fake_engine, sample_images):
    at = _main_app_authed()
    at.session_state["current_screening"] = _screening_form()
    at.switch_page("pages/02_upload_images.py").run()
    assert not at.exception

    content = sample_images["od_image"].read_bytes()
    at.file_uploader[0].set_value(("same.png", content, "image/png"))
    at.file_uploader[1].set_value(("same.png", content, "image/png"))
    at.run()

    assert not at.exception
    assert any("same file was used for both eyes" in e.value for e in at.error)


def test_upload_page_is_never_a_dead_end_when_an_image_fails(isolated_env, fake_engine, sample_images):
    """An unusable image must warn but still offer a way forward.

    Blocking outright strands the operator with no route to the rest of the
    workflow. Continuing is safe because an eye whose image did not analyse is
    not a COMPLETE_IMAGE_STATE, so the referral engine keeps the encounter
    INCOMPLETE and it can never become a clean screen-negative.
    """
    # The upload page saves each eye as "<eye>_original<suffix>", so the fake
    # engine must be keyed on those stems rather than the uploaded filename.
    fake_engine({
        "od_original": _raw_result("UNGRADABLE", classified=False, tracking="FAIL", failure="TRACKING"),
        "os_original": _raw_result("NORMAL-LIKE"),
        "default": _raw_result("NORMAL-LIKE"),
    })
    at = _main_app_authed()
    at.session_state["current_screening"] = _screening_form()
    at.switch_page("pages/02_upload_images.py").run()
    at.file_uploader[0].set_value(("od_image.png", sample_images["od_image"].read_bytes(), "image/png"))
    at.file_uploader[1].set_value(("os_image.png", sample_images["os_image"].read_bytes(), "image/png"))
    at.run()

    assert not at.exception
    assert any("could not be analysed" in w.value for w in at.warning)
    assert any("recorded as incomplete" in w.value for w in at.warning)
    assert [link.label for link in at.get("page_link")] == ["Continue anyway →"]


def test_upload_images_accepts_two_distinct_images_and_links_to_measurements(isolated_env, fake_engine, sample_images):
    at = _main_app_authed()
    at.session_state["current_screening"] = _screening_form()
    at.switch_page("pages/02_upload_images.py").run()

    at.file_uploader[0].set_value(("od.png", sample_images["od_image"].read_bytes(), "image/png"))
    at.file_uploader[1].set_value(("os.png", sample_images["os_image"].read_bytes(), "image/png"))
    at.run()

    assert not at.exception
    assert at.session_state["od_image_verification"] is not None
    assert at.session_state["os_image_verification"] is not None
    assert any("Image analysed" in s.value for s in at.success)


# ---------------------------------------------------------------------------
# Page 3 — Measurements
# ---------------------------------------------------------------------------

def test_measurements_requires_images_first(isolated_env):
    at = AppTest.from_file(str(PAGES_DIR / "03_measurements.py"), default_timeout=60)
    at.session_state["operator_authenticated"] = True
    at.session_state["current_screening"] = _screening_form()
    at.run()
    assert any("Upload both required KeraScan images" in e.value for e in at.error)


_ABNORMAL_OD = {"k1_d": 43.0, "k2_d": 48.5, "pachymetry_um": 455.0, "cylinder_d": 2.25}
_NORMAL_OD = {"k1_d": 43.0, "k2_d": 44.0, "pachymetry_um": 530.0, "cylinder_d": 0.5}
_NORMAL_OS = {"k1_d": 43.1, "k2_d": 43.8, "pachymetry_um": 535.0, "cylinder_d": 0.5}


def _fill_measurements(at, od=_ABNORMAL_OD, os_=_NORMAL_OS):
    at.number_input(key="active_k1_OD").set_value(od["k1_d"])
    at.number_input(key="active_k2_OD").set_value(od["k2_d"])
    at.number_input(key="active_pachymetry_OD").set_value(od["pachymetry_um"])
    at.number_input(key="active_cylinder_OD").set_value(od["cylinder_d"])
    at.number_input(key="active_k1_OS").set_value(os_["k1_d"])
    at.number_input(key="active_k2_OS").set_value(os_["k2_d"])
    at.number_input(key="active_pachymetry_OS").set_value(os_["pachymetry_um"])
    at.number_input(key="active_cylinder_OS").set_value(os_["cylinder_d"])
    at.run()
    at.button[0].click().run()


def test_measurements_saves_and_advances(isolated_env, sample_images):
    at = _main_app_authed()
    at.session_state["current_screening"] = _screening_form()
    at.session_state["od_image_path"] = str(sample_images["od_image"])
    at.session_state["os_image_path"] = str(sample_images["os_image"])
    at.switch_page("pages/03_measurements.py").run()
    assert not at.exception

    _fill_measurements(at)

    assert not at.exception
    assert any("Simplified measurements saved" in s.value for s in at.success)
    assert at.session_state["od_measurements"]["k2_d"] == 48.5
    assert at.session_state["current_step"] == 4


# ---------------------------------------------------------------------------
# Full wizard: pages 1 -> 6 chained through switch_page
# ---------------------------------------------------------------------------

def _run_full_wizard(fake_engine, sample_images, *, od_result, os_result, od_measurements=_ABNORMAL_OD, os_measurements=_NORMAL_OS, screening_id="SCR-APP-E2E"):
    fake_engine({"od_image": od_result, "os_image": os_result, "default": od_result})

    at = _main_app_authed()

    at.switch_page("pages/01_new_screening.py").run()
    _submit_new_screening(at, screening_id)
    assert not at.exception
    assert at.session_state["current_screening"]["screening_id"] == screening_id

    at.switch_page("pages/02_upload_images.py").run()
    at.file_uploader[0].set_value(("od.png", sample_images["od_image"].read_bytes(), "image/png"))
    at.file_uploader[1].set_value(("os.png", sample_images["os_image"].read_bytes(), "image/png"))
    at.run()
    assert not at.exception

    at.switch_page("pages/03_measurements.py").run()
    _fill_measurements(at, od=od_measurements, os_=os_measurements)
    assert not at.exception

    at.switch_page("pages/04_analysis.py").run()
    at.button[0].click().run()
    assert not at.exception

    at.switch_page("pages/05_review_findings.py").run()
    assert not at.exception

    at.switch_page("pages/06_confirm_report.py").run()
    assert not at.exception
    return at


def test_full_wizard_with_real_current_pipeline_behaviour_ends_incomplete(isolated_env, fake_engine, sample_images):
    """Real images can never classify (see test_classification_never_performed.py), so an
    ANALYSIS_BLOCKED-shaped engine result must flow through the whole wizard as INCOMPLETE_SCREENING
    without crashing any page — this is what actually happens today for every real screening."""
    blocked = _raw_result("ANALYSIS_BLOCKED", classified=False, failure="CONFIGURATION", geometry="ANALYSIS_BLOCKED")
    at = _run_full_wizard(fake_engine, sample_images, od_result=blocked, os_result=blocked)

    result = at.session_state["analysis_result"]
    assert result.child_result.decision == "INCOMPLETE_SCREENING"
    assert any("Incomplete" in w.value for w in at.warning)

    at.checkbox(key="confirm_outcome").set_value(True).run()
    assert not at.exception
    assert any("only when a referral is needed" in c.value for c in at.caption)


def test_full_wizard_with_high_risk_screen_positive_offers_referral_pdf(isolated_env, fake_engine, sample_images):
    """A suspicious OD image plus abnormal OD measurements should reach a REFER decision and
    expose the referral-PDF export button on the confirm page."""
    at = _run_full_wizard(
        fake_engine, sample_images,
        od_result=_raw_result("SUSPICIOUS"), os_result=_raw_result("NORMAL-LIKE"),
    )

    result = at.session_state["analysis_result"]
    assert result.od_eye_result.action == "REFER"
    assert result.child_result.action == "REFER"
    assert any("Refer" in e.value for e in at.error)

    at.checkbox(key="confirm_outcome").set_value(True).run()
    assert not at.exception
    pdf_buttons = [b for b in at.button if b.label == "Referral letter (PDF)"]
    assert pdf_buttons, "Referral PDF button must be offered for a REFER outcome"
    pdf_buttons[0].click().run()
    assert not at.exception
    assert any("Download PDF" in d.label for d in at.download_button)


def test_full_wizard_with_screen_negative_offers_json_and_excel_only(isolated_env, fake_engine, sample_images):
    at = _run_full_wizard(
        fake_engine, sample_images,
        od_result=_raw_result("NORMAL-LIKE"), os_result=_raw_result("NORMAL-LIKE"),
        od_measurements=_NORMAL_OD, os_measurements=_NORMAL_OS,
    )
    result = at.session_state["analysis_result"]
    assert result.child_result.decision == "SCREEN_NEGATIVE"

    at.checkbox(key="confirm_outcome").set_value(True).run()
    assert not at.exception
    assert not [b for b in at.button if b.label == "Referral letter (PDF)"]

    excel_buttons = [b for b in at.button if b.label == "Screening record (Excel)"]
    assert excel_buttons
    excel_buttons[0].click().run()
    assert not at.exception
    assert any("Download Excel" in d.label for d in at.download_button)


# ---------------------------------------------------------------------------
# Page 7 — Search History
# ---------------------------------------------------------------------------

def test_search_history_finds_a_persisted_screening(isolated_env, fake_engine, sample_images):
    fake_engine({"default": _raw_result("NORMAL-LIKE")})
    from app.database import SessionLocal

    with SessionLocal() as session:
        svc = ScreeningService(db_session=session)
        result = svc.conduct_screening({
            "form": _screening_form("SCR-SEARCH-001"),
            "od_image_path": str(sample_images["od_image"]),
            "os_image_path": str(sample_images["os_image"]),
            "od_measurements": {"k1_d": 43.0, "k2_d": 44.0, "pachymetry_um": 530.0, "cylinder_d": 0.5},
            "os_measurements": {"k1_d": 43.1, "k2_d": 43.8, "pachymetry_um": 535.0, "cylinder_d": 0.5},
        })
        assert result.success is True

    at = _main_app_authed()
    at.switch_page("pages/07_search_history.py").run()
    assert not at.exception

    at.text_input[0].set_value("SCR-SEARCH-001").run()
    assert not at.exception
    assert any("1" in w.value for w in at.markdown if "result" in w.value)
    assert at.dataframe


def test_failed_eye_images_are_labelled_as_rejected_not_as_a_result(isolated_env, fake_engine, sample_images):
    """A rejected eye must not present its discarded working images as findings.

    The engine still writes intermediate artefacts for an eye it refused, and
    those "detected rings" can be tracing eyelashes. Showing them under normal
    captions would imply a measurement that was never accepted.
    """
    fake_engine({
        "od_original": _raw_result("UNGRADABLE", classified=False, tracking="FAIL", failure="TRACKING"),
        "os_original": _raw_result("NORMAL-LIKE"),
        "default": _raw_result("NORMAL-LIKE"),
    })
    at = _main_app_authed()
    at.session_state["current_screening"] = _screening_form()
    at.switch_page("pages/02_upload_images.py").run()
    at.file_uploader[0].set_value(("od_image.png", sample_images["od_image"].read_bytes(), "image/png"))
    at.file_uploader[1].set_value(("os_image.png", sample_images["os_image"].read_bytes(), "image/png"))
    at.run()
    at.switch_page("pages/03_measurements.py").run()
    _fill_measurements(at, od=_NORMAL_OD, os_=_NORMAL_OS)
    at.switch_page("pages/04_analysis.py").run()
    at.button[0].click().run()
    at.switch_page("pages/05_review_findings.py").run()
    assert not at.exception

    warnings = " ".join(w.value for w in at.warning)
    assert "not a result" in warnings
    assert "eyelashes" in warnings
    assert "upload a clearer photo" in warnings.lower()

    # The good eye still gets its normal, un-hedged comparison caption.
    captions = " ".join(c.value for c in at.caption)
    assert "Not a corneal map" in captions
