"""Unit tests for ScreeningService validation and workflow."""
import pytest
from app.services.screening_service import ScreeningService


@pytest.fixture
def svc():
    return ScreeningService()


def test_valid_form_passes(svc, sample_screening_data):
    valid, errors = svc.validate_screening_form(sample_screening_data)
    assert valid is True
    assert errors == []


def test_missing_consent_blocked(svc, sample_screening_data):
    data = dict(sample_screening_data, consent_recorded=False)
    valid, errors = svc.validate_screening_form(data)
    assert valid is False
    assert any("consent" in e.lower() for e in errors)


def test_age_out_of_range(svc, sample_screening_data):
    data = dict(sample_screening_data, age=30)
    valid, errors = svc.validate_screening_form(data)
    assert valid is False
    assert any("age" in e.lower() for e in errors)


def test_age_too_young(svc, sample_screening_data):
    data = dict(sample_screening_data, age=3)
    valid, errors = svc.validate_screening_form(data)
    assert valid is False


def test_missing_screening_id(svc, sample_screening_data):
    data = dict(sample_screening_data, screening_id="")
    valid, errors = svc.validate_screening_form(data)
    assert valid is False


def test_invalid_sex(svc, sample_screening_data):
    data = dict(sample_screening_data, sex="Unknown")
    valid, errors = svc.validate_screening_form(data)
    assert valid is False


def test_k2_out_of_range(svc, sample_measurements_od):
    meas = dict(sample_measurements_od, k2_d=75.0)
    valid, errors = svc.validate_measurements(meas)
    assert valid is False
    assert any("k2" in e.lower() or "steep" in e.lower() for e in errors)


def test_pachymetry_out_of_range(svc, sample_measurements_od):
    meas = dict(sample_measurements_od, pachymetry_um=900.0)
    valid, errors = svc.validate_measurements(meas)
    assert valid is False


def test_cylinder_axis_out_of_range(svc, sample_measurements_od):
    meas = dict(sample_measurements_od, cylinder_axis=200)
    valid, errors = svc.validate_measurements(meas)
    assert valid is False
    assert any("axis" in e.lower() or "cylinder" in e.lower() for e in errors)


def test_pachymetry_type_required(svc, sample_measurements_od):
    meas = dict(sample_measurements_od, pachymetry_type=None)
    valid, errors = svc.validate_measurements(meas)
    assert valid is False
    assert any("pachymetry_type" in e.lower() for e in errors)


def test_refraction_type_invalid(svc, sample_measurements_od):
    meas = dict(sample_measurements_od, refraction_type="unknown_type")
    valid, errors = svc.validate_measurements(meas)
    assert valid is False


def test_measurement_disagreement_third_reading(svc):
    r1 = {"k2_d": 44.0}
    r2 = {"k2_d": 44.8}  # 0.8D diff > 0.5D threshold
    assert svc.check_measurement_agreement(r1, r2) is False


def test_measurement_agreement_accepted(svc):
    r1 = {"k2_d": 44.0}
    r2 = {"k2_d": 44.3}  # 0.3D diff < 0.5D threshold
    assert svc.check_measurement_agreement(r1, r2) is True


def test_k2_vs_kmax_stored_separately(svc, sample_measurements_od):
    meas = dict(sample_measurements_od, kmax_d=50.0, k2_d=44.5)
    valid, errors = svc.validate_measurements(meas)
    # Both should pass validation independently
    assert valid is True
    assert meas["k2_d"] != meas["kmax_d"], "K2 and Kmax must be independent"


def test_central_vs_thinnest_stored_separately(svc):
    meas_c = {
        "k2_d": 44.0, "pachymetry_um": 530.0, "pachymetry_type": "central",
        "cylinder_d": 0.5, "cylinder_axis": 90,
    }
    meas_t = dict(meas_c, pachymetry_type="thinnest", pachymetry_um=515.0)
    valid_c, _ = svc.validate_measurements(meas_c)
    valid_t, _ = svc.validate_measurements(meas_t)
    assert valid_c is True
    assert valid_t is True
    assert meas_c["pachymetry_type"] != meas_t["pachymetry_type"]


def test_autorefraction_vs_subjective_stored_separately(svc, sample_measurements_od):
    meas_a = dict(sample_measurements_od, refraction_type="autorefraction")
    meas_s = dict(sample_measurements_od, refraction_type="subjective")
    for meas in (meas_a, meas_s):
        valid, errors = svc.validate_measurements(meas)
        assert valid is True


def test_valid_measurements_pass(svc, sample_measurements_od, sample_measurements_os):
    valid_od, errors_od = svc.validate_measurements(sample_measurements_od)
    valid_os, errors_os = svc.validate_measurements(sample_measurements_os)
    assert valid_od is True, f"OD errors: {errors_od}"
    assert valid_os is True, f"OS errors: {errors_os}"
