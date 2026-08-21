# KERASCAN — Keratoconus Screening System (Phase 2)

An offline-first laptop screening application for keratoconus risk assessment. The system accepts left-eye (OS) and right-eye (OD) Placido KERASCAN images, keratometry, pachymetry, and refraction values; invokes the verified Phase 1 image-analysis engine through its public interface; applies transparent, auditable clinical referral rules; persists screening encounters locally in SQLite; and produces comprehensive multi-format exports (PDF, JSON, Excel).

> **Clinical Disclaimer:** AI-assisted keratoconus screening result. This is not a confirmed diagnosis. Positive, discordant, ungradable, or clinically concerning findings require repeat assessment, corneal tomography (e.g. Pentacam), and qualified clinical review.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Madhumasa84/Keratoconus.git
cd Keratoconus

# Install requirements
pip install -r app/requirements.txt
```

### 2. Launch the Application (One Documented Command)
```bash
streamlit run app/streamlit_app.py
```

### 3. Run Automated Test Suite
```bash
pytest app/tests/ -v
```

---

## 🏛 System Architecture

```text
app/
├── streamlit_app.py              # Main multi-page entry point & operator session
├── pages/
│   ├── 01_new_screening.py      # Demographic & encounter metadata collection
│   ├── 02_upload_images.py      # Independent OD/OS image upload & ROI review
│   ├── 03_measurements.py       # Keratometry, pachymetry, refraction entry & validation
│   ├── 04_analysis.py           # Pipeline execution & deterministic referral rules
│   ├── 05_review_findings.py    # Per-eye & child-level findings visualization
│   ├── 06_confirm_report.py     # Operator confirmation, auditable override & exports
│   ├── 07_search_history.py     # Search and filter previous screening records
│   └── 08_followup.py           # Pentacam / corneal tomography follow-up tracking
├── services/
│   ├── referral_engine.py       # Deterministic rule engine & versioned protocol
│   ├── screening_service.py     # End-to-end screening orchestration & validation
│   ├── report_service.py        # ReportLab 3-page PDF, JSON, & Excel exports
│   └── audit_service.py         # Immutable audit logging & override validation
├── database/
│   ├── models.py                # SQLAlchemy 2.0 ORM models (9 tables)
│   ├── repository.py            # Data access layer & query repository
│   └── migrations/
│       └── 001_initial.sql      # Raw SQL schema definition
├── templates/
│   └── report_styles.py         # Colorblind-safe palette (Navy/Amber) styling
├── tests/                       # Comprehensive pytest suite (72 unit tests)
└── config/
    └── referral_protocol.yaml   # Versioned clinical thresholds & rules
```

---

## 📋 Database Schema

The database uses SQLite with WAL mode and foreign-key constraints enabled:

1. **`screenings`**: Child/encounter metadata (`screening_id`, `age`, `sex`, `site`, `screening_date`, `operator_id`, `device_id`, `consent_recorded`, `overall_result`, `referral_priority`, `protocol_version`).
2. **`eyes`**: Per-eye records (`laterality` [OD/OS], `eye_result`, `reason_codes`, image paths/hashes, ROI bounding boxes, quality metrics, feature vectors).
3. **`measurements`**: Keratometry (K1, K2/steep K, Kmax, Mean K), Pachymetry (central/thinnest in µm), Refraction (sphere, cylinder, axis, autorefraction vs subjective), repeat readings (1, 2, 3).
4. **`image_analysis`**: Raw Phase 1 engine outputs, model hashes, prototype scores.
5. **`decisions`**: Eye and child-level decisions with automated results, reason codes, and full override tracking (`is_overridden`, `override_by`, `override_at`, `override_reason`, `override_original`).
6. **`referrals`**: Destination, priority, and clinical notes.
7. **`pentacam_followup`**: Pentacam follow-up records (Kmax OD/OS, Belin-Ambrósio D OD/OS, exam date).
8. **`audit_log`**: Immutable audit records for every INSERT, UPDATE, and DELETE.

---

## ⚖️ Referral Rule Matrix & Reason Codes

### Reason Codes:
* `IMG_SUSPICIOUS`: Suspicious Placido ring pattern detected by Phase 1 engine.
* `IMG_UNGRADABLE`: Ungradable image (never converted to normal).
* `K_HIGH`: Steep K2 ≥ 47.0 D.
* `PACHY_LOW`: Corneal pachymetry ≤ 480 µm.
* `CYL_HIGH`: Astigmatism cylinder magnitude ≥ 2.0 D.
* `TWO_DOMAIN_ABNORMAL`: Two or more quantitative domains abnormal.
* `CLINICAL_SIGN`: Slit-lamp clinical sign present (Vogt striae, Fleischer ring, etc.).
* `INTER_EYE_ASYMMETRY`: Inter-eye K2 difference ≥ 1.5 D.
* `REPEAT_REQUIRED`: Repeat measurement required to confirm isolated abnormality.
* `MEASUREMENT_MISSING`: Required clinical measurements incomplete.

### Output Decisions:
* `SCREEN_NEGATIVE`: All domains normal.
* `STANDARD_REFERRAL`: Suspicious image alone, confirmed isolated abnormality, or asymmetry.
* `PRIORITY_REFERRAL`: Suspicious image + quantitative abnormality, or multiple quantitative abnormalities.
* `RECAPTURE_REQUIRED`: Ungradable image requiring recapture.
* `INCOMPLETE`: Required fields missing.
* `MANUAL_REVIEW`: Inconclusive / unhandled cases.

---

## 📄 Multi-Page PDF Report Structure

* **Page 1 (Clinical Summary)**: Header with protocol version, patient/encounter metadata, OD & OS measurement table, per-eye decisions with reason codes, overall child-level decision, referral priority, and regulatory disclaimer.
* **Page 2 (Image Findings & Quality)**: Placido images, ROI detection parameters, quality score breakdown, quality flags.
* **Page 3 (Detailed Data & Audit)**: Repeat readings, algorithm versions & model hashes, decision audit trail, and Pentacam follow-up tracking section.
* **Accessibility**: Styled using a colorblind-safe palette (Navy Blue `#003366` and Amber `#FF8C00`).

---

## 🧪 Verification & Acceptance

Run the test suite:
```bash
pytest app/tests/ -v
```
All 72 tests covering every referral combination, edge case, input validation rule, database transaction, audit override, and export pipeline pass with 100% success.
