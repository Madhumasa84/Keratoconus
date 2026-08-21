-- KERASCAN Phase 2 — Migration 001: Initial schema
-- Applied to: SQLite >= 3.35
-- Date: 2026-08-21
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS screenings (
    id                  CHAR(36)     NOT NULL PRIMARY KEY,
    screening_id        VARCHAR(128) NOT NULL UNIQUE,
    age                 INTEGER,
    sex                 VARCHAR(16),
    site                VARCHAR(128),
    screening_date      VARCHAR(32),
    operator_id         VARCHAR(128),
    device_id           VARCHAR(128),
    consent_recorded    INTEGER      NOT NULL DEFAULT 0 CHECK (consent_recorded IN (0, 1)),
    overall_result      VARCHAR(32),
    referral_priority   VARCHAR(32),
    pdf_path            TEXT,
    json_path           TEXT,
    excel_path          TEXT,
    protocol_version    VARCHAR(32),
    created_at          DATETIME     NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW')),
    updated_at          DATETIME     NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW'))
);
CREATE INDEX IF NOT EXISTS ix_screenings_screening_id   ON screenings (screening_id);
CREATE INDEX IF NOT EXISTS ix_screenings_site           ON screenings (site);
CREATE INDEX IF NOT EXISTS ix_screenings_screening_date ON screenings (screening_date);
CREATE INDEX IF NOT EXISTS ix_screenings_created_at     ON screenings (created_at);
CREATE TRIGGER IF NOT EXISTS trg_screenings_updated_at AFTER UPDATE ON screenings FOR EACH ROW
BEGIN UPDATE screenings SET updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW') WHERE id = OLD.id; END;

CREATE TABLE IF NOT EXISTS eyes (
    id                      CHAR(36)  NOT NULL PRIMARY KEY,
    screening_id            CHAR(36)  NOT NULL REFERENCES screenings(id) ON DELETE CASCADE,
    laterality              VARCHAR(4) NOT NULL CHECK (laterality IN ('OD', 'OS')),
    eye_result              VARCHAR(32),
    reason_codes            TEXT,
    image_path              TEXT,
    image_hash              VARCHAR(64),
    processed_image_hash    VARCHAR(64),
    roi_box                 TEXT,
    roi_center              TEXT,
    roi_radius              REAL,
    roi_confidence          REAL,
    roi_method              VARCHAR(64),
    quality_gradable        INTEGER CHECK (quality_gradable IN (0, 1) OR quality_gradable IS NULL),
    quality_score           REAL,
    quality_flags           TEXT,
    quality_metrics         TEXT,
    features                TEXT,
    pipeline_version        VARCHAR(64),
    model_version           VARCHAR(64),
    protocol_version        VARCHAR(32),
    created_at              DATETIME  NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW'))
);
CREATE INDEX IF NOT EXISTS ix_eyes_screening_id ON eyes (screening_id);
CREATE INDEX IF NOT EXISTS ix_eyes_laterality   ON eyes (laterality);

CREATE TABLE IF NOT EXISTS measurements (
    id                  CHAR(36)    NOT NULL PRIMARY KEY,
    eye_id              CHAR(36)    NOT NULL REFERENCES eyes(id) ON DELETE CASCADE,
    k1_d                REAL, k1_axis REAL, k2_d REAL, k2_axis REAL,
    kmax_d              REAL, mean_k_d REAL,
    pachymetry_um       REAL,
    pachymetry_type     VARCHAR(16) CHECK (pachymetry_type IN ('central','thinnest') OR pachymetry_type IS NULL),
    sphere_d            REAL, cylinder_d REAL, cylinder_axis REAL,
    va_logmar           REAL, va_method VARCHAR(64),
    measurement_quality VARCHAR(16),
    reading_number      INTEGER CHECK (reading_number IN (1,2,3) OR reading_number IS NULL),
    refraction_type     VARCHAR(16) CHECK (refraction_type IN ('autorefraction','subjective') OR refraction_type IS NULL),
    clinical_flags      TEXT,
    created_at          DATETIME    NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW'))
);
CREATE INDEX IF NOT EXISTS ix_measurements_eye_id ON measurements (eye_id);

CREATE TABLE IF NOT EXISTS image_analysis (
    id                      CHAR(36) NOT NULL PRIMARY KEY,
    eye_id                  CHAR(36) NOT NULL REFERENCES eyes(id) ON DELETE CASCADE,
    engine_result           TEXT,
    screening_result        VARCHAR(32),
    prototype_score         REAL,
    classification_skipped  INTEGER CHECK (classification_skipped IN (0,1) OR classification_skipped IS NULL),
    model_hash              VARCHAR(64),
    pipeline_version        VARCHAR(64),
    created_at              DATETIME NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW'))
);
CREATE INDEX IF NOT EXISTS ix_image_analysis_eye_id ON image_analysis (eye_id);

CREATE TABLE IF NOT EXISTS decisions (
    id                      CHAR(36)  NOT NULL PRIMARY KEY,
    eye_id                  CHAR(36)  REFERENCES eyes(id) ON DELETE SET NULL,
    screening_id            CHAR(36)  NOT NULL REFERENCES screenings(id) ON DELETE CASCADE,
    decision_level          VARCHAR(8) NOT NULL CHECK (decision_level IN ('eye','child')),
    automated_result        VARCHAR(32),
    automated_reason_codes  TEXT,
    final_result            VARCHAR(32),
    is_overridden           INTEGER   NOT NULL DEFAULT 0 CHECK (is_overridden IN (0,1)),
    override_by             VARCHAR(128),
    override_at             DATETIME,
    override_original       VARCHAR(32),
    override_new            VARCHAR(32),
    override_reason         TEXT,
    protocol_version        VARCHAR(32),
    created_at              DATETIME  NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW'))
);
CREATE INDEX IF NOT EXISTS ix_decisions_screening_id ON decisions (screening_id);
CREATE INDEX IF NOT EXISTS ix_decisions_eye_id       ON decisions (eye_id);

CREATE TABLE IF NOT EXISTS referrals (
    id                  CHAR(36) NOT NULL PRIMARY KEY,
    screening_id        CHAR(36) NOT NULL REFERENCES screenings(id) ON DELETE CASCADE,
    referral_date       VARCHAR(32),
    referral_destination TEXT,
    referral_notes      TEXT,
    referral_priority   VARCHAR(32),
    created_at          DATETIME NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW'))
);
CREATE INDEX IF NOT EXISTS ix_referrals_screening_id ON referrals (screening_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id           CHAR(36)   NOT NULL PRIMARY KEY,
    table_name   VARCHAR(128) NOT NULL,
    record_id    VARCHAR(36)  NOT NULL,
    action       VARCHAR(8)   NOT NULL CHECK (action IN ('INSERT','UPDATE','DELETE')),
    old_value    TEXT,
    new_value    TEXT,
    performed_by VARCHAR(128),
    performed_at DATETIME     NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW'))
);
CREATE INDEX IF NOT EXISTS ix_audit_log_record_id  ON audit_log (record_id);
CREATE INDEX IF NOT EXISTS ix_audit_log_table_name ON audit_log (table_name);
CREATE INDEX IF NOT EXISTS ix_audit_log_performed_at ON audit_log (performed_at);
