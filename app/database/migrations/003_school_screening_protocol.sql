-- KERASCAN Phase 4: bilateral image verification and simplified study inputs.
--
-- This migration is intentionally additive. It preserves all legacy readings
-- and deprecated measurement columns; new screening decisions read only the
-- canonical K1, K2, pachymetry, and cylinder fields.
--
-- SQLite ALTER TABLE lacks portable ADD COLUMN IF NOT EXISTS support. The
-- idempotent column checks are executed by app.database.init_db before this
-- migration version is recorded in schema_migrations.

-- screenings: referral PDF audit uses a local hash rather than an external URL
--   overall_action VARCHAR(32)
--   affected_eyes TEXT
--   pdf_generated INTEGER NOT NULL DEFAULT 0
--   pdf_sha256 VARCHAR(64)
--   report_identifier VARCHAR(64)
--   software_version VARCHAR(64)

-- eyes: bind a labelled source image and the exact processed artefacts to eye
--   kerascan_image_id VARCHAR(64)
--   image_status VARCHAR(48)
--   image_failure_stage VARCHAR(48)
--   image_message TEXT
--   processed_output_hashes TEXT
--   analysis_artifacts TEXT
--   analysis_provenance_hash VARCHAR(64)
--   geometry_validation_status VARCHAR(16)

-- measurements: configured once per study, not selected independently per eye
--   pachymetry_measurement_type VARCHAR(24)

-- image_analysis: per-run immutable provenance
--   model_version VARCHAR(64)
--   image_status VARCHAR(48)
--   failure_stage VARCHAR(48)
--   geometry_validation_status VARCHAR(16)
--   original_image_hash VARCHAR(64)
--   processed_output_hashes TEXT
--   artifact_manifest TEXT
--   provenance_hash VARCHAR(64)
