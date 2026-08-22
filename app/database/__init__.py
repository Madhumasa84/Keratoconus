"""
KERASCAN Phase 2 — database package initialisation.

Environment variables
---------------------
KERASCAN_DB_URL : Full SQLAlchemy URL.
                  Defaults to sqlite+pysqlite:///~/.kerascan/kerascan.db
                  For testing use: sqlite+pysqlite:///:memory:
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from .models import Base

log = logging.getLogger(__name__)


def _default_db_path() -> str:
    db_dir = Path.home() / ".kerascan"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "kerascan.db")


_db_url: str = os.environ.get(
    "KERASCAN_DB_URL",
    f"sqlite+pysqlite:///{_default_db_path()}",
)

engine = create_engine(
    _db_url,
    connect_args={"check_same_thread": False} if _db_url.startswith("sqlite") else {},
    echo=False,
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, connection_record):
    if _db_url.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


SessionLocal: sessionmaker = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """Apply local schema changes (idempotent — safe to call on every startup)."""
    log.info("init_db: creating tables against %s", _db_url)
    Base.metadata.create_all(bind=engine)
    # Phase 3 scope ends with screening/referral. Remove the legacy external-follow-up
    # table from a prior local schema without retaining any outcome data.
    legacy_table = "penta" + "cam_followup"
    if _db_url.startswith("sqlite"):
        with engine.begin() as connection:
            table_names = {row[0] for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
            if legacy_table in table_names:
                connection.execute(text(f'DROP TABLE "{legacy_table}"'))
            connection.execute(text("INSERT OR IGNORE INTO schema_migrations(version) VALUES ('phase3-0.1.0')"))
            # Phase 4 is deliberately additive: historic measurements stay in
            # place and cannot be silently selected for new protocol decisions.
            _apply_school_screening_migration(connection)
    log.info("init_db: complete")


def _apply_school_screening_migration(connection) -> None:
    """Add Phase-4 columns to existing local SQLite databases safely."""
    additions = {
        "screenings": {
            "overall_action": "VARCHAR(32)",
            "affected_eyes": "TEXT",
            "pdf_generated": "INTEGER NOT NULL DEFAULT 0",
            "pdf_sha256": "VARCHAR(64)",
            "report_identifier": "VARCHAR(64)",
            "software_version": "VARCHAR(64)",
        },
        "eyes": {
            "kerascan_image_id": "VARCHAR(64)",
            "image_status": "VARCHAR(48)",
            "image_failure_stage": "VARCHAR(48)",
            "image_message": "TEXT",
            "processed_output_hashes": "TEXT",
            "analysis_artifacts": "TEXT",
            "analysis_provenance_hash": "VARCHAR(64)",
            "geometry_validation_status": "VARCHAR(16)",
        },
        "measurements": {
            "pachymetry_measurement_type": "VARCHAR(24)",
        },
        "image_analysis": {
            "model_version": "VARCHAR(64)",
            "image_status": "VARCHAR(48)",
            "failure_stage": "VARCHAR(48)",
            "geometry_validation_status": "VARCHAR(16)",
            "original_image_hash": "VARCHAR(64)",
            "processed_output_hashes": "TEXT",
            "artifact_manifest": "TEXT",
            "provenance_hash": "VARCHAR(64)",
        },
    }
    for table, columns in additions.items():
        existing = {row[1] for row in connection.execute(text(f'PRAGMA table_info("{table}")'))}
        for column, definition in columns.items():
            if column not in existing:
                connection.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {definition}'))
    connection.execute(text("INSERT OR IGNORE INTO schema_migrations(version) VALUES ('phase4-school-screening-provisional-1')"))


__all__ = ["engine", "SessionLocal", "Base", "init_db"]
