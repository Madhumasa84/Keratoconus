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

from sqlalchemy import create_engine, event
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
    """Create all tables (idempotent — safe to call on every startup)."""
    log.info("init_db: creating tables against %s", _db_url)
    Base.metadata.create_all(bind=engine)
    log.info("init_db: complete")


__all__ = ["engine", "SessionLocal", "Base", "init_db"]
