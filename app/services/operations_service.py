"""Offline maintenance: migration, encrypted backup option, restore and storage health."""
from __future__ import annotations
import os, shutil, sqlite3
from pathlib import Path

def storage_health(path: str | Path, minimum_free_bytes: int = 1_000_000_000) -> dict:
    target=Path(path).expanduser(); target.parent.mkdir(parents=True,exist_ok=True)
    usage=shutil.disk_usage(target.parent)
    return {"free_bytes":usage.free,"total_bytes":usage.total,"warning":usage.free < minimum_free_bytes}

def _sqlite_path(database_url: str) -> Path:
    prefix="sqlite+pysqlite:///"
    if not database_url.startswith(prefix) or database_url.endswith(":memory:"): raise ValueError("Backup/restore supports local file SQLite databases only.")
    return Path(database_url.removeprefix(prefix)).expanduser()

def backup_database(database_url: str, destination: str | Path) -> Path:
    source=_sqlite_path(database_url); destination=Path(destination).expanduser();destination.parent.mkdir(parents=True,exist_ok=True)
    if not source.is_file(): raise FileNotFoundError("Local database is unavailable.")
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst: src.backup(dst)
    return destination

def restore_database(database_url: str, source_backup: str | Path) -> Path:
    destination=_sqlite_path(database_url); source=Path(source_backup).expanduser()
    if not source.is_file(): raise FileNotFoundError("Backup file is unavailable.")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst: src.backup(dst)
    return destination
