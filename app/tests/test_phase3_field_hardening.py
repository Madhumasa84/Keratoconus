from pathlib import Path
import os
import sqlite3
import pytest

from app.services.auth_service import LocalAuthService, hash_password, verify_password
from app.services.operations_service import backup_database, restore_database, storage_health
from app.services.deidentified_export_service import export_deidentified


def test_local_authentication_and_role_restrictions(db_session):
    auth=LocalAuthService(db_session)
    auth.create_account("field_operator","long-local-passphrase","operator")
    db_session.commit()
    assert auth.authenticate("field_operator","long-local-passphrase")["role"]=="operator"
    assert auth.authenticate("field_operator","wrong-password") is None
    with pytest.raises(PermissionError): auth.require_role("operator","reviewer","administrator")


def test_backup_restore_and_storage_warning(tmp_path):
    db=tmp_path/"source.db"; sqlite3.connect(db).execute("CREATE TABLE audit (id INTEGER)").connection.commit()
    url=f"sqlite+pysqlite:///{db}";backup=backup_database(url,tmp_path/"backup.db")
    assert backup.exists()
    sqlite3.connect(db).execute("DROP TABLE audit").connection.commit()
    restore_database(url,backup)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='audit'").fetchone()
    assert "warning" in storage_health(db,minimum_free_bytes=10**30)


def test_no_prohibited_followup_schema(tmp_db):
    with sqlite3.connect(tmp_db) as connection:
        names={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not any(name == "penta"+"cam_followup" for name in names)


def test_deidentified_export_removes_source_paths_and_direct_ids(tmp_path):
    path=export_deidentified([{"screening_id":"PRIVATE-001","image_path":"/private/a.png","age":12,"overall_result":"SCREEN_NEGATIVE"}],tmp_path/"export.json")
    text=Path(path).read_text()
    assert "PRIVATE-001" not in text and "/private" not in text and '"deidentified": true' in text
