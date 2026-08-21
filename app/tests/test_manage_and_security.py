"""
Tests for app/manage.py CLI operations and ui_security helper.
"""
import os
from pathlib import Path
import pytest
from app.manage import main as manage_main
from app.services.ui_security import require_authenticated


def test_manage_migrate_and_health(tmp_db):
    """Test manage.py migrate and health actions."""
    res_migrate = manage_main(["migrate"])
    assert res_migrate == 0

    res_health = manage_main(["health"])
    assert res_health == 0


def test_manage_create_user(tmp_db):
    """Test manage.py create-user action."""
    res = manage_main([
        "create-user",
        "--operator-id", "test_admin",
        "--role", "administrator",
        "--password", "AdminSecret123!",
    ])
    assert res == 0


def test_manage_backup_and_restore(tmp_db, tmp_path):
    """Test manage.py backup and restore actions."""
    backup_file = str(tmp_path / "test_backup.db")
    res_backup = manage_main(["backup", "--output", backup_file])
    assert res_backup == 0
    assert Path(backup_file).exists()

    res_restore = manage_main(["restore", "--input", backup_file])
    assert res_restore == 0


def test_ui_security_guard():
    """Test require_authenticated helper."""
    class FakeStreamlit:
        def __init__(self, authenticated=False):
            self.session_state = {"operator_authenticated": authenticated}
            self.stopped = False
            self.error_msg = None

        def error(self, msg):
            self.error_msg = msg

        def stop(self):
            self.stopped = True

    st_unauth = FakeStreamlit(authenticated=False)
    require_authenticated(st_unauth)
    assert st_unauth.stopped is True
    assert st_unauth.error_msg is not None

    st_auth = FakeStreamlit(authenticated=True)
    require_authenticated(st_auth)
    assert st_auth.stopped is False
