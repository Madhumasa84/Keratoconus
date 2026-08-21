"""Offline local account authentication with PBKDF2 hashes and role checks."""
from __future__ import annotations
import base64
import hashlib
import hmac
import os

ROLES = frozenset({"operator", "reviewer", "administrator"})

def hash_password(password: str, *, iterations: int = 600_000, salt: bytes | None = None) -> str:
    if len(password) < 12: raise ValueError("Passwords must be at least 12 characters.")
    salt = salt or os.urandom(16)
    digest=hashlib.pbkdf2_hmac("sha256",password.encode("utf-8"),salt,iterations)
    return f"pbkdf2_sha256${iterations}${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"

def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme,iterations,salt,digest=encoded.split("$",3)
        if scheme != "pbkdf2_sha256": return False
        actual=hashlib.pbkdf2_hmac("sha256",password.encode(),base64.urlsafe_b64decode(salt),int(iterations))
        return hmac.compare_digest(actual,base64.urlsafe_b64decode(digest))
    except (ValueError, TypeError): return False

class LocalAuthService:
    def __init__(self, session): self._session=session
    def authenticate(self, operator_id: str, password: str) -> dict | None:
        from sqlalchemy import select
        from app.database.models import OperatorAccount
        account=self._session.scalars(select(OperatorAccount).where(OperatorAccount.operator_id==operator_id.strip())).first()
        if not account or not account.active or not verify_password(password,account.password_hash): return None
        return {"operator_id":account.operator_id,"role":account.role}
    def create_account(self, operator_id: str, password: str, role: str, actor_role: str="administrator") -> str:
        from app.database.models import OperatorAccount
        if actor_role != "administrator": raise PermissionError("Administrator role required to provision accounts.")
        if role not in ROLES: raise ValueError("Invalid local role.")
        row=OperatorAccount(operator_id=operator_id.strip(),password_hash=hash_password(password),role=role)
        self._session.add(row);self._session.flush();return row.id
    @staticmethod
    def require_role(current_role: str | None, *permitted: str) -> None:
        if current_role not in permitted: raise PermissionError("Authenticated role is not permitted for this action.")
