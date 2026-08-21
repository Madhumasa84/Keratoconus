"""Local-only privacy, hashing, and safe-path utilities for Phase 3."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def redact_path(path: str | Path | None) -> str:
    """Return a non-identifying log representation; never return the parent path."""
    if not path:
        return "<missing>"
    name = Path(path).name
    return f"<local>/{name or '<unnamed>'}"


def ensure_local_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output
