"""Prevent local source paths from appearing in logs and de-identified exports."""
from __future__ import annotations
from pathlib import Path

def redact_path(path) -> str:
    return f"<local>/{Path(str(path)).name}" if path else "<missing>"

def redact_paths(value):
    if isinstance(value, dict):
        # Artifact manifests use a plain ``path`` key; redact it as strictly as
        # historic ``*_path`` fields so exports never disclose local layout.
        return {
            key: (redact_path(item) if (key == "path" or key.endswith("_path")) and item else redact_paths(item))
            for key, item in value.items()
        }
    if isinstance(value, list): return [redact_paths(item) for item in value]
    return value
