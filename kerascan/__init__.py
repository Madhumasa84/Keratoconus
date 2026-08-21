"""Source-layout compatibility wrapper for offline ``python -m kerascan...`` commands."""
from pathlib import Path

_source_package = Path(__file__).resolve().parent / "src" / "kerascan"
if str(_source_package) not in __path__:
    __path__.append(str(_source_package))

from .config import EngineConfig
from .inference import KerascanEngine

__all__ = ["EngineConfig", "KerascanEngine"]
