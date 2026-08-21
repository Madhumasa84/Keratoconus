"""Versioned, local frozen-model bundle handling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import joblib

from .features import FEATURE_ORDER
from .privacy import sha256_file, sha256_text


class ModelBundleError(ValueError):
    pass


@dataclass
class ModelBundle:
    estimator: Any
    feature_order: list[str]
    threshold: float
    pipeline_version: str
    model_version: str
    frozen: bool
    training_partition: str
    calibration_partition: str | None = None
    protocol_version: str | None = None
    model_hash: str | None = None
    feature_schema_hash: str | None = None

    def metadata(self) -> dict:
        values = asdict(self)
        values.pop("estimator", None)
        return values


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(asdict(bundle), path)
    return path


def load_model_bundle(path: str | Path, *, require_frozen: bool = False) -> ModelBundle:
    path = Path(path).expanduser()
    if not path.is_file(): raise ModelBundleError("Model file is unavailable locally.")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "estimator" not in payload:
        raise ModelBundleError("Model must be a KERASCAN versioned model bundle, not a bare estimator.")
    required = {"feature_order", "threshold", "pipeline_version", "model_version", "frozen", "training_partition"}
    missing = required - set(payload)
    if missing: raise ModelBundleError("Model bundle missing provenance fields: " + ", ".join(sorted(missing)))
    bundle = ModelBundle(**payload)
    if bundle.feature_order != FEATURE_ORDER:
        raise ModelBundleError("Feature schema mismatch: the frozen model does not match this pipeline.")
    if not 0 < float(bundle.threshold) < 1:
        raise ModelBundleError("Model threshold must be between 0 and 1.")
    if require_frozen and not bundle.frozen:
        raise ModelBundleError("Locked evaluation requires a model bundle explicitly marked frozen.")
    bundle.model_hash = sha256_file(path)
    bundle.feature_schema_hash = sha256_text("\n".join(bundle.feature_order))
    return bundle
