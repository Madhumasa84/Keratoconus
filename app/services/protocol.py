"""Versioned, local KeraScan school-screening protocol loader.

The application deliberately has one policy source.  UI text, validation,
decision rules, and referral reports obtain threshold values from this module
rather than repeating clinical-looking constants in multiple layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_PACHYMETRY_MEASUREMENT_TYPES = frozenset({"device_reported", "central", "thinnest"})


@dataclass(frozen=True)
class ScreeningProtocol:
    """Immutable validated representation of the provisional study protocol."""

    protocol_version: str
    initial_image_policy: str
    require_od_image: bool
    require_os_image: bool
    require_complete_measurements: bool
    k2_abnormal_above_d: float
    pachymetry_abnormal_below_um: float
    cylinder_magnitude_abnormal_above_d: float
    pachymetry_measurement_type: str
    detailed_pdf_for_actions: tuple[str, ...]
    disclaimer: str
    software_version: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ScreeningProtocol":
        required = {
            "protocol_version", "initial_image_policy", "require_od_image",
            "require_os_image", "require_complete_measurements",
            "k2_abnormal_above_d", "pachymetry_abnormal_below_um",
            "cylinder_magnitude_abnormal_above_d", "pachymetry_measurement_type",
            "detailed_pdf_for_actions", "disclaimer", "software_version",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Protocol is missing required keys: {', '.join(missing)}")
        measurement_type = str(data["pachymetry_measurement_type"])
        if measurement_type not in ALLOWED_PACHYMETRY_MEASUREMENT_TYPES:
            raise ValueError(
                "pachymetry_measurement_type must be one of: "
                + ", ".join(sorted(ALLOWED_PACHYMETRY_MEASUREMENT_TYPES))
            )
        actions = tuple(str(action) for action in data["detailed_pdf_for_actions"])
        if not actions:
            raise ValueError("detailed_pdf_for_actions cannot be empty")
        return cls(
            protocol_version=str(data["protocol_version"]),
            initial_image_policy=str(data["initial_image_policy"]),
            require_od_image=bool(data["require_od_image"]),
            require_os_image=bool(data["require_os_image"]),
            require_complete_measurements=bool(data["require_complete_measurements"]),
            k2_abnormal_above_d=float(data["k2_abnormal_above_d"]),
            pachymetry_abnormal_below_um=float(data["pachymetry_abnormal_below_um"]),
            cylinder_magnitude_abnormal_above_d=float(data["cylinder_magnitude_abnormal_above_d"]),
            pachymetry_measurement_type=measurement_type,
            detailed_pdf_for_actions=actions,
            disclaimer=str(data["disclaimer"]).strip(),
            software_version=str(data["software_version"]),
        )


def default_protocol_path() -> Path:
    return Path(__file__).parent.parent / "config" / "referral_protocol.yaml"


def load_protocol(path: str | Path | None = None) -> ScreeningProtocol:
    """Load the local configuration without network or external services."""
    source = Path(path) if path is not None else default_protocol_path()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Protocol configuration must be a YAML mapping")
    return ScreeningProtocol.from_mapping(data)
