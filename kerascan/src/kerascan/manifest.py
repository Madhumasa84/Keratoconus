"""Confidential-dataset manifest validation; all image access remains local."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .privacy import redact_path, sha256_file, sha256_text

REQUIRED_COLUMNS = ("patient_id", "eye", "session_id", "image_path", "reference_label")
OPTIONAL_COLUMNS = ("site", "device_id", "operator_id")
VALID_EYES = {"OD", "OS"}
VALID_REFERENCE_LABELS = {"NORMAL", "SUSPICIOUS", "UNGRADABLE", "EXCLUDE"}


class ManifestValidationError(ValueError):
    """A manifest safety/integrity invariant has been violated."""


@dataclass(frozen=True)
class ManifestRecord:
    patient_id: str
    eye: str
    session_id: str
    image_path: str
    reference_label: str
    site: str = ""
    device_id: str = ""
    operator_id: str = ""
    row_number: int = 0
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.patient_id, self.eye, self.session_id


@dataclass
class DatasetAudit:
    manifest_hash: str
    source_manifest: str
    total_records: int
    total_patients: int
    label_counts: dict[str, int]
    records_by_eye: dict[str, int]
    sessions_per_patient: dict[str, int]
    missing_images: list[dict]
    duplicate_records: list[dict]
    duplicate_image_hashes: dict[str, list[dict]]
    invalid_rows: list[dict]
    private_paths_redacted: bool = True
    inference_performed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical_hash(rows: Iterable[dict[str, str]]) -> str:
    canonical = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    return sha256_text(canonical)


def load_manifest(path: str | Path) -> tuple[list[ManifestRecord], str]:
    """Load a local CSV, validating schema but not copying images anywhere."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ManifestValidationError(f"Manifest is unavailable: {redact_path(path)}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
        if missing:
            raise ManifestValidationError(f"Manifest is missing required columns: {', '.join(missing)}")
        raw_rows = list(reader)
    records: list[ManifestRecord] = []
    errors: list[str] = []
    for row_number, row in enumerate(raw_rows, 2):
        values = {key: (row.get(key) or "").strip() for key in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
        if not all(values[key] for key in REQUIRED_COLUMNS):
            errors.append(f"row {row_number}: required values are missing")
            continue
        values["eye"] = values["eye"].upper()
        values["reference_label"] = values["reference_label"].upper()
        if values["eye"] not in VALID_EYES:
            errors.append(f"row {row_number}: eye must be OD or OS")
        if values["reference_label"] not in VALID_REFERENCE_LABELS:
            errors.append(f"row {row_number}: reference label is not supported")
        extras = {key: value for key, value in row.items() if key not in values and value not in (None, "")}
        records.append(ManifestRecord(**values, row_number=row_number, extras=extras))
    if errors:
        raise ManifestValidationError("; ".join(errors[:20]))
    return records, _canonical_hash(raw_rows)


def validate_partition_leakage(partitions: dict[str, list[ManifestRecord]]) -> None:
    """Reject a patient appearing in more than one explicitly supplied partition."""
    seen: dict[str, str] = {}
    overlaps: list[str] = []
    for name, records in partitions.items():
        for patient_id in {record.patient_id for record in records}:
            previous = seen.setdefault(patient_id, name)
            if previous != name:
                overlaps.append(patient_id)
    if overlaps:
        # IDs are anonymous but do not echo them into a broadly copied terminal log.
        raise ManifestValidationError(f"Patient-level partition leakage detected ({len(set(overlaps))} anonymous IDs). Stop and correct manifests.")


def audit_manifest(path: str | Path, *, hash_images: bool = True,
                   partitions: dict[str, list[ManifestRecord]] | None = None) -> tuple[list[ManifestRecord], DatasetAudit]:
    """Run a no-inference audit. Image hashes are read locally and not retained with paths."""
    records, manifest_hash = load_manifest(path)
    if partitions:
        validate_partition_leakage(partitions)
    duplicate_keys: dict[tuple[str, str, str], list[ManifestRecord]] = {}
    for record in records:
        duplicate_keys.setdefault(record.key, []).append(record)
    duplicate_records = [
        {"patient_id": records_[0].patient_id, "eye": records_[0].eye, "session_id": records_[0].session_id,
         "rows": [record.row_number for record in records_]}
        for records_ in duplicate_keys.values() if len(records_) > 1
    ]
    missing_images: list[dict] = []
    hashes: dict[str, list[dict]] = {}
    for record in records:
        candidate = Path(record.image_path).expanduser()
        if not candidate.is_file():
            missing_images.append({"row": record.row_number, "image": redact_path(candidate)})
            continue
        if hash_images:
            digest = sha256_file(candidate)
            hashes.setdefault(digest, []).append({"patient_id": record.patient_id, "eye": record.eye, "session_id": record.session_id, "row": record.row_number})
    duplicate_hashes = {digest: entries for digest, entries in hashes.items() if len(entries) > 1}
    labels = {label: sum(record.reference_label == label for record in records) for label in sorted(VALID_REFERENCE_LABELS)}
    audit = DatasetAudit(
        manifest_hash=manifest_hash, source_manifest=redact_path(path), total_records=len(records),
        total_patients=len({record.patient_id for record in records}), label_counts=labels,
        records_by_eye={eye: sum(record.eye == eye for record in records) for eye in sorted(VALID_EYES)},
        sessions_per_patient={"min": min((len({r.session_id for r in records if r.patient_id == patient}) for patient in {r.patient_id for r in records}), default=0),
                              "max": max((len({r.session_id for r in records if r.patient_id == patient}) for patient in {r.patient_id for r in records}), default=0)},
        missing_images=missing_images, duplicate_records=duplicate_records, duplicate_image_hashes=duplicate_hashes, invalid_rows=[])
    return records, audit


def assert_audit_passes(audit: DatasetAudit) -> None:
    problems = []
    if audit.missing_images: problems.append(f"{len(audit.missing_images)} image paths are missing")
    if audit.duplicate_records: problems.append(f"{len(audit.duplicate_records)} duplicate patient/eye/session records")
    if audit.duplicate_image_hashes: problems.append(f"{len(audit.duplicate_image_hashes)} duplicate image hashes")
    if problems:
        raise ManifestValidationError("Dataset audit failed: " + "; ".join(problems))
