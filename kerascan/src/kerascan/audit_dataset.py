"""CLI for confidential local-manifest auditing; it performs no inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .manifest import ManifestValidationError, audit_manifest, load_manifest, validate_partition_leakage
from .privacy import ensure_local_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KERASCAN local dataset audit (no inference; no upload).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-manifest")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--locked-test-manifest")
    parser.add_argument("--skip-image-hashes", action="store_true", help="Not recommended; skips duplicate-image detection.")
    parser.add_argument("--dry-run", action="store_true", help="Explicit no-inference mode (the audit never runs inference).")
    args = parser.parse_args(argv)
    try:
        partitions = {}
        for name, supplied in (("development", args.development_manifest), ("calibration", args.calibration_manifest), ("locked_test", args.locked_test_manifest)):
            if supplied:
                partitions[name] = load_manifest(supplied)[0]
        validate_partition_leakage(partitions)
        _, audit = audit_manifest(args.manifest, hash_images=not args.skip_image_hashes, partitions=partitions)
        output = Path(args.output).expanduser(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"Dataset audit completed locally: {output.name}; records={audit.total_records}; patients={audit.total_patients}; inference=false")
        return 0
    except ManifestValidationError as error:
        print(f"Dataset audit stopped: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
