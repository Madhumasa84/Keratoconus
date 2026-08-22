"""Offline local evaluation CLI for development, calibration, or locked data only when authorised."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from .config import EngineConfig
from .evaluation import EVALUATION_VERSION, grouped_bootstrap_ci, infer_cases, metric_summary, write_evaluation_outputs
from .manifest import ManifestValidationError, assert_audit_passes, audit_manifest, load_manifest, validate_partition_leakage
from .model_bundle import ModelBundleError, load_model_bundle
from .privacy import ensure_local_output, sha256_text


def installed_pipeline_version() -> str:
    """Return the pipeline version from the installed engine configuration."""
    return EngineConfig().pipeline_version


def run_evaluation(manifest: str, model: str, output: str, *, locked: bool=False, development_manifest: str|None=None, calibration_manifest: str|None=None, bootstrap_iterations: int=1000) -> dict:
    if not locked and "locked" in Path(manifest).name.lower():
        raise ManifestValidationError("Locked data require `python -m kerascan.evaluate_locked --confirm-locked-evaluation`.")
    records, audit = audit_manifest(manifest, hash_images=True)
    assert_audit_passes(audit)
    partitions={"evaluation":records}
    partition_hashes={"evaluation_manifest_hash":audit.manifest_hash}
    for name,path in (("development",development_manifest),("calibration",calibration_manifest)):
        if path:
            partition_records, partition_hash=load_manifest(path); partitions[name]=partition_records; partition_hashes[f"{name}_manifest_hash"]=partition_hash
    validate_partition_leakage(partitions)
    bundle=load_model_bundle(model,require_frozen=locked)
    expected_pipeline = installed_pipeline_version()
    if bundle.pipeline_version != expected_pipeline:
        raise ModelBundleError("Pipeline-version mismatch between frozen model and installed KERASCAN engine.")
    referral_engine = None
    try:
        from app.services.referral_engine import ReferralEngine
        referral_engine = ReferralEngine()
        installed_protocol = referral_engine.get_protocol_version()
        if bundle.protocol_version and bundle.protocol_version != installed_protocol:
            raise ModelBundleError("Referral-protocol version mismatch between frozen model bundle and installed application.")
    except ImportError:
        installed_protocol = bundle.protocol_version
    cases=infer_cases(records,bundle)
    metrics=metric_summary(cases); intervals=grouped_bootstrap_ci(cases,iterations=bootstrap_iterations)
    provenance={"evaluation_version":EVALUATION_VERSION,"test_manifest_hash":audit.manifest_hash,**partition_hashes,"model_hash":bundle.model_hash,"pipeline_version":bundle.pipeline_version,"model_version":bundle.model_version,"feature_schema_hash":bundle.feature_schema_hash,"threshold":bundle.threshold,"referral_protocol_version":installed_protocol,"executed_at":datetime.now(timezone.utc).isoformat(),"patient_leakage_check":"passed","locked_test_evaluation":locked,"image_paths_in_output":False,"telemetry":False}
    paths=write_evaluation_outputs(output,cases=cases,metrics=metrics,confidence_intervals=intervals,provenance=provenance,referral_engine=referral_engine)
    if locked:
        record=Path(output)/"immutable_locked_evaluation_record.json"; record.write_text(json.dumps({"provenance":provenance,"metrics":metrics,"warning":"Do not modify or retune the model based on locked-test results. Any subsequent model must be evaluated on a new untouched test set."},indent=2,sort_keys=True),encoding="utf-8"); record.chmod(0o444);paths["immutable_record"]=str(record)
    return {"metrics":metrics,"paths":paths,"provenance":provenance}


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="KERASCAN local offline evaluation. No images are uploaded or copied.")
    parser.add_argument("--manifest",required=True);parser.add_argument("--model",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--development-manifest");parser.add_argument("--calibration-manifest");parser.add_argument("--bootstrap-iterations",type=int,default=1000)
    args=parser.parse_args(argv)
    try:
        result=run_evaluation(args.manifest,args.model,args.output,development_manifest=args.development_manifest,calibration_manifest=args.calibration_manifest,bootstrap_iterations=args.bootstrap_iterations)
        print(f"Local evaluation complete: cases={result['metrics']['total_cases']}; patients={result['metrics']['total_patients']}; accuracy={result['metrics']['accuracy']}")
        return 0
    except (ManifestValidationError,ModelBundleError,ValueError) as error:
        print(f"Evaluation stopped: {error}");return 2


if __name__=="__main__": raise SystemExit(main())
