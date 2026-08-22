"""Explicit development-only training CLI. It rejects locked-test paths by design."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .config import EngineConfig
from .features import FEATURE_ORDER, feature_vector
from .inference import KerascanEngine
from .manifest import ManifestValidationError, assert_audit_passes, audit_manifest, load_manifest, validate_partition_leakage
from .model_bundle import ModelBundle, save_model_bundle
from .model_development import grouped_model_comparison, select_calibration_threshold

def _feature_data(records):
    engine=KerascanEngine(EngineConfig());values=[];labels=[];groups=[]
    for record in records:
        if record.reference_label not in {"NORMAL","SUSPICIOUS"}: continue
        result=engine.analyze(record.image_path)
        if result.get("classification_skipped") or not result.get("features"): continue
        values.append(feature_vector(result["features"])); labels.append(record.reference_label=="SUSPICIOUS"); groups.append(record.patient_id)
    if not values: raise ValueError("No gradable development images yielded features; improve acquisition/ROI quality, not the locked-test model.")
    return np.asarray(values),np.asarray(labels,dtype=int),np.asarray(groups)

def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="Approved development-only KERASCAN model comparison/training.")
    parser.add_argument("--development-manifest",required=True);parser.add_argument("--calibration-manifest");parser.add_argument("--output-model",required=True);parser.add_argument("--approved-development",action="store_true")
    args=parser.parse_args(argv)
    if not args.approved_development: parser.error("Training requires explicit --approved-development; never use locked-test data.")
    if "locked" in Path(args.development_manifest).name.lower() or (args.calibration_manifest and "locked" in Path(args.calibration_manifest).name.lower()): parser.error("Locked-test data are evaluation-only and cannot be supplied for training/calibration.")
    try:
        development,audit=audit_manifest(args.development_manifest);assert_audit_passes(audit);partitions={"development":development}
        calibration=None
        if args.calibration_manifest:
            calibration,_=load_manifest(args.calibration_manifest);partitions["calibration"]=calibration
        validate_partition_leakage(partitions)
        engine_config = EngineConfig()
        x,y,g=_feature_data(development);comparison=grouped_model_comparison(x,y,g);winner=max(comparison,key=lambda item:item.metrics.get("balanced_accuracy") or -1)
        winner.estimator.fit(x,y);threshold=engine_config.model.decision_threshold
        if calibration:
            cx,cy,_=_feature_data(calibration);threshold=select_calibration_threshold(winner.estimator.predict_proba(cx)[:,1],cy)
        bundle=ModelBundle(
            winner.estimator,
            FEATURE_ORDER,
            threshold,
            engine_config.pipeline_version,
            f"development-{winner.name}",
            False,
            "development",
            "calibration" if calibration else None,
            None,
        )
        save_model_bundle(bundle,args.output_model)
        Path(str(args.output_model)+".comparison.json").write_text(json.dumps([{ "name":item.name,"metrics":item.metrics} for item in comparison],indent=2),encoding="utf-8")
        print(f"Development-only model bundle written: {Path(args.output_model).name}; selected={winner.name}; frozen=false")
        return 0
    except (ManifestValidationError,ValueError) as error: print(f"Development training stopped: {error}");return 2
if __name__=="__main__":raise SystemExit(main())
