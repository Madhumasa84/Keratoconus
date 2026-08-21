import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from kerascan.evaluation import EvaluationCase, grouped_bootstrap_ci, metric_summary, write_evaluation_outputs
from kerascan.manifest import ManifestValidationError, assert_audit_passes, audit_manifest, load_manifest, validate_partition_leakage
from kerascan.model_bundle import ModelBundle, ModelBundleError, load_model_bundle, save_model_bundle


def _manifest(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["patient_id","eye","session_id","image_path","reference_label","site","device_id","operator_id"])
        writer.writeheader();writer.writerows(rows)


def _row(patient, eye, session, image, label):
    return {"patient_id":patient,"eye":eye,"session_id":session,"image_path":str(image),"reference_label":label,"site":"site","device_id":"device","operator_id":"operator"}


def test_manifest_audit_duplicate_hash_path_redaction_and_no_inference(tmp_path):
    image=tmp_path/"confidential_child_image.png";image.write_bytes(b"local-only-image")
    manifest=tmp_path/"manifest.csv";_manifest(manifest,[_row("P001","OD","S1",image,"NORMAL"),_row("P002","OS","S1",image,"SUSPICIOUS")])
    records,audit=audit_manifest(manifest)
    assert audit.inference_performed is False
    assert len(audit.duplicate_image_hashes)==1
    assert str(tmp_path) not in json.dumps(audit.to_dict())
    with pytest.raises(ManifestValidationError): assert_audit_passes(audit)


def test_manifest_detects_duplicate_record_and_partition_leakage(tmp_path):
    image=tmp_path/"i.png";image.write_bytes(b"x")
    development=tmp_path/"development.csv";locked=tmp_path/"locked.csv"
    _manifest(development,[_row("P001","OD","S1",image,"NORMAL"),_row("P001","OD","S1",image,"NORMAL")])
    _manifest(locked,[_row("P001","OS","S1",image,"SUSPICIOUS")])
    dev,_=load_manifest(development);test,_=load_manifest(locked)
    with pytest.raises(ManifestValidationError,match="leakage"):
        validate_partition_leakage({"development":dev,"locked":test})
    _,audit=audit_manifest(development,hash_images=False)
    assert len(audit.duplicate_records)==1


def test_frozen_versioned_model_bundle_and_mismatch_protection(tmp_path):
    from kerascan.features import FEATURE_ORDER
    estimator=DummyClassifier(strategy="prior",random_state=1).fit(np.zeros((2,len(FEATURE_ORDER))),[0,1])
    path=tmp_path/"frozen.joblib"
    save_model_bundle(ModelBundle(estimator,FEATURE_ORDER,.55,"phase1-0.1.0","test-frozen",True,"development","calibration","2.0.0"),path)
    bundle=load_model_bundle(path,require_frozen=True)
    assert bundle.model_hash and bundle.feature_schema_hash
    bad=tmp_path/"bad.joblib";save_model_bundle(ModelBundle(estimator,["wrong"],.55,"p","v",True,"development"),bad)
    with pytest.raises(ModelBundleError,match="schema"):
        load_model_bundle(bad)


def test_grouped_metrics_outputs_and_target_not_fabricated(tmp_path):
    cases=[
        EvaluationCase("P1","OD","S1","NORMAL","NORMAL-LIKE",.1,[],"Unknown"),
        EvaluationCase("P1","OS","S1","NORMAL","SUSPICIOUS",.8,[],"Image-model false positive"),
        EvaluationCase("P2","OD","S1","SUSPICIOUS","SUSPICIOUS",.9,[],"Unknown"),
        EvaluationCase("P3","OD","S1","SUSPICIOUS","UNGRADABLE",None,["blur"],"Blur"),
    ]
    metrics=metric_summary(cases);ci=grouped_bootstrap_ci(cases,iterations=20)
    paths=write_evaluation_outputs(tmp_path,cases=cases,metrics=metrics,confidence_intervals=ci,provenance={"test_manifest_hash":"abc","patient_leakage_check":"passed"})
    assert metrics["ungradable_rate"]==.25
    assert metrics["accuracy"] != 1.0
    for path in paths.values(): assert Path(path).exists()
    assert "patient_id" not in Path(paths["aggregate_metrics_json"]).read_text()


def test_locked_command_requires_explicit_confirmation():
    from kerascan.evaluate_locked import main
    with pytest.raises(SystemExit) as failure:
        main(["--manifest","x.csv","--model","model.joblib","--output","out"])
    assert failure.value.code == 2


def test_local_evaluation_writes_required_aggregate_files_and_stops_on_protocol_mismatch(tmp_path):
    from kerascan.features import FEATURE_ORDER
    from kerascan.evaluate import run_evaluation
    image=tmp_path/"blank.png"; image.write_bytes(b"not-a-decodable-image")
    manifest=tmp_path/"evaluation.csv"; _manifest(manifest,[_row("P001","OD","S1",image,"UNGRADABLE")])
    estimator=DummyClassifier(strategy="prior").fit(np.zeros((2,len(FEATURE_ORDER))),[0,1])
    model=tmp_path/"model.joblib"
    save_model_bundle(ModelBundle(estimator,FEATURE_ORDER,.55,"phase1-0.1.0","frozen",True,"development",protocol_version="3.0.0"),model)
    result=run_evaluation(str(manifest),str(model),str(tmp_path/"results"),bootstrap_iterations=5)
    assert Path(result["paths"]["aggregate_metrics_xlsx"]).exists()
    assert Path(result["paths"]["final_pdf"]).exists()
    bad=tmp_path/"bad_protocol.joblib"
    save_model_bundle(ModelBundle(estimator,FEATURE_ORDER,.55,"phase1-0.1.0","bad",True,"development",protocol_version="mismatch"),bad)
    with pytest.raises(ModelBundleError,match="protocol"):
        run_evaluation(str(manifest),str(bad),str(tmp_path/"bad"),bootstrap_iterations=2)
