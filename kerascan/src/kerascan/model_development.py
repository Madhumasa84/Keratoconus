"""Approved development/calibration workflow; never consumes locked-test data."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.calibration import CalibratedClassifierCV

from .evaluation import EvaluationCase, metric_summary
from .features import FEATURE_ORDER

@dataclass
class ModelComparison:
    name: str
    metrics: dict
    estimator: object

def candidate_estimators(seed: int = 20260821) -> dict[str, object]:
    candidates = {
        "logistic_regression": Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed))]),
        "calibrated_svm": Pipeline([("scale",StandardScaler()),("model",CalibratedClassifierCV(SVC(kernel="rbf",class_weight="balanced",random_state=seed),method="sigmoid",cv=3))]),
        "random_forest": RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=seed,min_samples_leaf=2),
        "extra_trees": ExtraTreesClassifier(n_estimators=400,class_weight="balanced",random_state=seed,min_samples_leaf=2),
        "geometry_quality_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    # Optional local-only gradient package. Absence is not an installation error.
    try:
        from xgboost import XGBClassifier
        candidates["xgboost"] = XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,random_state=seed,n_jobs=1,eval_metric="logloss")
    except ImportError:
        try:
            from lightgbm import LGBMClassifier
            candidates["lightgbm"] = LGBMClassifier(n_estimators=200,max_depth=3,learning_rate=.05,random_state=seed,verbosity=-1)
        except ImportError:
            pass
    return candidates

def grouped_model_comparison(features: np.ndarray, labels: np.ndarray, patient_ids: np.ndarray, *, seed: int = 20260821) -> list[ModelComparison]:
    """Compare interpretable/robust baselines with patient-grouped CV only."""
    unique=np.unique(patient_ids)
    if len(unique)<3: raise ValueError("At least three development patients are required for grouped cross-validation.")
    folds=min(5,len(unique))
    cv=GroupKFold(n_splits=folds); results=[]
    for name,estimator in candidate_estimators(seed).items():
        probabilities=cross_val_predict(estimator,features,labels,groups=patient_ids,cv=cv,method="predict_proba")[:,1]
        predictions=(probabilities>=.5).astype(int)
        cases=[EvaluationCase(str(patient),"OD","cv","SUSPICIOUS" if truth else "NORMAL","SUSPICIOUS" if prediction else "NORMAL-LIKE",float(score),[],"Unknown") for patient,truth,prediction,score in zip(patient_ids,labels,predictions,probabilities)]
        results.append(ModelComparison(name,metric_summary(cases),estimator))
    # Normal-only anomaly baseline. It is deliberately not a disease label: it only
    # measures departure from development NORMAL geometry under grouped folds.
    anomaly_scores=np.full(len(labels),np.nan)
    for train,test in cv.split(features,labels,groups=patient_ids):
        normal=features[train][labels[train]==0]
        if len(normal)<2: continue
        detector=Pipeline([("scale",StandardScaler()),("model",OneClassSVM(gamma="scale",nu=.10))]).fit(normal)
        anomaly_scores[test]=-detector.decision_function(features[test])
    if np.any(np.isfinite(anomaly_scores)):
        ranked=np.nan_to_num(anomaly_scores,nan=np.nanmedian(anomaly_scores[np.isfinite(anomaly_scores)]))
        probabilities=(ranked-ranked.min())/max(ranked.max()-ranked.min(),1e-9)
        predictions=(probabilities>=.5).astype(int)
        cases=[EvaluationCase(str(patient),"OD","cv","SUSPICIOUS" if truth else "NORMAL","SUSPICIOUS" if prediction else "NORMAL-LIKE",float(score),[],"Unknown") for patient,truth,prediction,score in zip(patient_ids,labels,predictions,probabilities)]
        results.append(ModelComparison("normal_only_anomaly_detector",metric_summary(cases),OneClassSVM(gamma="scale",nu=.10)))
    return results

def select_calibration_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Calibration-only threshold selection: maximise Youden J, never use locked data."""
    if len(np.unique(labels)) != 2: raise ValueError("Calibration data must include NORMAL and SUSPICIOUS labels.")
    grid=np.linspace(.05,.95,181); values=[((probabilities>=t)[labels==1].mean()-(probabilities>=t)[labels==0].mean(),t) for t in grid]
    return float(max(values,key=lambda item:item[0])[1])
