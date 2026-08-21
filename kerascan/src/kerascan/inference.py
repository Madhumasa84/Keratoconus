"""Orchestration. Ungradable images return before any classifier invocation."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib, json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .config import EngineConfig
from .image_io import read_image, save_png, image_sha256
from .roi_detection import detect_placido_roi
from .segmentation import TraditionalSegmenter
from .quality import evaluate_quality
from .radial_scan import radial_scan
from .graph_tracking import track_rings
from .features import extract_features, feature_vector, FEATURE_ORDER
from .synthetic import synthetic_feature_table

EXPERIMENTAL = 'experimental—not clinically calibrated'

class KerascanEngine:
    def __init__(self, config: EngineConfig = EngineConfig(), classifier='logistic'):
        self.config=config; self.classifier_name=classifier; self.model=None; self.model_metadata=None

    def fit_synthetic_baseline(self, n=500):
        """Fit exclusively deterministic synthetic proxy data; real inputs are rejected."""
        x,y=synthetic_feature_table(FEATURE_ORDER,n,self.config.model.random_seed)
        estimator=LogisticRegression(max_iter=1000,random_state=self.config.model.random_seed) if self.classifier_name=='logistic' else RandomForestClassifier(n_estimators=150,random_state=self.config.model.random_seed)
        self.model=Pipeline([('scale',StandardScaler()),('classifier',estimator)]) if self.classifier_name=='logistic' else estimator
        self.model.fit(x,y)
        blob=repr(self.model.get_params(deep=True)).encode()
        self.model_metadata={'feature_order':FEATURE_ORDER,'input_size':len(FEATURE_ORDER),'threshold':self.config.model.decision_threshold,'pipeline_version':self.config.pipeline_version,'model_hash':hashlib.sha256(blob).hexdigest(),'training_data':'synthetic development proxies only','score_status':EXPERIMENTAL}
        return self.model_metadata

    def analyze(self, source: str | Path | np.ndarray, output_dir: str | Path | None = None) -> dict:
        image=read_image(source) if isinstance(source,(str,Path)) else source.copy()
        roi=detect_placido_roi(image,self.config.roi)
        segmenter=TraditionalSegmenter(self.config.segmentation)
        seg=segmenter.segment(roi.crop,roi.center_roi,roi.outer_radius_px)
        quality=evaluate_quality(roi.crop,roi.center_roi,roi.outer_radius_px,self.config.quality,seg.mask)
        source_pattern_fraction=roi.outer_radius_px/max(min(image.shape[:2])/2,1)
        quality['metrics']['source_pattern_radius_fraction']=float(source_pattern_fraction)
        if source_pattern_fraction < .15:
            quality['flags'].append('placido_pattern_too_small')
            quality['flags']=sorted(set(quality['flags']))
            quality['gradable']=False
            quality['quality_score']=max(0,quality['quality_score']-12)
        result={'pipeline_version':self.config.pipeline_version,'original_shape':list(image.shape),'original_sha256':image_sha256(image),
                'roi':{'box_xyxy':roi.box,'center_full':roi.center_full,'center_roi':roi.center_roi,'outer_radius_px':roi.outer_radius_px,'confidence':roi.confidence,'method':roi.method},'quality':quality}
        if output_dir:
            d=Path(output_dir); save_png(d/'original_full_resolution.png',image); save_png(d/'cropped_roi.png',roi.crop); save_png(d/'ring_mask.png',seg.mask)
            from .visualization import save_roi_overlay
            save_roi_overlay(image,roi.box,d)
        if not quality['gradable']:
            result.update({'screening_result':'UNGRADABLE','message':'recapture required','experimental_status':EXPERIMENTAL,'classification_skipped':True})
            if output_dir: (Path(output_dir)/'result.json').write_text(json.dumps(result,indent=2,default=float))
            return result
        scan=radial_scan(seg.mask,roi.center_roi,roi.outer_radius_px,self.config.radial)
        tracked=track_rings(scan,self.config.tracking)
        feats=extract_features(tracked.radii,scan.angles_deg,roi.center_roi,seg.confidence,quality['quality_score'])
        quality['metrics'].update({
            'detected_ring_count': feats['detected_ring_count'],
            'missing_ring_fraction': tracked.missing_fraction,
            'ring_order_change_fraction': scan.order_change_fraction,
            'ring_tracking_confidence': tracked.confidence,
        })
        # Broken/missing rings are measured after radial localisation but still gate
        # the classifier. A configured maximum is not itself counted as a missing ring.
        if tracked.missing_fraction > .65:
            quality['flags']=sorted(set(quality['flags']+['excessive_missing_or_broken_rings']))
            quality['gradable']=False; quality['quality_score']=max(0,quality['quality_score']-12)
        tracking={'confidence':tracked.confidence,'missing_point_fraction':tracked.missing_fraction,'duplicate_removals':tracked.duplicate_removals,'identity_shift_fraction':tracked.identity_shift_fraction,'order_change_fraction':scan.order_change_fraction}
        artifacts={'mask':seg.mask,'radii':tracked.radii,'angles':scan.angles_deg,'roi_crop':roi.crop,'center':roi.center_roi}
        if not quality['gradable']:
            result.update({'screening_result':'UNGRADABLE','message':'recapture required','experimental_status':EXPERIMENTAL,'classification_skipped':True,'features':feats,'tracking':tracking,'_artifacts':artifacts})
            if output_dir:
                from .visualization import save_visualizations
                save_visualizations(result,d)
                serial={k:v for k,v in result.items() if k!='_artifacts'}
                (d/'result.json').write_text(json.dumps(serial,indent=2,default=float))
            return result
        if self.model is None: self.fit_synthetic_baseline()
        probability=float(self.model.predict_proba(feature_vector(feats).reshape(1,-1))[0,1])
        label='SUSPICIOUS' if probability>=self.config.model.decision_threshold else 'NORMAL-LIKE'
        result.update({'screening_result':label,'experimental_status':EXPERIMENTAL,'classification_skipped':False,'prototype_score':probability,
                       'model':self.model_metadata,'features':feats,'tracking':tracking,'_artifacts':artifacts})
        if output_dir:
            from .visualization import save_visualizations
            save_visualizations(result,d)
            serial={k:v for k,v in result.items() if k!='_artifacts'}
            (d/'result.json').write_text(json.dumps(serial,indent=2,default=float))
        return result
