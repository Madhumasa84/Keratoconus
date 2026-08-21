"""Explainable image-space geometric proxies, not physical corneal topography."""
from __future__ import annotations
import numpy as np
import cv2

FEATURE_ORDER = [
 "mean_ring_spacing","std_ring_spacing","spacing_cv","max_local_spacing_change",
 "superior_inferior_asymmetry","left_right_asymmetry","opposite_meridian_difference",
 "mean_ring_circularity","mean_ellipse_axis_ratio","mean_fitted_center_displacement",
 "consecutive_center_drift","missing_ring_fraction","detected_ring_count","angular_coverage",
 "segmentation_confidence","quality_score"]

def _nanmean(x, default=0.0):
    y=np.nanmean(x) if np.any(np.isfinite(x)) else default
    return float(y) if np.isfinite(y) else default

def extract_features(radii: np.ndarray, angles_deg: np.ndarray, center: tuple[float,float], segmentation_confidence: float, quality_score: int) -> dict[str,float]:
    # Configured capacity is not an observed missing ring: measure missingness only
    # across ring identities which were detected at least once.
    active=np.any(np.isfinite(radii),axis=1)
    radii=radii[active]
    finite=np.isfinite(radii)
    spacing=np.diff(radii,axis=0)
    valid_spacing=spacing[np.isfinite(spacing) & (spacing>0)]
    mean_sp=_nanmean(valid_spacing); std_sp=float(np.nanstd(valid_spacing)) if len(valid_spacing) else 0.
    local=np.abs(np.diff(spacing,axis=1)); max_local=float(np.nanmax(local)) if np.any(np.isfinite(local)) else 0.
    n= len(angles_deg); upper=slice(n//4,3*n//4); # image y: lower half inferior
    per_angle=np.divide(np.nansum(spacing,axis=0),np.sum(np.isfinite(spacing),axis=0),out=np.full(spacing.shape[1],np.nan),where=np.sum(np.isfinite(spacing),axis=0)>0)
    superior=_nanmean(per_angle[:n//2]); inferior=_nanmean(per_angle[n//2:])
    left=_nanmean(per_angle[n//4:3*n//4]); right=_nanmean(np.r_[per_angle[:n//4],per_angle[3*n//4:]])
    opposite=np.abs(per_angle-np.roll(per_angle,n//2)); opp=_nanmean(opposite)
    circularities=[]; axis=[]; displacements=[]; fitted=[]
    for rr in radii:
        ids=np.flatnonzero(np.isfinite(rr))
        if len(ids)<8: continue
        theta=np.deg2rad(angles_deg[ids]); pts=np.column_stack((center[0]+rr[ids]*np.cos(theta),center[1]+rr[ids]*np.sin(theta))).astype(np.float32)
        contour=pts.reshape(-1,1,2)
        area=cv2.contourArea(contour); peri=cv2.arcLength(contour,True)
        if peri>0: circularities.append(4*np.pi*area/(peri*peri))
        if len(pts)>=5:
            (cx,cy),(a,b),_=cv2.fitEllipse(contour)
            axis.append(min(a,b)/max(a,b)); displacements.append(np.hypot(cx-center[0],cy-center[1])); fitted.append((cx,cy))
    drift=[np.hypot(fitted[i][0]-fitted[i-1][0],fitted[i][1]-fitted[i-1][1]) for i in range(1,len(fitted))]
    coverage=np.mean(np.any(finite,axis=0)); detected=int(np.sum(np.mean(finite,axis=1)>0.2))
    return dict(zip(FEATURE_ORDER,[mean_sp,std_sp,std_sp/max(mean_sp,1e-6),max_local,
        abs(superior-inferior)/max(mean_sp,1e-6),abs(left-right)/max(mean_sp,1e-6),opp/max(mean_sp,1e-6),
        _nanmean(circularities),_nanmean(axis),_nanmean(displacements),_nanmean(drift),
        float(1-np.mean(finite)),float(detected),float(coverage),float(segmentation_confidence),float(quality_score)]))

def feature_vector(features: dict[str,float]) -> np.ndarray:
    return np.asarray([features[k] for k in FEATURE_ORDER],dtype=float)
