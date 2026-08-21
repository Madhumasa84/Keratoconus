"""Synthetic development data only: no clinical or supplied image enters fit()."""
from __future__ import annotations
import numpy as np
import cv2

def synthetic_placido(shape=(480,480), center=None, rings=12, distortion=0., rotation=0.,
                      blur=0., glare=False, darkness=0., occlusion=False, noise=0., low_contrast=False, seed=1):
    rng=np.random.default_rng(seed); h,w=shape
    center=center or (w/2,h/2); img=np.full((h,w),35,np.uint8)
    for i in range(rings):
        r=24+i*14
        pts=[]
        for angle in np.linspace(0,2*np.pi,720):
            # smooth asymmetric geometric deformation; not a disease simulation.
            radius=r*(1+distortion*np.cos(2*(angle+rotation)))
            pts.append((int(center[0]+radius*np.cos(angle)),int(center[1]+radius*np.sin(angle))))
        cv2.polylines(img,[np.asarray(pts,np.int32)],True,205,2,cv2.LINE_AA)
    if occlusion: cv2.rectangle(img,(0,0),(w,int(h*.34)),25,-1)
    if glare: cv2.circle(img,(int(w*.62),int(h*.38)),int(min(h,w)*.18),255,-1)
    if low_contrast: img=cv2.normalize(img,None,75,120,cv2.NORM_MINMAX)
    if darkness: img=np.clip(img.astype(float)*(1-darkness),0,255).astype(np.uint8)
    if noise: img=np.clip(img.astype(float)+rng.normal(0,noise,img.shape),0,255).astype(np.uint8)
    if blur: img=cv2.GaussianBlur(img,(0,0),blur)
    return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

def synthetic_feature_table(feature_order, n=500, seed=20260821):
    """Deterministic synthetic proxy table for baseline-only prototype fitting."""
    rng=np.random.default_rng(seed); x=[]; y=[]
    for label in (0,1):
        for _ in range(n//2):
            v=np.zeros(len(feature_order),float)
            values={
             'mean_ring_spacing':rng.normal(14,1.3), 'std_ring_spacing':rng.normal(1.2 if not label else 3.1,.45),
             'spacing_cv':rng.normal(.09 if not label else .25,.04), 'max_local_spacing_change':rng.normal(2 if not label else 7,1.5),
             'superior_inferior_asymmetry':abs(rng.normal(.08 if not label else .35,.08)),
             'left_right_asymmetry':abs(rng.normal(.08 if not label else .30,.08)),
             'opposite_meridian_difference':abs(rng.normal(.10 if not label else .32,.08)),
             'mean_ring_circularity':rng.normal(.92 if not label else .73,.07), 'mean_ellipse_axis_ratio':rng.normal(.94 if not label else .75,.08),
             'mean_fitted_center_displacement':abs(rng.normal(2 if not label else 10,2)), 'consecutive_center_drift':abs(rng.normal(1 if not label else 4,1)),
             'missing_ring_fraction':np.clip(rng.normal(.08 if not label else .18,.05),0,1), 'detected_ring_count':rng.normal(11,1),
             'angular_coverage':np.clip(rng.normal(.88 if not label else .70,.1),0,1), 'segmentation_confidence':np.clip(rng.normal(.82,.1),0,1), 'quality_score':rng.normal(88,6)}
            x.append([values[k] for k in feature_order]); y.append(label)
    return np.asarray(x),np.asarray(y)
