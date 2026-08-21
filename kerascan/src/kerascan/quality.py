"""Conservative pre-classification image-quality gate."""
from __future__ import annotations
import cv2
import numpy as np
from .config import QualityConfig
from .image_io import to_gray


def _robust_noise(gray: np.ndarray) -> float:
    residual=gray.astype(float)-cv2.GaussianBlur(gray,(0,0),1.2).astype(float)
    return float(1.4826*np.median(np.abs(residual-np.median(residual))))


def evaluate_quality(image: np.ndarray, center: tuple[float,float], outer_radius: float,
                     config: QualityConfig = QualityConfig(), ring_mask: np.ndarray | None = None) -> dict:
    """Return machine-readable gate result. This gate never assigns clinical labels."""
    gray=to_gray(image); h,w=gray.shape
    flags=[]; metrics={}
    metrics['roi_width_px'],metrics['roi_height_px']=int(w),int(h)
    metrics['resolution_min_side_px']=int(min(w,h))
    if min(w,h)<config.min_roi_side_px: flags.append('low_resolution')
    lap=float(cv2.Laplacian(gray,cv2.CV_64F).var()); metrics['laplacian_variance']=lap
    if lap<config.min_laplacian_variance: flags.append('blur')
    mean=float(gray.mean()); contrast=float(gray.std()); metrics.update(mean_intensity=mean,contrast_std=contrast)
    if mean<config.min_mean_intensity: flags.append('underexposed')
    if contrast<config.min_contrast: flags.append('low_contrast')
    sat=float(np.mean(gray>=250)); metrics['saturation_fraction']=sat
    if sat>config.max_saturation_fraction: flags.append('glare_or_saturation')
    noise=_robust_noise(gray); metrics['noise_sigma_estimate']=noise
    if noise>config.max_noise_sigma: flags.append('sensor_noise')
    dist=np.hypot(center[0]-w/2,center[1]-h/2); centring=1-dist/max(outer_radius,1)
    metrics['pattern_centring_ratio']=centring
    if centring<config.min_centring_ratio: flags.append('pattern_off_centre')
    pattern_ratio=float(outer_radius/max(min(h,w)/2,1)); metrics['pattern_radius_fraction']=pattern_ratio
    if outer_radius<config.min_roi_side_px*0.22: flags.append('placido_pattern_too_small')
    if ring_mask is not None:
        yy,xx=np.ogrid[:h,:w]; rr=np.hypot(xx-center[0],yy-center[1]); th=(np.degrees(np.arctan2(yy-center[1],xx-center[0]))%360)
        ann=(rr>outer_radius*.10)&(rr<outer_radius*.95)
        density=float(np.count_nonzero((ring_mask>0)&ann)/max(np.count_nonzero(ann),1)); metrics['ring_pixel_fraction']=density
        sectors=[]
        for a in range(24):
            sec=ann&(th>=a*15)&(th<(a+1)*15)
            sectors.append(np.count_nonzero((ring_mask>0)&sec)/max(np.count_nonzero(sec),1)>0.004)
        coverage=float(np.mean(sectors)); metrics['visible_ring_sector_coverage']=coverage
        if coverage<config.min_angular_coverage: flags.append('insufficient_ring_sector_coverage')
        if density<config.min_ring_fraction: flags.append('non_ring_or_incorrect_input')
        # Obstruction: unusually empty or dark contiguous peripheral sectors (not a diagnosis).
        empty=np.asarray(sectors,dtype=int)==0
        ring_band=(rr>outer_radius*.50)&(rr<outer_radius*.90)
        sector_means=np.asarray([np.mean(gray[ring_band&(th>=a*15)&(th<(a+1)*15)]) for a in range(24)])
        dark=sector_means < max(8., np.median(sector_means)*.65)
        metrics['dark_peripheral_sector_fraction']=float(np.mean(dark))
        obstruction=np.r_[empty|dark,(empty|dark)[:3]]
        if np.any(np.convolve(obstruction,np.ones(4,dtype=int),'valid')>=4): flags.append('possible_eyelid_or_eyelash_obstruction')
    else:
        metrics['visible_ring_sector_coverage']=None
    # Score is transparent and intentionally not a medical quality standard.
    score=max(0,100-12*len(set(flags)))
    critical={'low_resolution','underexposed','blur','non_ring_or_incorrect_input','insufficient_ring_sector_coverage','placido_pattern_too_small'}
    return {'gradable':not any(f in critical for f in flags),'quality_score':int(score),'flags':sorted(set(flags)),'metrics':metrics}
