"""Bounded, explainable refinement of an ROI-derived Placido centre."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .config import CentreRefinementConfig
from .image_io import to_gray


@dataclass(frozen=True)
class CentreRefinementResult:
    initial_center: tuple[float, float]
    refined_center: tuple[float, float]
    displacement_px: float
    confidence: float
    method: str
    objective_initial: float
    objective_refined: float


def _ring_consistency_objective(gray: np.ndarray, center: tuple[float,float], outer_radius: float, angles: int) -> float:
    """Nested rings align into sharp radial peaks only when the centre is correct."""
    radii=np.arange(max(4.,outer_radius*.06),outer_radius*.92,.75)
    theta=np.linspace(0,2*np.pi,angles,endpoint=False)
    xx=np.rint(center[0]+np.cos(theta)[:,None]*radii).astype(int)
    yy=np.rint(center[1]+np.sin(theta)[:,None]*radii).astype(int)
    valid=(xx>=0)&(xx<gray.shape[1])&(yy>=0)&(yy<gray.shape[0])
    samples=np.where(valid,gray[np.clip(yy,0,gray.shape[0]-1),np.clip(xx,0,gray.shape[1]-1)],np.nan)
    profile=gaussian_filter1d(np.nanmedian(samples,axis=0),1.2)
    if not np.any(np.isfinite(profile)): return -np.inf
    prominence=max(3.,float(np.nanstd(profile)*.18))
    peaks, properties=find_peaks(profile,prominence=prominence,distance=5)
    if not len(peaks): return 0.
    # Peak count plus normalised prominence rewards radial symmetry, while the
    # angular standard deviation penalises an off-centre fit.
    angular_variation=float(np.nanmean(np.nanstd(samples[:,peaks],axis=0)))
    return float(len(peaks)+np.mean(properties["prominences"])/max(np.nanstd(profile),1.)-angular_variation/max(np.nanstd(profile),1.))


def refine_centre(image: np.ndarray, initial_center: tuple[float,float], outer_radius: float,
                  config: CentreRefinementConfig = CentreRefinementConfig()) -> CentreRefinementResult:
    gray=to_gray(image)
    initial=(float(initial_center[0]),float(initial_center[1]))
    baseline=_ring_consistency_objective(gray,initial,outer_radius,config.radial_samples)
    best_center,best_score=initial,baseline
    offsets=np.arange(-config.max_displacement_px,config.max_displacement_px+1e-6,config.search_step_px)
    for dx in offsets:
        for dy in offsets:
            candidate=(initial[0]+float(dx),initial[1]+float(dy))
            if not (0<=candidate[0]<gray.shape[1] and 0<=candidate[1]<gray.shape[0]): continue
            score=_ring_consistency_objective(gray,candidate,outer_radius,config.radial_samples)
            if score>best_score: best_center,best_score=candidate,score
    displacement=float(np.hypot(best_center[0]-initial[0],best_center[1]-initial[1]))
    improvement=(best_score-baseline)/max(abs(baseline),1.)
    if improvement<config.min_improvement_fraction:
        best_center,best_score,displacement=initial,baseline,0.
    confidence=float(np.clip(max(improvement,0.)/.20,0.,1.))
    return CentreRefinementResult(initial,best_center,displacement,confidence,"bounded_radial_ring_consistency",float(baseline),float(best_score))
