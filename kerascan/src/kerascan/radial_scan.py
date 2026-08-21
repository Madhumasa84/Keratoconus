"""Configurable radial crossing extraction; missing crossings remain NaN."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from .config import RadialConfig

@dataclass
class RadialResult:
    radii: np.ndarray  # ring x angle
    angles_deg: np.ndarray
    crossings: list[list[float]]
    order_change_fraction: float


def _runs(values: np.ndarray) -> list[tuple[int,int]]:
    idx=np.flatnonzero(values)
    if not len(idx): return []
    cuts=np.flatnonzero(np.diff(idx)>1)+1
    return [(int(a[0]),int(a[-1])) for a in np.split(idx,cuts)]


def radial_scan(mask: np.ndarray, center: tuple[float,float], outer_radius: float, config: RadialConfig = RadialConfig()) -> RadialResult:
    angles=np.linspace(0,360,config.meridians,endpoint=False)
    result=np.full((config.max_rings,config.meridians),np.nan,dtype=float)
    all_cross=[]; counts=[]
    min_r=outer_radius*config.min_radius_fraction
    for ai,deg in enumerate(angles):
        t=np.arange(min_r, outer_radius, config.radial_sample_step)
        xs=np.rint(center[0]+t*np.cos(np.deg2rad(deg))).astype(int)
        ys=np.rint(center[1]+t*np.sin(np.deg2rad(deg))).astype(int)
        valid=(xs>=0)&(xs<mask.shape[1])&(ys>=0)&(ys<mask.shape[0])
        t,xs,ys=t[valid],xs[valid],ys[valid]
        v=mask[ys,xs]>0
        crossings=[]
        for a,b in _runs(v):
            if b-a+1 <= 1: crossings.append(float(np.median(t[a:b+1])))
        crossings=sorted(crossings)
        # Thin edges can produce a double crossing. Merge immediately adjacent edges.
        merged=[]
        for r in crossings:
            if not merged or r-merged[-1] > config.max_gap_px: merged.append(r)
            else: merged[-1]=(merged[-1]+r)/2
        all_cross.append(merged); counts.append(len(merged))
        result[:min(config.max_rings,len(merged)),ai]=merged[:config.max_rings]
    med=np.median(counts) if counts else 0
    order_changes=float(np.mean(np.abs(np.asarray(counts)-med)>2)) if counts else 1.0
    return RadialResult(result,angles,all_cross,order_changes)
