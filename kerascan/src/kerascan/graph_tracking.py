"""Spatially consistent assignment that preserves gaps and resolves duplicates."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import TrackingConfig
from .radial_scan import RadialResult

@dataclass
class TrackingResult:
    radii: np.ndarray
    confidence: float
    missing_fraction: float
    duplicate_removals: int
    identity_shift_fraction: float


def track_rings(scan: RadialResult, config: TrackingConfig = TrackingConfig()) -> TrackingResult:
    raw=scan.radii
    rings,angles=raw.shape
    out=np.full_like(raw,np.nan)
    # Radial order is a valid initial identity.  Reassignment is permitted only when
    # it has strong support from a nearby angular history; this avoids cascading all
    # crossings onto one ring when a broken ring produces an omitted crossing.
    duplicate=shifts=assigned=0
    history=[[] for _ in range(rings)]
    for a in range(angles):
        candidates=raw[:,a][np.isfinite(raw[:,a])].tolist()
        used=set()
        for r, value in enumerate(candidates):
            expected=np.nanmedian(history[r][-10:]) if history[r] else np.nan
            target=r
            if np.isfinite(expected) and abs(value-expected)>config.radial_tolerance_px:
                choices=[(abs(value-np.nanmedian(history[k][-10:])),k) for k in range(rings) if history[k]]
                d,target=min(choices,default=(np.inf,r))
                if d>config.radial_tolerance_px or target in used: target=r
                if target!=r: shifts+=1
            if target in used:
                duplicate+=1; continue
            out[target,a]=value; history[target].append(value); used.add(target); assigned+=1
    active=np.any(np.isfinite(out),axis=1)
    missing=float(np.mean(~np.isfinite(out[active]))) if np.any(active) else 1.0
    conf=float(np.clip((1-missing)*(1-scan.order_change_fraction)*(1-duplicate/max(assigned,1)),0,1))
    return TrackingResult(out,conf,missing,duplicate,shifts/max(assigned,1))
