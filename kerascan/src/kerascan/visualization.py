"""Auditable Phase 1 visual artefacts."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_visualizations(result: dict, output_dir: str|Path):
    d=Path(output_dir); d.mkdir(parents=True,exist_ok=True)
    a=result['_artifacts']; crop=a['roi_crop'][:,:,:3].copy() if a['roi_crop'].ndim==3 else a['roi_crop'].copy(); center=tuple(map(int,a['center']))
    cv2.drawMarker(crop,center,(0,0,255),cv2.MARKER_CROSS,18,2)
    radii=a['radii']; angles=a['angles']
    for i,row in enumerate(radii):
        ids=np.flatnonzero(np.isfinite(row))
        rgba=plt.cm.hsv(i/max(len(radii),1))
        color=(int(rgba[0]*255),int(rgba[1]*255),int(rgba[2]*255))
        for j in ids[::max(1,len(ids)//100)]:
            x=int(center[0]+row[j]*np.cos(np.deg2rad(angles[j]))); y=int(center[1]+row[j]*np.sin(np.deg2rad(angles[j])))
            cv2.circle(crop,(x,y),1,color,-1)
    cv2.imwrite(str(d/'centre_and_detected_rings.png'),crop)
    box=result['roi']['box_xyxy']; full=cv2.imread(str(d/'original_full_resolution.png')); cv2.rectangle(full,box[:2],box[2:],(0,255,0),2); cv2.imwrite(str(d/'detected_roi_box.png'),full)
    spacing=np.diff(radii,axis=0)
    counts=np.sum(np.isfinite(spacing),axis=0)
    mean_spacing=np.divide(np.nansum(spacing,axis=0),counts,out=np.full(len(angles),np.nan),where=counts>0)
    fig,ax=plt.subplots(figsize=(9,4)); ax.plot(angles,mean_spacing); ax.set(xlabel='Meridian (degrees)',ylabel='Mean spacing (pixels)',title='Directional spacing (image-space proxy)'); fig.tight_layout(); fig.savefig(d/'directional_spacing.png',dpi=150); plt.close(fig)
    missing=np.mean(~np.isfinite(radii),axis=0)
    fig,ax=plt.subplots(figsize=(9,2)); ax.bar(angles,missing,width=360/len(angles)); ax.set(xlabel='Meridian (degrees)',ylabel='Missing fraction',title='Missing-sector map'); fig.tight_layout(); fig.savefig(d/'missing_sector_map.png',dpi=150); plt.close(fig)
    (d/'features.txt').write_text('\n'.join(f'{k}: {v:.6g}' for k,v in result['features'].items())+'\n\n'+f"Result: {result['screening_result']} ({result['experimental_status']})\nQuality warnings: {', '.join(result['quality']['flags']) or 'none'}\n")

def save_roi_overlay(original: np.ndarray, box: tuple[int,int,int,int], output_dir: str|Path):
    d=Path(output_dir); d.mkdir(parents=True,exist_ok=True)
    overlay=original[:,:,:3].copy() if original.ndim==3 else original.copy(); cv2.rectangle(overlay,box[:2],box[2:],(0,255,0),2)
    cv2.imwrite(str(d/'detected_roi_box.png'),overlay)
