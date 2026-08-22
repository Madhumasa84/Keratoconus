"""Ring segmentation interface; traditional processing is the default."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import cv2
import numpy as np
from .config import SegmentationConfig
from .image_io import to_gray


@dataclass
class SegmentationResult:
    mask: np.ndarray
    enhanced: np.ndarray
    confidence: float
    method: str
    metrics: dict


class Segmenter(Protocol):
    def segment(self, image: np.ndarray, center: tuple[float,float], outer_radius: float) -> SegmentationResult: ...


class TraditionalSegmenter:
    def __init__(self, config: SegmentationConfig = SegmentationConfig()): self.config = config
    def segment(self, image, center, outer_radius):
        gray = to_gray(image)
        clahe = cv2.createCLAHE(self.config.clahe_clip_limit, (self.config.clahe_tile_size,)*2)
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 5, 30, 30)
        # Bright ridge evidence only.  In particular, no dilation/closing is used:
        # connecting adjacent white bands would make connected components meaningless.
        local_background=cv2.GaussianBlur(denoised,(0,0),self.config.ridge_sigma_px)
        ridge=denoised.astype(np.float32)-local_background.astype(np.float32)
        threshold=max(2.0,float(np.std(ridge)*self.config.ridge_threshold_sigma))
        mask=(ridge>threshold).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
        yy,xx = np.ogrid[:gray.shape[0],:gray.shape[1]]
        rr = np.hypot(xx-center[0], yy-center[1])
        valid = (rr >= outer_radius*self.config.inner_exclusion_fraction) & (rr <= outer_radius*self.config.circular_mask_fraction)
        mask[~valid] = 0
        n, lab, stat, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for i in range(1,n):
            if stat[i,cv2.CC_STAT_AREA] >= self.config.min_component_pixels: clean[lab==i]=255
        density = np.count_nonzero(clean)/max(1,np.count_nonzero(valid))
        component_sizes=stat[1:,cv2.CC_STAT_AREA] if n>1 else np.array([])
        largest=float(component_sizes.max()/max(component_sizes.sum(),1)) if len(component_sizes) else 0.
        # This is a preliminary ridge plausibility only; polar peak and tracking
        # confidence are authoritative later in the pipeline.
        confidence=float(np.clip(1-abs(density-.06)/.12,0,1)*(1-min(largest,.95)*.25))
        return SegmentationResult(clean, denoised, confidence, "traditional_ridge", {
            "ring_pixel_fraction":float(density),
            "largest_component_fraction":largest,
            "component_count":int(n - 1),
            "ridge_threshold":threshold,
        })


class UNetAdapter:
    """Optional adapter; callers own model loading/validation and domain evidence."""
    def __init__(self, predictor): self.predictor = predictor
    def segment(self, image, center, outer_radius):
        mask = np.asarray(self.predictor(image), dtype=np.uint8)
        if mask.shape != image.shape[:2]: raise ValueError("U-Net adapter returned wrong mask shape")
        return SegmentationResult((mask>0).astype(np.uint8)*255, to_gray(image), 0.5, "unet_adapter", {})
