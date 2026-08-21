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


class Segmenter(Protocol):
    def segment(self, image: np.ndarray, center: tuple[float,float], outer_radius: float) -> SegmentationResult: ...


class TraditionalSegmenter:
    def __init__(self, config: SegmentationConfig = SegmentationConfig()): self.config = config
    def segment(self, image, center, outer_radius):
        gray = to_gray(image)
        clahe = cv2.createCLAHE(self.config.clahe_clip_limit, (self.config.clahe_tile_size,)*2)
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 5, 30, 30)
        edges = cv2.Canny(denoised, 35, 110)
        adaptive = cv2.adaptiveThreshold(denoised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,3)
        # Either bright/dark ring transitions are retained as sparse line evidence.
        mask = cv2.bitwise_or(edges, cv2.bitwise_and(adaptive, edges))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        yy,xx = np.ogrid[:gray.shape[0],:gray.shape[1]]
        rr = np.hypot(xx-center[0], yy-center[1])
        valid = (rr >= outer_radius*self.config.inner_exclusion_fraction) & (rr <= outer_radius*self.config.circular_mask_fraction)
        mask[~valid] = 0
        n, lab, stat, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for i in range(1,n):
            if stat[i,cv2.CC_STAT_AREA] >= self.config.min_component_pixels: clean[lab==i]=255
        density = np.count_nonzero(clean)/max(1,np.count_nonzero(valid))
        confidence = float(np.clip(1 - abs(density-0.055)/0.055, 0, 1))
        return SegmentationResult(clean, enhanced, confidence, "traditional")


class UNetAdapter:
    """Optional adapter; callers own model loading/validation and domain evidence."""
    def __init__(self, predictor): self.predictor = predictor
    def segment(self, image, center, outer_radius):
        mask = np.asarray(self.predictor(image), dtype=np.uint8)
        if mask.shape != image.shape[:2]: raise ValueError("U-Net adapter returned wrong mask shape")
        return SegmentationResult((mask>0).astype(np.uint8)*255, to_gray(image), 0.5, "unet_adapter")
