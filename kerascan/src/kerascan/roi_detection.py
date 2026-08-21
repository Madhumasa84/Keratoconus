"""Full-resolution Placido-pattern localisation with manual fallbacks."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .config import ROIConfig
from .image_io import to_gray


@dataclass
class ROIResult:
    crop: np.ndarray
    box: tuple[int, int, int, int]
    center_full: tuple[float, float]
    center_roi: tuple[float, float]
    outer_radius_px: float
    confidence: float
    method: str


def _square_box(center: tuple[float, float], radius: float, shape: tuple[int, int], margin: float):
    h, w = shape[:2]
    half = max(1, int(round(radius * (1 + margin))))
    cx, cy = map(int, map(round, center))
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    # Shift instead of clipping one side, retaining a square crop when possible.
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= x1 - w; x1 = w
    if y1 > h: y0 -= y1 - h; y1 = h
    x0, y0 = max(0, x0), max(0, y0)
    side = min(x1 - x0, y1 - y0)
    x1, y1 = x0 + side, y0 + side
    return int(x0), int(y0), int(x1), int(y1)


def detect_placido_roi(image: np.ndarray, config: ROIConfig = ROIConfig()) -> ROIResult:
    """Locate ring-rich, approximately circular content without resizing input."""
    h, w = image.shape[:2]
    if config.manual_box is not None:
        x0, y0, x1, y1 = config.manual_box
        x0, y0, x1, y1 = max(0,x0), max(0,y0), min(w,x1), min(h,y1)
        if x1 <= x0 or y1 <= y0: raise ValueError("manual_box is outside image")
        crop = image[y0:y1, x0:x1].copy()
        c = config.manual_center or ((x0+x1)/2, (y0+y1)/2)
        return ROIResult(crop, (x0,y0,x1,y1), c, (c[0]-x0,c[1]-y0), min(x1-x0,y1-y0)/2, 1.0, "manual_box")
    if config.manual_center is not None:
        c = config.manual_center
        r = min(h,w) * 0.30
        box = _square_box(c, r, image.shape, config.margin)
        x0,y0,x1,y1 = box
        return ROIResult(image[y0:y1,x0:x1].copy(), box, c, (c[0]-x0,c[1]-y0), r, 1.0, "manual_center")
    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (0,0), 2.0)
    # Ring texture is high after difference-of-Gaussians; eye/background changes less.
    dog = cv2.absdiff(blur, cv2.GaussianBlur(blur, (0,0), 8.0))
    _, binary = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary)
    candidates = []
    for i in range(1, n):
        x,y,bw,bh,area = stats[i]
        if area < max(80, h*w*0.0005): continue
        r = max(bw,bh)/2
        if r < config.min_radius_px or r > min(h,w)*config.max_radius_fraction: continue
        aspect = min(bw,bh)/max(bw,bh)
        # Textured patterns can be non-solid; bounding-circle proxy is intentional.
        score = area * aspect * (1 - abs(1 - bw/max(bh,1))*0.25)
        candidates.append((score, tuple(cents[i]), r))
    # A true Placido centre makes many radial intensity transitions align. This score
    # distinguishes nested rings from the larger iris/eye contours without a learned model.
    circles = cv2.HoughCircles(cv2.medianBlur(gray,5), cv2.HOUGH_GRADIENT, 1.2,
                               minDist=max(30,min(h,w)//8), param1=90,param2=25,
                               minRadius=int(max(config.min_radius_px, min(h,w)*.10)), maxRadius=int(min(h,w)*config.max_radius_fraction))
    hough_candidates=[]
    if circles is not None:
        for x,y,r in circles[0]:
            t=np.arange(max(4,r*.06), min(r*1.4,min(h,w)*.48), 1.0)
            theta=np.linspace(0,2*np.pi,72,endpoint=False)
            xs=np.rint(x+np.outer(np.cos(theta),t)).astype(int); ys=np.rint(y+np.outer(np.sin(theta),t)).astype(int)
            valid=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
            # Candidates are already inside bounds; the validity mask protects border cases.
            profile=np.median(np.where(valid, gray[np.clip(ys,0,h-1),np.clip(xs,0,w-1)], np.nan),axis=0)
            # Normalise transitions by scale so one large eyelid/iris contour
            # cannot outscore the compact, densely nested Placido pattern.
            ringness=float(np.sum(np.abs(np.diff(profile))>5) / max(float(r), 1.0))
            hough_candidates.append((ringness,(float(x),float(y)),float(r)*1.35))
    if hough_candidates and max(hough_candidates,key=lambda q:q[0])[0] >= .12:
        score, center, radius=max(hough_candidates,key=lambda q:q[0])
        method, conf="hough_radial_ringness", min(.95,.35+score*1.8)
    elif candidates:
        _, center, radius = max(candidates, key=lambda q:q[0])
        method, conf = "texture_component", min(0.95, len(candidates)/4 + 0.35)
    else:
        center, radius, method, conf = (w/2,h/2), min(h,w)*0.22, "fallback_image_center", 0.0
    box = _square_box(center, radius, image.shape, config.margin)
    x0,y0,x1,y1 = box
    return ROIResult(image[y0:y1,x0:x1].copy(), box, center, (center[0]-x0,center[1]-y0), radius, conf, method)
