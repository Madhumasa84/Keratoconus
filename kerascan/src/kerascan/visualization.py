"""Auditable debug artefacts for the non-diagnostic image-analysis engine."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _display_crop(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image[:, :, :3].copy()


def _ring_rgba(index: int, count: int) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in plt.cm.hsv(index / max(count, 1)))


def _ring_bgr(index: int, count: int) -> tuple[int, int, int]:
    red, green, blue, _ = _ring_rgba(index, count)
    return int(blue * 255), int(green * 255), int(red * 255)


def _save_centres(crop: np.ndarray, initial: tuple[float, float], refined: tuple[float, float], output: Path) -> None:
    canvas = _display_crop(crop)
    cv2.drawMarker(canvas, tuple(map(int, initial)), (255, 160, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.drawMarker(canvas, tuple(map(int, refined)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    cv2.imwrite(str(output / "cropped_roi_centres.png"), canvas)


def _plot_polar(scan, output: Path) -> None:
    if scan is None:
        return
    extent = [0, 360, float(scan.radial_positions_px[0]), float(scan.radial_positions_px[-1])]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(scan.polar_image.T, aspect="auto", cmap="gray", origin="lower", extent=extent)
    ax.set(xlabel="Meridian (degrees)", ylabel="Radius (pixels)", title="Polar-unwrapped Placido pattern")
    fig.tight_layout()
    fig.savefig(output / "polar_unwrapped.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(scan.polar_smoothed.T, aspect="auto", cmap="gray", origin="lower", extent=extent)
    for angle, (radii, strengths) in enumerate(zip(scan.candidate_radii, scan.candidate_strengths)):
        if len(radii):
            ax.scatter(np.full(len(radii), scan.angles_deg[angle]), radii, s=np.maximum(3, strengths), c="#00bcd4", alpha=.55)
    if len(scan.reference_radii):
        ax.hlines(scan.reference_radii, 0, 360, colors="#ffb000", linewidth=.6, linestyles="dashed", label="aggregate references")
        ax.legend(loc="upper right")
    ax.set(xlabel="Meridian (degrees)", ylabel="Radius (pixels)", title="Detected radial peaks in polar space")
    fig.tight_layout()
    fig.savefig(output / "polar_detected_peaks.png", dpi=150)
    plt.close(fig)


def _plot_tracking(scan, tracked, output: Path) -> None:
    if scan is None or tracked is None:
        return
    extent = [0, 360, float(scan.radial_positions_px[0]), float(scan.radial_positions_px[-1])]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(scan.polar_smoothed.T, aspect="auto", cmap="gray", origin="lower", extent=extent, alpha=.75)
    for ring, row in enumerate(tracked.radii):
        colour = _ring_rgba(ring, len(tracked.radii))
        observed = np.flatnonzero(tracked.observed[ring])
        interpolated = np.flatnonzero(tracked.interpolated[ring])
        if len(observed):
            ax.scatter(scan.angles_deg[observed], row[observed], s=7, color=colour, label=f"ring {ring + 1}" if ring < 12 else None)
        if len(interpolated):
            ax.scatter(scan.angles_deg[interpolated], row[interpolated], s=12, marker="x", color=colour)
    ax.set(xlabel="Meridian (degrees)", ylabel="Radius (pixels)", title="Tracked ring identities in polar space")
    if len(tracked.radii) <= 12:
        ax.legend(loc="upper right", ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "polar_tracked_rings.png", dpi=150)
    plt.close(fig)


def _draw_cartesian_tracking(crop: np.ndarray, center: tuple[float, float], angles: np.ndarray, tracked, output: Path) -> None:
    canvas = _display_crop(crop)
    cv2.drawMarker(canvas, tuple(map(int, center)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    centreline = np.zeros(canvas.shape[:2], dtype=np.uint8)
    if tracked is not None:
        for ring, row in enumerate(tracked.radii):
            colour = _ring_bgr(ring, len(tracked.radii))
            ids = np.flatnonzero(np.isfinite(row))
            for angle in ids:
                x = int(round(center[0] + row[angle] * np.cos(np.deg2rad(angles[angle]))))
                y = int(round(center[1] + row[angle] * np.sin(np.deg2rad(angles[angle]))))
                if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                    cv2.circle(canvas, (x, y), 1, colour, -1)
                    cv2.circle(centreline, (x, y), 1, 255, -1)
    cv2.imwrite(str(output / "tracked_rings_cartesian.png"), canvas)
    # Existing consumers expect this historic file name.
    cv2.imwrite(str(output / "centre_and_detected_rings.png"), canvas)
    cv2.imwrite(str(output / "ring_centreline_mask.png"), centreline)


def _plot_completeness_and_states(scan, tracked, output: Path) -> None:
    if scan is None or tracked is None:
        return
    fig, ax = plt.subplots(figsize=(9, 3.5))
    rings = np.arange(1, len(tracked.ring_completeness) + 1)
    ax.bar(rings, tracked.ring_completeness, color=[_ring_rgba(index, len(rings)) for index in range(len(rings))])
    ax.set(ylim=(0, 1), xlabel="Tracked ring identity", ylabel="Completeness", title="Per-ring completeness")
    fig.tight_layout()
    fig.savefig(output / "per_ring_completeness.png", dpi=150)
    plt.close(fig)

    # 0 observed, 1 interpolated, 2 missing, 3 rejected.  These are states,
    # not fabricated radii; short interpolation is visibly distinct.
    states = np.full(tracked.radii.shape, 2, dtype=np.uint8)
    states[tracked.observed] = 0
    states[tracked.interpolated] = 1
    states[tracked.rejected] = 3
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(states) * .23)))
    image = ax.imshow(states, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#2ca02c", "#ffbf00", "#d62728", "#111111"]), vmin=0, vmax=3)
    ax.set(xlabel="Meridian index (0° at left)", ylabel="Tracked ring identity", title="Observed / interpolated / missing / rejected sectors")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["observed", "short-gap interpolated", "missing", "rejected outlier"])
    fig.tight_layout()
    fig.savefig(output / "missing_sector_map.png", dpi=150)
    plt.close(fig)


def _plot_spacing(angles: np.ndarray, tracked, output: Path) -> None:
    if tracked is None or len(tracked.radii) < 2:
        return
    spacing = np.diff(tracked.radii, axis=0)
    # The tracker guarantees ordered radii.  Retain NaN gaps and make a second
    # defensive conversion so an invalid value can never appear in this plot.
    spacing[~np.isfinite(spacing) | (spacing <= 0)] = np.nan
    fig, ax = plt.subplots(figsize=(10, 4))
    for ring, row in enumerate(spacing):
        if np.any(np.isfinite(row)):
            ax.plot(angles, row, color=_ring_rgba(ring + 1, len(tracked.radii)), linewidth=.9, label=f"r{ring + 1}→r{ring + 2}")
    counts = np.sum(np.isfinite(spacing), axis=0)
    mean = np.divide(np.nansum(spacing, axis=0), counts, out=np.full(len(angles), np.nan), where=counts > 0)
    ax.plot(angles, mean, color="black", linewidth=1.4, linestyle="--", label="mean")
    ax.set(xlabel="Meridian (degrees)", ylabel="Ring spacing (pixels)", title="Directional spacing (image-space proxy)")
    if len(spacing) <= 12:
        ax.legend(loc="upper right", ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "directional_spacing.png", dpi=150)
    plt.close(fig)


def save_visualizations(result: dict, output_dir: str | Path):
    """Save auditable debug plots without promoting any clinical conclusion."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = result["_artifacts"]
    crop = artifacts["roi_crop"]
    initial = artifacts.get("initial_center", artifacts["center"])
    center = artifacts["center"]
    _save_centres(crop, initial, center, output)
    scan = artifacts.get("scan")
    tracked = artifacts.get("tracking_result")
    _plot_polar(scan, output)
    _plot_tracking(scan, tracked, output)
    angles = artifacts.get("angles", np.empty(0))
    _draw_cartesian_tracking(crop, center, angles, tracked, output)
    _plot_completeness_and_states(scan, tracked, output)
    _plot_spacing(angles, tracked, output)
    
    geometry_grids = artifacts.get("geometry_grids")
    if geometry_grids:
        _plot_geometry_grids(angles, tracked, geometry_grids, output)

    features = result.get("features")
    lines = []
    if features:
        lines.extend(f"{key}: {value:.6g}" for key, value in features.items())
    else:
        lines.append("Model-ready geometry features: not calculated")
    lines.extend(
        [
            "",
            f"Result: {result['screening_result']} ({result.get('experimental_status', 'experimental')})",
            f"Analysis status: {result['analysis_status']}; failure stage: {result['failure_stage']}",
            f"Acquisition warnings: {', '.join(result.get('acquisition_quality', {}).get('flags', [])) or 'none'}",
            f"Segmentation warnings: {', '.join(result.get('segmentation', {}).get('flags', [])) or 'none'}",
            f"Tracking warnings: {', '.join(result.get('tracking', {}).get('flags', [])) or 'none'}",
        ]
    )
    (output / "features.txt").write_text("\n".join(lines) + "\n")


def save_roi_overlay(original: np.ndarray, box: tuple[int, int, int, int], output_dir: str | Path):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overlay = _display_crop(original)
    cv2.rectangle(overlay, box[:2], box[2:], (0, 255, 0), 2)
    cv2.imwrite(str(output / "detected_roi_box.png"), overlay)

def _plot_geometry_grids(angles: np.ndarray, tracked, geometry_grids: dict, output: Path) -> None:
    if tracked is None or not geometry_grids:
        return
    import matplotlib.pyplot as plt
    
    # 1. Normalized spacing heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(geometry_grids["normalized_spacing"], aspect="auto", cmap="viridis", origin="lower", 
                   extent=[angles[0], angles[-1], 1, len(tracked.radii)-1])
    fig.colorbar(im, ax=ax, label="Normalized Spacing")
    ax.set(xlabel="Meridian (degrees)", ylabel="Ring Pair", title="Normalized Spacing Heatmap")
    fig.savefig(output / "normalized_spacing.png", dpi=150)
    plt.close(fig)

    # 2. Opposite-meridian asymmetry
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(geometry_grids["opp_asym_grid"], aspect="auto", cmap="magma", origin="lower", 
                   extent=[angles[0], angles[-1], 1, len(tracked.radii)-1])
    fig.colorbar(im, ax=ax, label="Asymmetry (relative)")
    ax.set(xlabel="Meridian (degrees)", ylabel="Ring Pair", title="Opposite-Meridian Asymmetry")
    fig.savefig(output / "opposite_meridian_asymmetry.png", dpi=150)
    plt.close(fig)

    # 3. Compression heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(geometry_grids["compression_grid"], aspect="auto", cmap="Reds", origin="lower", 
                   extent=[angles[0], angles[-1], 1, len(tracked.radii)-1])
    fig.colorbar(im, ax=ax, label="Compression")
    ax.set(xlabel="Meridian (degrees)", ylabel="Ring Pair", title="Local Compression Heatmap")
    fig.savefig(output / "compression_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Multi-ring agreement map
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(angles, geometry_grids["multiring_agreement_grid"], color='blue')
    ax.set(xlabel="Meridian (degrees)", ylabel="Std Dev", title="Multi-Ring Spatial Agreement")
    fig.tight_layout()
    fig.savefig(output / "multi_ring_agreement.png", dpi=150)
    plt.close(fig)

    # 5. Residual plot
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(geometry_grids["residuals"], aspect="auto", cmap="coolwarm", origin="lower", 
                   extent=[angles[0], angles[-1], 1, len(tracked.radii)], vmin=-2, vmax=2)
    fig.colorbar(im, ax=ax, label="Residual (px)")
    ax.set(xlabel="Meridian (degrees)", ylabel="Ring Index", title="Fourier Shape Residuals")
    fig.savefig(output / "shape_residuals.png", dpi=150)
    plt.close(fig)
