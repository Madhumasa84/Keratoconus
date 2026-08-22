"""Auditable debug artefacts for the non-diagnostic image-analysis engine."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


MULTIRING_AGREEMENT_Y_LABEL = "Coherent-run weighted deviation"


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
    cv2.imwrite(str(output / "full_stack_tracked_rings.png"), canvas)
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
        _plot_full_stack_outputs(
            angles,
            tracked,
            geometry_grids,
            result.get("full_stack_analysis", {}),
            output,
        )
        if result.get("reference_geometry"):
            _plot_reference_outputs(
                crop,
                center,
                angles,
                tracked,
                geometry_grids,
                result,
                output,
            )

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
    reference = result.get("reference_geometry")
    if reference:
        concentric = reference.get("concentric_reference", {})
        smooth = reference.get("smooth_reference", {})
        invalid_rings = [
            str(index + 1)
            for index, valid in enumerate(concentric.get("valid_by_ring", []))
            if not valid
        ]
        lines.extend(
            [
                "",
                f"Reference geometry: {reference.get('reference_type', 'unavailable')}",
                f"Validated normal reference: {reference.get('validated_normal_reference', False)}",
                f"Concentric reference valid: {concentric.get('valid', False)}",
                f"Smooth reference valid: {smooth.get('valid', False)}",
                f"Invalid reference rings: {', '.join(invalid_rings) or 'none'}",
                f"Hardware ring count verified: {reference.get('ring_count_verified', False)}",
                "Reference limitation: fitted from this eye; not a validated normal template.",
            ]
        )
    (output / "features.txt").write_text("\n".join(lines) + "\n")

    # Generate clinician comparison panel if reference geometry is available
    if geometry_grids and result.get("reference_geometry"):
        _plot_clinician_comparison_panel(
            crop, center, angles, tracked, geometry_grids, result, output
        )


def _plot_clinician_comparison_panel(
    crop: np.ndarray,
    center: tuple[float, float],
    angles: np.ndarray,
    tracked,
    grids: dict,
    result: dict,
    output: Path,
) -> None:
    """Generate a natural clinician-readable 4-part comparison panel.

    Panel layout:
      1. De-identified cropped KeraScan ROI
      2. Observed rings vs concentric mathematical reference
      3. Simplified spacing-deviation map (not a raw polar plot)
      4. Plain-language geometry summary

    This panel is suitable for inclusion in a referral report for affected eyes.
    It does NOT include full facial images, local paths, or debug plots.
    """
    if tracked is None or not len(angles):
        return

    reference = result.get("reference_geometry", {})
    circle_reference = np.asarray(grids.get("reference_circle_reference", np.full_like(np.zeros((1, 1)), np.nan)), dtype=float)
    circle_valid = np.asarray(grids.get("reference_valid_concentric_by_ring", np.zeros(1, dtype=bool)), dtype=bool)
    circle_deviation = np.asarray(grids.get("reference_circle_deviation", np.full_like(np.zeros((1, 1)), np.nan)), dtype=float)

    if circle_reference.shape[0] == 0 or circle_valid.shape[0] == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Panel 1: De-identified cropped ROI ---
    ax1 = axes[0, 0]
    ax1.imshow(cv2.cvtColor(_display_crop(crop), cv2.COLOR_BGR2RGB))
    ax1.scatter([center[0]], [center[1]], marker="+", s=80, linewidths=2, color="#ff3030")
    ax1.set(xticks=[], yticks=[], title="KeraScan Placido Image\n(de-identified cropped ROI)")

    # --- Panel 2: Observed rings vs concentric reference ---
    ax2 = axes[0, 1]
    _plot_observed_reference_axis(
        ax2, crop, center, angles, tracked, circle_reference, circle_valid,
        reference_kind="concentric", legend=True,
    )
    ax2.set_title(
        "Observed Rings vs Mathematical Reference\n"
        "Solid = tracked; Dashed = concentric reference"
    )

    # --- Panel 3: Simplified spacing deviation map ---
    ax3 = axes[1, 0]
    if circle_deviation.ndim == 2 and circle_deviation.shape[0] > 0:
        finite_dev = np.abs(circle_deviation[np.isfinite(circle_deviation)])
        dev_limit = max(3.0, float(np.percentile(finite_dev, 95))) if finite_dev.size else 3.0
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#f0f0f0")
        im = ax3.imshow(
            np.ma.masked_invalid(circle_deviation),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=-dev_limit,
            vmax=dev_limit,
        )
        ax3.set(
            xlabel="Meridian index (0° = right)",
            ylabel="Ring identity (1 = innermost)",
            title="Ring Position Deviation from Reference\n(pixels; red = outward, blue = inward)",
            yticks=np.arange(circle_deviation.shape[0]),
            yticklabels=[str(i + 1) for i in range(circle_deviation.shape[0])],
        )
        fig.colorbar(im, ax=ax3, label="Deviation (pixels)")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "Deviation map\nunavailable", ha="center", va="center", transform=ax3.transAxes)

    # --- Panel 4: Plain-language geometry summary ---
    ax4 = axes[1, 1]
    ax4.axis("off")

    concentric = reference.get("concentric_reference", {})
    smooth = reference.get("smooth_reference", {})
    gates = result.get("gates", {})
    detected_rings = reference.get("detected_ring_count", "unknown")
    expected_rings = reference.get("expected_ring_count")
    geometry_status = result.get("geometry_status", "unavailable")

    # Build plain-language summary lines
    summary_lines = ["GEOMETRY SUMMARY", ""]

    ring_line = f"Detected rings: {detected_rings}"
    if expected_rings:
        ring_line += f" of {expected_rings} expected"
    summary_lines.append(ring_line)

    concentric_valid = concentric.get("valid", False)
    smooth_valid = smooth.get("valid", False)
    summary_lines.append(
        f"Concentric reference: {'computed' if concentric_valid else 'insufficient data'}"
    )
    summary_lines.append(
        f"Smooth reference: {'computed' if smooth_valid else 'insufficient data'}"
    )

    coverage_vals = concentric.get("coverage_by_ring")
    if coverage_vals:
        mean_cov = float(np.mean(coverage_vals)) * 100
        summary_lines.append(f"Mean ring coverage: {mean_cov:.0f}%")

    summary_lines.append("")

    # Gate status
    hw_gate = gates.get("verified_hardware_ring_count", "unavailable")
    threshold_gate = gates.get("approved_geometry_thresholds", "unavailable")
    summary_lines.append(f"Hardware config: {hw_gate}")
    summary_lines.append(f"Threshold config: {threshold_gate}")
    summary_lines.append(f"Status: {geometry_status}")

    summary_lines += [
        "",
        "─" * 32,
        "Solid lines: tracked ring positions",
        "Dashed lines: mathematical reference",
        "Reference fitted from this eye only.",
        "Not a validated normal template.",
        "This is an engineering comparison,",
        "not a diagnostic corneal map.",
    ]

    summary_text = "\n".join(summary_lines)
    ax4.text(
        0.05, 0.95, summary_text,
        transform=ax4.transAxes, va="top", ha="left",
        fontsize=9.5, linespacing=1.55,
        fontfamily="monospace",
        bbox={"facecolor": "#f8f8f8", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.5"},
    )

    fig.suptitle(
        "KeraScan Engineering Comparison Panel\n"
        "INITIAL SCREENING AID — NOT A DIAGNOSIS",
        fontsize=13, fontweight="bold", color="#003366",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "clinician_comparison_panel.png", dpi=170)
    plt.close(fig)


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
    ax.set(
        xlabel="Meridian (degrees)",
        ylabel=MULTIRING_AGREEMENT_Y_LABEL,
        title="Legacy multi-ring coherence proxy (not standard deviation)",
    )
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


def _heatmap(
    values: np.ndarray,
    angles: np.ndarray,
    output_file: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    pair_states: np.ndarray | None = None,
    ylabel: str = "Adjacent ring pair (1 = innermost)",
) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        return
    masked = np.ma.masked_invalid(values)
    colour_map = plt.get_cmap(cmap).copy()
    colour_map.set_bad("#f2f2f2")
    step = float(np.median(np.diff(angles))) if len(angles) > 1 else 360.0
    fig, ax = plt.subplots(figsize=(11, max(3.2, values.shape[0] * 0.38)))
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[float(angles[0]), float(angles[-1] + step), 0.5, values.shape[0] + 0.5],
        cmap=colour_map,
        vmin=vmin,
        vmax=vmax,
    )
    if pair_states is not None and np.asarray(pair_states).shape == values.shape:
        pair_states = np.asarray(pair_states)
        for state, marker, colour, label in (
            (1, ".", "#111111", "interpolated support"),
            (3, "x", "#555555", "rejected/invalid"),
        ):
            pair_ids, angle_ids = np.where(pair_states == state)
            if len(pair_ids):
                ax.scatter(angles[angle_ids], pair_ids + 1, s=10, marker=marker, color=colour, label=label)
        if np.any((pair_states == 1) | (pair_states == 3)):
            ax.legend(loc="upper right", fontsize=7)
    ax.set(
        xlabel="Meridian angle (degrees)",
        ylabel=ylabel,
        title=title,
        yticks=np.arange(1, values.shape[0] + 1),
        yticklabels=[str(index) for index in range(1, values.shape[0] + 1)],
    )
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _plot_full_stack_spacing_matrix(
    angles: np.ndarray,
    tracked,
    grids: dict,
    stack: dict,
    output: Path,
) -> None:
    """Save the ten audit plots required for complete-stack review."""
    if tracked is None or not stack:
        return
    pair_states = grids.get("pair_states")
    _heatmap(
        grids["spacing_matrix"],
        angles,
        output / "inter_ring_spacing_matrix.png",
        title="Full-stack adjacent spacing (image-space proxy)",
        colorbar_label="Spacing (pixels)",
        cmap="cividis",
        pair_states=pair_states,
    )


def _reference_grid(grids: dict, name: str) -> np.ndarray:
    return np.asarray(grids[f"reference_{name}"])


def _symmetric_limit(values: np.ndarray, minimum: float) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    return max(minimum, float(np.percentile(finite, 98))) if finite.size else minimum


def _plot_observed_reference_axis(
    ax,
    crop: np.ndarray,
    center: tuple[float, float],
    angles: np.ndarray,
    tracked,
    reference: np.ndarray,
    valid_by_ring: np.ndarray,
    *,
    reference_kind: str,
    legend: bool = True,
) -> None:
    """Draw direct observations and a visually distinct fitted reference."""
    from matplotlib.lines import Line2D

    ax.imshow(cv2.cvtColor(_display_crop(crop), cv2.COLOR_BGR2RGB))
    theta = np.deg2rad(angles)
    for ring, row in enumerate(tracked.radii):
        colour = _ring_rgba(ring, len(tracked.radii))
        direct_row = np.where(tracked.observed[ring], row, np.nan)
        observed_x = center[0] + direct_row * np.cos(theta)
        observed_y = center[1] + direct_row * np.sin(theta)
        ax.plot(observed_x, observed_y, color=colour, linewidth=1.35, solid_capstyle="round")
        if not valid_by_ring[ring] or not np.any(np.isfinite(reference[ring])):
            continue
        if reference_kind == "concentric":
            dense_theta = np.linspace(0.0, 2.0 * np.pi, 721)
            radius = float(reference[ring, np.flatnonzero(np.isfinite(reference[ring]))[0]])
            reference_x = center[0] + radius * np.cos(dense_theta)
            reference_y = center[1] + radius * np.sin(dense_theta)
            reference_colour = "white"
        else:
            reference_x = center[0] + reference[ring] * np.cos(theta)
            reference_y = center[1] + reference[ring] * np.sin(theta)
            reference_colour = colour
        ax.plot(
            reference_x,
            reference_y,
            color=reference_colour,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            alpha=0.9,
        )
        first = np.flatnonzero(np.isfinite(reference[ring]))[0]
        label_x = center[0] + reference[ring, first] * np.cos(theta[first])
        label_y = center[1] + reference[ring, first] * np.sin(theta[first])
        ax.text(label_x, label_y, str(ring + 1), color="white", fontsize=6, ha="left", va="bottom")
    ax.scatter([center[0]], [center[1]], marker="x", s=55, linewidths=1.4, color="#ff3030")
    ax.set(xlim=(0, crop.shape[1]), ylim=(crop.shape[0], 0), xticks=[], yticks=[])
    if legend:
        reference_label = (
            "Concentric mathematical reference"
            if reference_kind == "concentric"
            else "Smooth fitted reference"
        )
        ax.legend(
            handles=[
                Line2D([0], [0], color="#00d5ff", linewidth=1.5, label="Observed tracked ring"),
                Line2D([0], [0], color="white" if reference_kind == "concentric" else "#00d5ff",
                       linewidth=1.0, linestyle=(0, (4, 3)), label=reference_label),
            ],
            loc="lower right",
            fontsize=7,
            framealpha=0.75,
        )


def _plot_reference_spacing_residuals(
    angles: np.ndarray,
    grids: dict,
    output: Path,
) -> None:
    circle = _reference_grid(grids, "normalized_circle_spacing_residual")
    smooth = _reference_grid(grids, "normalized_smooth_spacing_residual")
    pair_states = _reference_grid(grids, "pair_states")
    limit = max(_symmetric_limit(circle, 0.10), _symmetric_limit(smooth, 0.10))
    step = float(np.median(np.diff(angles))) if len(angles) > 1 else 360.0
    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("#f2f2f2")
    fig, axes = plt.subplots(2, 1, figsize=(11, max(6.0, circle.shape[0] * 0.65)), sharex=True)
    for ax, values, title in (
        (axes[0], circle, "Observed spacing − concentric-reference spacing"),
        (axes[1], smooth, "Observed spacing − smooth-reference spacing"),
    ):
        image = ax.imshow(
            np.ma.masked_invalid(values), aspect="auto", interpolation="nearest", origin="lower",
            extent=[float(angles[0]), float(angles[-1] + step), 0.5, values.shape[0] + 0.5],
            cmap=cmap, vmin=-limit, vmax=limit,
        )
        rejected_pairs, rejected_angles = np.where(pair_states == 3)
        if len(rejected_pairs):
            ax.scatter(angles[rejected_angles], rejected_pairs + 1, marker="x", s=10, color="#333333")
        ax.set(ylabel="Adjacent ring pair", title=title, yticks=np.arange(1, values.shape[0] + 1))
        fig.colorbar(image, ax=ax, label="Residual / reference spacing")
    axes[-1].set(xlabel="Meridian angle (degrees)")
    fig.suptitle("Reference spacing residuals — dimensionless image-space proxy; not diagnostic")
    fig.tight_layout()
    fig.savefig(output / "reference_spacing_residual_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_reference_outputs(
    crop: np.ndarray,
    center: tuple[float, float],
    angles: np.ndarray,
    tracked,
    grids: dict,
    result: dict,
    output: Path,
) -> None:
    """Render the seven artificial-reference engineering outputs."""
    if tracked is None or not len(angles):
        return
    reference = result["reference_geometry"]
    circle_reference = _reference_grid(grids, "circle_reference").astype(float)
    smooth_reference = _reference_grid(grids, "smooth_reference").astype(float)
    circle_valid = _reference_grid(grids, "valid_concentric_by_ring").astype(bool)
    smooth_valid = _reference_grid(grids, "valid_smooth_by_ring").astype(bool)
    ring_states = _reference_grid(grids, "ring_states")

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _plot_observed_reference_axis(
        ax, crop, center, angles, tracked, circle_reference, circle_valid,
        reference_kind="concentric",
    )
    ax.set_title("Observed tracked rings vs concentric self-fitted reference\nEngineering comparison; not a validated normal template")
    fig.tight_layout(); fig.savefig(output / "observed_vs_concentric_reference.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _plot_observed_reference_axis(
        ax, crop, center, angles, tracked, smooth_reference, smooth_valid,
        reference_kind="smooth",
    )
    ax.set_title("Observed tracked rings vs smooth low-order reference\nSolid = directly observed; dashed = fitted through gaps")
    fig.tight_layout(); fig.savefig(output / "observed_vs_smooth_reference.png", dpi=180); plt.close(fig)

    circle_deviation = _reference_grid(grids, "circle_deviation").astype(float)
    theta = np.deg2rad(angles)
    vector_step = max(1, len(angles) // 24)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(cv2.cvtColor(_display_crop(crop), cv2.COLOR_BGR2RGB))
    inward_labelled = outward_labelled = False
    for ring in range(len(tracked.radii)):
        if not circle_valid[ring]:
            continue
        for angle in range(0, len(angles), vector_step):
            deviation = circle_deviation[ring, angle]
            if not np.isfinite(deviation):
                continue
            reference_radius = circle_reference[ring, angle]
            observed_radius = reference_radius + deviation
            start = (center[0] + reference_radius * np.cos(theta[angle]), center[1] + reference_radius * np.sin(theta[angle]))
            change = (deviation * np.cos(theta[angle]), deviation * np.sin(theta[angle]))
            inward = deviation < 0.0
            label = None
            if inward and not inward_labelled:
                label = "inward deviation"; inward_labelled = True
            if not inward and not outward_labelled:
                label = "outward deviation"; outward_labelled = True
            ax.arrow(
                start[0], start[1], change[0], change[1],
                color="#2166ac" if inward else "#b2182b", width=0.08,
                head_width=1.2, head_length=1.4, length_includes_head=True,
                alpha=0.8, label=label,
            )
            ax.scatter(
                [center[0] + observed_radius * np.cos(theta[angle])],
                [center[1] + observed_radius * np.sin(theta[angle])],
                s=3, color="#eeeeee",
            )
    ax.scatter([center[0]], [center[1]], marker="x", s=55, color="#ff3030")
    ax.set(xlim=(0, crop.shape[1]), ylim=(crop.shape[0], 0), xticks=[], yticks=[],
           title="Sparse radial deviation vectors (1× magnification; pixels)")
    if inward_labelled or outward_labelled:
        ax.legend(loc="lower right", fontsize=8)
    ax.text(0.01, 0.01, f"Vector sampling: every {vector_step * (360.0 / len(angles)):.1f}°; magnification 1×",
            transform=ax.transAxes, color="white", fontsize=8, backgroundcolor="#333333")
    fig.tight_layout(); fig.savefig(output / "radial_deviation_vectors.png", dpi=180); plt.close(fig)

    circle_limit = _symmetric_limit(circle_deviation, 1.0)
    _heatmap(
        circle_deviation, angles, output / "circle_deviation_heatmap.png",
        title="Signed radial deviation from concentric reference (image-space proxy; not diagnostic)",
        colorbar_label="Observed radius − reference radius (pixels)", cmap="coolwarm",
        vmin=-circle_limit, vmax=circle_limit, pair_states=ring_states,
        ylabel="Tracked ring identity (1 = innermost)",
    )
    smooth_deviation = _reference_grid(grids, "smooth_deviation").astype(float)
    smooth_limit = _symmetric_limit(smooth_deviation, 1.0)
    _heatmap(
        smooth_deviation, angles, output / "smooth_residual_heatmap.png",
        title="Signed residual from smooth low-order reference (image-space proxy; not diagnostic)",
        colorbar_label="Observed radius − smooth fitted radius (pixels)", cmap="coolwarm",
        vmin=-smooth_limit, vmax=smooth_limit, pair_states=ring_states,
        ylabel="Tracked ring identity (1 = innermost)",
    )
    _plot_reference_spacing_residuals(angles, grids, output)

    normalized_spacing = _reference_grid(grids, "normalized_smooth_spacing_residual").astype(float)
    spacing_limit = _symmetric_limit(normalized_spacing, 0.10)
    coverage = np.asarray(reference["concentric_reference"]["coverage_by_ring"], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(cv2.cvtColor(_display_crop(crop), cv2.COLOR_BGR2RGB))
    axes[0, 0].scatter([center[0]], [center[1]], marker="x", color="#ff3030")
    axes[0, 0].set(title="De-identified cropped ROI", xticks=[], yticks=[])
    _plot_observed_reference_axis(
        axes[0, 1], crop, center, angles, tracked, circle_reference, circle_valid,
        reference_kind="concentric", legend=False,
    )
    axes[0, 1].set_title("Observed solid / concentric reference dashed")
    image = axes[0, 2].imshow(np.ma.masked_invalid(circle_deviation), aspect="auto", origin="lower",
                              cmap="coolwarm", vmin=-circle_limit, vmax=circle_limit)
    axes[0, 2].set(title="Circle deviation", xlabel="Meridian index", ylabel="Ring identity")
    fig.colorbar(image, ax=axes[0, 2], label="pixels")
    image = axes[1, 0].imshow(np.ma.masked_invalid(normalized_spacing), aspect="auto", origin="lower",
                              cmap="coolwarm", vmin=-spacing_limit, vmax=spacing_limit)
    axes[1, 0].set(title="Smooth-reference spacing residual", xlabel="Meridian index", ylabel="Ring pair")
    fig.colorbar(image, ax=axes[1, 0], label="residual / reference spacing")
    axes[1, 1].bar(np.arange(1, len(coverage) + 1), coverage,
                   color=[_ring_rgba(index, len(coverage)) for index in range(len(coverage))])
    axes[1, 1].axhline(
        float(reference["minimum_direct_coverage_required"]),
        color="#555555", linestyle="--", linewidth=0.8, label="configured minimum",
    )
    axes[1, 1].set(ylim=(0, 1), xlabel="Ring identity", ylabel="Direct coverage", title="Per-ring direct support")
    axes[1, 1].legend(fontsize=7)
    gates = result.get("gates", {})
    status_text = "\n".join((
        f"Geometry status: {result.get('geometry_status', 'unavailable')}",
        f"Hardware ring-count gate: {gates.get('verified_hardware_ring_count', 'unavailable')}",
        f"Approved-threshold gate: {gates.get('approved_geometry_thresholds', 'unavailable')}",
        f"Expected rings: {reference.get('expected_ring_count')}",
        f"Detected rings: {reference.get('detected_ring_count')}",
        f"Concentric reference valid: {reference['concentric_reference']['valid']}",
        f"Smooth reference valid: {reference['smooth_reference']['valid']}",
        "Self-fitted reference ≠ validated normal reference",
    ))
    axes[1, 2].axis("off"); axes[1, 2].text(0.02, 0.96, status_text, va="top", fontsize=10, linespacing=1.5)
    fig.suptitle("MATHEMATICAL ENGINEERING COMPARISON\nNOT A CLINICAL DIAGNOSIS", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "full_stack_reference_comparison.png", dpi=170)
    plt.close(fig)


def _plot_full_stack_outputs(
    angles: np.ndarray,
    tracked,
    grids: dict,
    stack: dict,
    output: Path,
) -> None:
    """Save the ten audit plots required for complete-stack review."""
    if tracked is None or not stack:
        return
    pair_states = grids.get("pair_states")
    _plot_full_stack_spacing_matrix(angles, tracked, grids, stack, output)
    _heatmap(
        grids["normalized_spacing"],
        angles,
        output / "normalized_inter_ring_spacing_matrix.png",
        title="Ring-pair normalized spacing (1 = pair angular median; not a clinical normal)",
        colorbar_label="Normalized spacing (dimensionless image-space proxy)",
        cmap="coolwarm",
        vmin=0.5,
        vmax=1.5,
        pair_states=pair_states,
    )

    variation = np.asarray(grids["angular_variation"], dtype=float)
    angular_range = np.asarray(grids["angular_range"], dtype=float)
    pair_ids = np.arange(1, len(variation) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pair_ids - 0.18, variation, width=0.36, color="#3366cc", label="MAD / pair median")
    ax.bar(pair_ids + 0.18, angular_range, width=0.36, color="#dc7533", label="(P90-P10) / pair median")
    ax.set(
        xlabel="Adjacent ring pair (1 = innermost)",
        ylabel="Dimensionless image-space proxy",
        title="Angular spacing variation for every adjacent ring pair",
        xticks=pair_ids,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "angular_variation_by_ring_pair.png", dpi=150)
    plt.close(fig)

    radial_deviation = np.asarray(grids["radial_stack_deviation"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(angles, radial_deviation, color="#4b0082", linewidth=1.2)
    ax.set(
        xlim=(0, 360),
        xlabel="Meridian angle (degrees)",
        ylabel="Median |normalized spacing - 1|",
        title="Robust radial-stack deviation by meridian (image-space proxy)",
    )
    fig.tight_layout()
    fig.savefig(output / "radial_stack_deviation_by_meridian.png", dpi=150)
    plt.close(fig)

    longest_compressed = np.asarray(grids["longest_compressed_pair_run"], dtype=float)
    longest_expanded = np.asarray(grids["longest_expanded_pair_run"], dtype=float)
    compressed_fraction = np.asarray(grids["compression_stack_fraction"], dtype=float)
    expanded_fraction = np.asarray(grids["expansion_stack_fraction"], dtype=float)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    top.step(angles, longest_compressed, where="mid", color="#2166ac", label="compressed pair run")
    top.step(angles, longest_expanded, where="mid", color="#b2182b", label="expanded pair run")
    top.set(ylabel="Longest neighbouring-pair run", title="Neighbouring-ring coherence (direct observations only)")
    top.legend()
    bottom.plot(angles, compressed_fraction, color="#2166ac", label="compressed stack fraction")
    bottom.plot(angles, expanded_fraction, color="#b2182b", label="expanded stack fraction")
    bottom.set(xlim=(0, 360), ylim=(0, 1), xlabel="Meridian angle (degrees)", ylabel="Stack fraction")
    bottom.legend()
    fig.tight_layout()
    fig.savefig(output / "neighbouring_ring_coherence.png", dpi=150)
    plt.close(fig)

    cumulative = np.asarray(grids["normalized_cumulative_residual"], dtype=float)
    finite_cumulative = np.abs(cumulative[np.isfinite(cumulative)])
    cumulative_limit = max(0.10, float(np.percentile(finite_cumulative, 95))) if finite_cumulative.size else 0.10
    _heatmap(
        cumulative,
        angles,
        output / "cumulative_radial_residual.png",
        title="Normalized cumulative inner-to-outer residual (image-space proxy)",
        colorbar_label="Residual / expected cumulative radius",
        cmap="coolwarm",
        vmin=-cumulative_limit,
        vmax=cumulative_limit,
    )

    ring_items = stack.get("ring_completeness", [])
    pair_items = stack.get("ring_pair_completeness", [])
    fig, (ring_ax, pair_ax) = plt.subplots(2, 1, figsize=(11, 7))
    colours = ["#2ca02c", "#ffbf00", "#d62728", "#111111"]
    labels = ["observed", "interpolated", "missing", "rejected"]
    for ax, items, label in ((ring_ax, ring_items, "Ring"), (pair_ax, pair_items, "Adjacent ring pair")):
        x = np.arange(1, len(items) + 1)
        bottom_values = np.zeros(len(items))
        for field, colour, state_label in zip(
            ("observed_fraction", "interpolated_fraction", "missing_fraction", "rejected_fraction"),
            colours,
            labels,
        ):
            values = np.asarray([item[field] for item in items], dtype=float)
            ax.bar(x, values, bottom=bottom_values, color=colour, label=state_label)
            bottom_values += values
        ax.set(ylim=(0, 1), ylabel="Fraction", xlabel=f"{label} identity", xticks=x)
    ring_ax.set_title("Per-ring and adjacent-pair completeness")
    ring_ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "ring_and_pair_completeness.png", dpi=150)
    plt.close(fig)

    regions = [
        stack.get("inner_region_summary", {}),
        stack.get("middle_region_summary", {}),
        stack.get("outer_region_summary", {}),
    ]
    region_x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(region_x - 0.18, [item.get("ring_observed_fraction", 0.0) for item in regions], 0.36, label="ring direct coverage")
    ax.bar(region_x + 0.18, [item.get("pair_direct_observation_fraction", 0.0) for item in regions], 0.36, label="pair direct coverage")
    ax.set(
        ylim=(0, 1),
        ylabel="Direct observation fraction",
        xticks=region_x,
        xticklabels=["Inner", "Middle", "Outer"],
        title="Inner / middle / outer full-stack coverage comparison",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "inner_middle_outer_comparison.png", dpi=150)
    plt.close(fig)

    normalized = np.asarray(grids["normalized_spacing"], dtype=float)
    compression_cells = np.asarray(grids["coherent_compression"], dtype=bool)
    expansion_cells = np.asarray(grids["coherent_expansion"], dtype=bool)
    sector_map = np.full_like(normalized, np.nan)
    sector_map[compression_cells] = normalized[compression_cells] - 1.0
    sector_map[expansion_cells] = normalized[expansion_cells] - 1.0
    _heatmap(
        sector_map,
        angles,
        output / "full_stack_sector_map.png",
        title="Coherent full-stack compression / expansion sectors (not diagnostic)",
        colorbar_label="Normalized spacing - 1 (dimensionless image-space proxy)",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
        pair_states=pair_states,
    )
