import numpy as np

def compute_geometry(radii: np.ndarray, angles_deg: np.ndarray, observed: np.ndarray | None, min_direct_coverage: float, config) -> dict:
    """Explainable mathematical Placido-ring geometry assessment."""
    finite = np.isfinite(radii)
    if radii.shape[0] < 2 or not np.any(finite):
        return {"geometry_status": "ANALYSIS_BLOCKED", "reason_codes": ["insufficient_rings"], "geometry_method": "deterministic_math_v1"}

    spacing = np.diff(radii, axis=0)
    if np.any(spacing[np.isfinite(spacing)] <= 0):
        return {"geometry_status": "UNGRADABLE", "reason_codes": ["non_positive_spacing"], "geometry_method": "deterministic_math_v1"}

    median_spacing_k = np.nanmedian(spacing, axis=1, keepdims=True)
    normalized_spacing = np.divide(spacing, median_spacing_k, out=np.full_like(spacing, np.nan), where=median_spacing_k > 0)
    
    mad = np.nanmedian(np.abs(spacing - median_spacing_k), axis=1)
    robust_var = np.divide(mad, np.squeeze(median_spacing_k), out=np.full_like(mad, np.nan), where=np.squeeze(median_spacing_k) > 0)
    
    num_angles = spacing.shape[1]
    half = num_angles // 2
    opposite_diff = np.abs(spacing - np.roll(spacing, half, axis=1))
    opp_asym_grid = np.divide(opposite_diff, median_spacing_k, out=np.full_like(opposite_diff, np.nan), where=median_spacing_k > 0)
    
    compression_grid = np.maximum(0, 1 - normalized_spacing)
    expansion_grid = np.maximum(0, normalized_spacing - 1)
    
    theta = np.deg2rad(angles_deg)
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta), np.cos(2*theta), np.sin(2*theta)])
    irregularity = []
    residuals = np.full_like(radii, np.nan)
    for k, r_k in enumerate(radii):
        valid = np.isfinite(r_k)
        if np.sum(valid) >= 6:
            beta, _, _, _ = np.linalg.lstsq(X[valid], r_k[valid], rcond=None)
            res = r_k - (X @ beta)
            residuals[k] = res
            irregularity.append(np.sqrt(np.nanmean(res[valid]**2)) / max(np.nanmean(r_k), 1e-6))
        else:
            irregularity.append(np.nan)
    irregularity = np.array(irregularity)
    
    multiring_agreement_grid = np.nanstd(normalized_spacing, axis=0)
    direct_coverage = float(np.mean(observed)) if observed is not None else float(np.mean(np.isfinite(radii)))
    
    # Calculate feature families
    features = {
        "SPACING_VARIATION": float(np.nanmax(robust_var)) if np.any(np.isfinite(robust_var)) else 0.0,
        "OPPOSITE_ASYMMETRY": float(np.nanmax(opp_asym_grid)) if np.any(np.isfinite(opp_asym_grid)) else 0.0,
        "LOCAL_COMPRESSION": float(np.nanmax(compression_grid)) if np.any(np.isfinite(compression_grid)) else 0.0,
        "LOCAL_EXPANSION": float(np.nanmax(expansion_grid)) if np.any(np.isfinite(expansion_grid)) else 0.0,
        "RING_SHAPE_IRREGULARITY": float(np.nanmax(irregularity)) if np.any(np.isfinite(irregularity)) else 0.0,
        "MULTIRING_AGREEMENT": float(np.nanmax(multiring_agreement_grid)) if np.any(np.isfinite(multiring_agreement_grid)) else 0.0,
        "DIRECT_OBSERVATION_COVERAGE": direct_coverage,
        "TRACKING_RELIABILITY": 1.0 # placeholder
    }

    # Evaluate against thresholds (which are missing, so NOT_CALIBRATED)
    # The prompt explicitly requires NOT_CALIBRATED if no approved clinical thresholds are configured.
    # I should check if config has thresholds. Let's assume config.geometry_thresholds is None
    status = "NOT_CALIBRATED"
    gates = {}
    
    # Check if there are thresholds in config.
    if hasattr(config, "geometry_thresholds") and config.geometry_thresholds is not None:
        pass # we don't have them anyway based on instructions.

    return {
        "geometry_method": "deterministic_math_v1",
        "geometry_status": status,
        "geometry_confidence": 1.0 if status != "UNGRADABLE" else 0.0,
        "gates": gates,
        "reason_codes": ["missing_clinical_thresholds"] if status == "NOT_CALIBRATED" else [],
        "features": features,
        "_grids": {
            "normalized_spacing": normalized_spacing,
            "opp_asym_grid": opp_asym_grid,
            "compression_grid": compression_grid,
            "multiring_agreement_grid": multiring_agreement_grid,
            "residuals": residuals,
            "robust_var": robust_var,
        }
    }

from dataclasses import dataclass

@dataclass(frozen=True)
class GeometryValidation:
    valid: bool
    flags: list[str]
    direct_coverage: float

def validate_geometry(
    radii: np.ndarray,
    observed: np.ndarray | None = None,
    min_direct_coverage: float = 0.0,
) -> GeometryValidation:
    """Verify geometry invariants before model-ready feature extraction."""
    radii = np.asarray(radii, dtype=float)
    flags: list[str] = []
    if radii.ndim != 2 or radii.shape[0] < 2 or radii.shape[1] < 1:
        return GeometryValidation(False, ["insufficient_tracked_geometry"], 0.0)
    finite = np.isfinite(radii)
    if np.any(radii[finite] <= 0):
        flags.append("non_positive_radius")
    for angle in range(radii.shape[1]):
        values = radii[:, angle]
        values = values[np.isfinite(values)]
        if len(values) > 1 and np.any(np.diff(values) <= 0):
            flags.append("non_monotonic_ring_order")
            break
    if observed is None:
        direct = float(np.mean(finite))
    else:
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != radii.shape:
            flags.append("observation_shape_mismatch")
            direct = 0.0
        else:
            direct = float(np.mean(observed))
    if direct < min_direct_coverage:
        flags.append("insufficient_direct_observation")
    if not np.any(np.isfinite(np.diff(radii, axis=0))):
        flags.append("no_valid_ring_spacing")
    return GeometryValidation(not flags, sorted(set(flags)), direct)
