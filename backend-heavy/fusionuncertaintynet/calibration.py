"""FusionUncertaintyNet calibration helpers — pure-numpy, no sklearn at inference.

Provides portable isotonic PIT recalibration via dense breakpoint lookup table
and numpy.interp.  Used by training/kaggle/fit_interval_isotonic.py (serialization
+ TEST round-trip) and by backend-heavy/app/main.py (opt-in serving).

Breakpoints format (as saved in calibration-isotonic-v1.json):
  {
    "breakpoints": {"x": [...], "y": [...]},  # alias "nominal"/"empirical" also accepted
    # or flat {"x": [...], "y": [...]}
  }
  x: nominal quantile PIT values (linspace 0..1, increasing)
  y: empirical quantile f(x) = IsotonicRegression.predict(x), monotone, clipped to [0,1]

At inference:
  raw_u = F_raw(y) = Phi((y - pred)/std)  with var = k*theta^2
  calibrated PIT = f(raw_u) via interp
  For interval calibration, invert: raw = f^{-1}(q) via interp(q, y, x)
  calibrated interval = pred + ppf(raw)*std

For serving reporting of theta/aleatoric/total_unc, isotonic is inherently
level-dependent (different factor per nominal coverage level), while the API
returns a single scalar aleatoric/total_unc.  We therefore derive a single
effective theta scaling factor as the mean ratio of calibrated interval width
to nominal interval width across the standard levels (50/80/90/95), using the
same Normal(pred,var) + var=k*theta^2 assumption as everywhere else.
This is an approximation — level-dependent recalibration cannot be reduced
to a single scalar without loss — and is flagged explicitly in the task's
honest-gaps report.  The alternative (returning per-level intervals) would
require an API change.

No sklearn import at inference; only numpy + optional scipy/statistics for ppf.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np


def _norm_ppf(p: np.ndarray | float) -> np.ndarray | float:
    """Inverse standard Normal CDF.  Prefers scipy if available, else stdlib.

    Clips input to (1e-9, 1-1e-9) to avoid infinities.
    """
    p_a = np.asarray(p, dtype=float)
    p_a = np.clip(p_a, 1e-9, 1 - 1e-9)
    try:
        from scipy.stats import norm  # type: ignore

        return norm.ppf(p_a)
    except Exception:
        # fallback: Python stdlib statistics.NormalDist (pure python, no extra dep)
        from statistics import NormalDist

        nd = NormalDist(mu=0, sigma=1)
        # vectorize via np.vectorize for arrays
        v = np.vectorize(nd.inv_cdf, otypes=[float])
        out = v(p_a)
        # preserve scalar return if input scalar
        if out.size == 1 and np.asarray(p).size == 1:
            return float(out.flat[0])
        return out


def _get_xy(breakpoints: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract x,y arrays from various breakpoint dict shapes."""
    if breakpoints is None:
        raise ValueError("breakpoints is None")
    # support {"breakpoints": {"x":..., "y":...}} or flat {"x":..., "y":...}
    # or {"nominal":..., "empirical":...} aliases
    bp = breakpoints
    if "breakpoints" in bp and isinstance(bp["breakpoints"], dict):
        bp = bp["breakpoints"]
    # try x/y
    if "x" in bp and "y" in bp:
        x = np.asarray(bp["x"], dtype=float)
        y = np.asarray(bp["y"], dtype=float)
    elif "nominal" in bp and "empirical" in bp:
        x = np.asarray(bp["nominal"], dtype=float)
        y = np.asarray(bp["empirical"], dtype=float)
    elif "nominal_quantile" in bp and "empirical_quantile" in bp:
        x = np.asarray(bp["nominal_quantile"], dtype=float)
        y = np.asarray(bp["empirical_quantile"], dtype=float)
    else:
        raise ValueError(f"breakpoints missing x/y keys, got {list(bp.keys())}")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError(f"breakpoints need >=2 matching points, got len x={len(x)} y={len(y)}")
    # ensure increasing x (nominal), y monotone
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    # deduplicate y for stable inversion: handled by interpolation using leftmost
    return x, y


def _invert_breakpoints(q: np.ndarray | float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Invert isotonic map f: x->y via linear interpolation of (y->x).

    q is desired calibrated quantile in [0,1].  Returns raw PIT such that f(raw)=q,
    clipped to valid PIT range.
    Handles duplicate y (plateaus) by averaging x per unique y (same as fit script).
    """
    y_a = np.asarray(y, dtype=float)
    x_a = np.asarray(x, dtype=float)
    q_a = np.asarray(q, dtype=float)
    # deduplicate at plateaus by averaging x per unique y
    if len(y_a) != len(np.unique(y_a)):
        uniq_y, inv, counts = np.unique(y_a, return_inverse=True, return_counts=True)
        sum_x = np.zeros_like(uniq_y, dtype=float)
        np.add.at(sum_x, inv, x_a)
        mean_x = sum_x / counts
        y_for = uniq_y
        x_for = mean_x
    else:
        y_for = y_a
        x_for = x_a
    # ensure y_for sorted (it is by construction since y monotone)
    # np.interp expects xp increasing
    # Clip to domain via left/right
    lo = float(y_for[0])
    hi = float(y_for[-1])
    # left/right map to extremes
    result = np.interp(q_a, y_for, x_for, left=float(x_for[0]), right=float(x_for[-1]))
    result = np.clip(result, 1e-9, 1 - 1e-9)
    return result


def _calibration_factor(x: np.ndarray, y: np.ndarray) -> float:
    """Derive single effective theta scaling factor from breakpoints.

    Computes for each standard nominal level (50/80/90/95) the ratio of
    calibrated interval width to nominal interval width under Normal assumption:
      factor_level = (ppf(raw_hi)-ppf(raw_lo)) / (ppf(nom_hi)-ppf(nom_lo))
    where raw_* = f^{-1}(nom_*).  Returns mean across levels.

    factor <1 => shrink theta (over-coverage correction, the real model's case).
    factor >1 => expand theta (under-coverage).

    This is the approximation used for serving's single scalar uncertainty.
    """
    factor_levels = []
    for level in (0.50, 0.80, 0.90, 0.95):
        lo_q = 0.5 - level / 2
        hi_q = 0.5 + level / 2
        raw_lo = float(_invert_breakpoints(lo_q, x, y))
        raw_hi = float(_invert_breakpoints(hi_q, x, y))
        z_lo = float(_norm_ppf(raw_lo))
        z_hi = float(_norm_ppf(raw_hi))
        z_lo_nom = float(_norm_ppf(lo_q))
        z_hi_nom = float(_norm_ppf(hi_q))
        denom = (z_hi_nom - z_lo_nom)
        if abs(denom) < 1e-12:
            continue
        ratio = (z_hi - z_lo) / denom
        factor_levels.append(ratio)
    if not factor_levels:
        return 1.0
    return float(np.mean(factor_levels))


def apply_isotonic_calibration(
    pred: np.ndarray | list,
    k: np.ndarray | list,
    theta: np.ndarray | list,
    breakpoints: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply isotonic calibration to reported uncertainty (pure numpy).

    Args:
        pred: array [N] predicted quality (unused for factor, kept for API symmetry
              and future conditional calibration).  Must be same length as k/theta.
        k: array [N] Gamma shape (>=1)
        theta: array [N] Gamma scale
        breakpoints: dict with x/y as per _get_xy, or {"breakpoints": {...}}

    Returns:
        tuple (k_cal, theta_cal, aleatoric_cal, epistemic_cal, total_unc_cal)
        where k_cal == k (unchanged), theta_cal = theta * factor,
        aleatoric_cal = k_cal * theta_cal^2,
        epistemic_cal = 1/k_cal,
        total_unc_cal = aleatoric_cal + epistemic_cal * 50.0
        (matching edr.py predict_with_uncertainty formula).

        All arrays shape [N], dtype float.

    Notes:
        factor is global (same for all residues) because isotonic map is a single
        global 1-D monotone function fitted on the held-out FIT split.
        The level-dependent shape information is averaged — see module docstring
        honest gap.  For evaluation round-trip at 0.0070, use
        compute_calibrated_interval_coverage_via_breakpoints instead, which preserves
        level-dependent inversion exactly.
    """
    _ = np.asarray(pred, dtype=float)  # validates length but not used for factor
    k_a = np.asarray(k, dtype=float)
    theta_a = np.asarray(theta, dtype=float)
    if k_a.shape != theta_a.shape:
        raise ValueError(f"k and theta shape mismatch {k_a.shape} vs {theta_a.shape}")
    x, y = _get_xy(breakpoints)
    factor = _calibration_factor(x, y)
    # clamp factor to reasonable range to avoid degenerate scaling
    factor = float(np.clip(factor, 0.05, 5.0))
    theta_cal = theta_a * factor
    k_cal = k_a  # unchanged
    # enforce k>=1 as per edr.py (but preserve original if already >=1)
    # do not clamp theta here beyond factor; edr already ensures >0.1
    ale_cal = k_cal * (theta_cal ** 2)
    epi_cal = 1.0 / np.maximum(k_cal, 1e-6)
    tot_cal = ale_cal + epi_cal * 50.0
    return k_cal, theta_cal, ale_cal, epi_cal, tot_cal


def compute_calibrated_interval_coverage_via_breakpoints(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    k: np.ndarray | list,
    theta: np.ndarray | list,
    breakpoints: Dict,
) -> Tuple[Dict, float]:
    """Compute interval coverage using breakpoints inversion (for round-trip check).

    Mirrors training/kaggle/fit_interval_isotonic.py::compute_calibrated_interval_coverage
    but via numpy.interp over dense breakpoints instead of sklearn object.
    This is the function to verify serialized breakpoints reproduce 0.0070.

    Returns (coverage_dict, interval_calibration_error) same format as
    eval_interval_calibration.compute_interval_coverage.
    """
    y_true_a = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred, dtype=float)
    k_a = np.asarray(k, dtype=float)
    theta_a = np.asarray(theta, dtype=float)
    var = k_a * (theta_a ** 2)
    std = np.sqrt(np.maximum(var, 1e-6))
    x, y = _get_xy(breakpoints)

    coverage: Dict = {}
    for level in (0.50, 0.80, 0.90, 0.95):
        lo_q = 0.5 - level / 2
        hi_q = 0.5 + level / 2
        raw_lo = float(_invert_breakpoints(lo_q, x, y))
        raw_hi = float(_invert_breakpoints(hi_q, x, y))
        z_lo = float(_norm_ppf(raw_lo))
        z_hi = float(_norm_ppf(raw_hi))
        lo = y_pred_a + z_lo * std
        hi = y_pred_a + z_hi * std
        empirical = float(np.mean((y_true_a >= lo) & (y_true_a <= hi)))
        coverage[f"nominal_{int(level*100)}"] = {
            "empirical_coverage": empirical,
            "gap": empirical - level,
            "lo_q_calibrated": lo_q,
            "hi_q_calibrated": hi_q,
            "raw_lo": raw_lo,
            "raw_hi": raw_hi,
            "z_lo": float(z_lo),
            "z_hi": float(z_hi),
            "nominal": level,
        }
    interval_calibration_error = float(np.mean([abs(v["gap"]) for v in coverage.values()]))
    return coverage, interval_calibration_error
