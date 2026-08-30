#!/usr/bin/env python3
"""
FusionUncertaintyNet — Interval-Calibration Isotonic Recalibration.

Follow-up to:
  - training/kaggle/eval_interval_calibration.py  (real 0.181 error, point r=0.881)
  - training/kaggle/fit_interval_temperature.py   (scalar T=0.21 halves error to
    0.086 but overcorrects at high nominal — see docs/interval-calibration-result-2026-08-30.md)

This script keeps fit_interval_temperature.py as-is and adds the genuine fix the
doc calls for: level-dependent recalibration via isotonic regression on PIT values,
not a single multiplicative constant. The Gamma(k, theta) head's SHAPE, not just
its scale, is miscalibrated — a single T cannot fix it.

What this does
  1) Loads the SAME real checkpoint + real held-out split as eval_interval_calibration.py
     and fit_interval_temperature.py (same md5(accession)%10==9 val set, same
     encoder-mode auto-detection, same plddt-withheld/phi-psi-forwarded
     extraction — reuses collect_predictions() directly, no duplicated logic).
  2) Splits the held-out set PROTEIN-LEVEL (not residue-level) into a fit half
     and a final test half, via md5(accession + salt) parity — deterministic,
     no shuffling. Uses a distinct salt "|isotonic-fit-split" so this split is
     independent of the temperature split, but follows the exact same convention
     (md5-salted disjoint hash, no protein leaks across the split).
  3) On the FIT split only:
       - computes PIT values u_i = F(y_i | pred_i, k_i, theta_i) using the SAME
         Normal(pred, var=k*theta^2) approximation as compute_interval_coverage
         (scipy.stats.norm.cdf, var=k*theta^2, std=sqrt(max(var,1e-6))) for
         consistency with the already-reported numbers.
       - sorts u_i and fits an IsotonicRegression (sklearn, PAV) mapping
         nominal quantile u -> empirical quantile rank (i/n). This is the
         standard "recalibration via isotonic regression on PIT values".
  4) Applies the fitted isotonic map to the HELD-OUT TEST split (never used to
     fit the map) by inverting it to get calibrated intervals at each nominal
     level (50/80/90/95) and reports the real interval_calibration_error on
     that TEST split — this is the honest, un-leaked number.
  5) Also reports pre-recalibration (raw) and post-temperature-only baselines
     on the SAME TEST split (temperature T refitted on the same FIT split for
     a fair from-scratch comparison; the already-recorded T=0.21 from the Modal
     run is also noted for audit).

Usage (same environment as eval_interval_calibration.py — Kaggle T4, real
ESM2/ProtT5 encoders, internet + disk):
  python training/kaggle/fit_interval_isotonic.py \
      --out eval_results/interval_isotonic.json

Synthetic sanity-check (no GPU/weights needed):
  The helper functions below (compute_pit, fit_isotonic_map,
  compute_calibrated_interval_coverage) are importable and tested locally
  with synthetic fixtures — see tests/test_interval_isotonic_calibration.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_interval_calibration import (  # noqa: E402
    HF_CHECKPOINT_REPO,
    _detect_encoder_mode,
    _ensure_repo_on_path,
    _pick_device,
    _try_load_kaggle_token,
    collect_predictions,
    compute_interval_coverage,
    download_checkpoint,
    resolve_manifest,
)


# ---------------------------------------------------------------------------
# PIT + isotonic helpers (importable for synthetic unit tests)
# ---------------------------------------------------------------------------

def compute_pit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute PIT values u_i = Phi((y_true - y_pred)/std) where Phi is
    standard Normal CDF and std = sqrt(max(k*theta^2, 1e-6)). This reuses
    the EXACT variance logic from eval_interval_calibration.compute_interval_coverage
    (var = k*theta^2) for consistency — the runner approximates Gamma(k,theta)
    as Normal(pred, var=k*theta^2) throughout.

    Returns array in [~0, ~1] (clipped to [1e-9, 1-1e-9] internally only
    if caller needs to avoid ppf infinities; we return raw unclipped but
    callers clipping before ppf is recommended).
    """
    from scipy.stats import norm

    y_true_a = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred, dtype=float)
    k_a = np.asarray(k, dtype=float)
    theta_a = np.asarray(theta, dtype=float)
    var = k_a * (theta_a ** 2)
    std = np.sqrt(np.maximum(var, 1e-6))
    z = (y_true_a - y_pred_a) / std
    u = norm.cdf(z)
    # Clip only to avoid exact 0/1 that would break downstream ppf; keep tiny epsilon
    # but not so much it affects isotonic shape. Use 1e-9 (ppf still ~ +/-6).
    u = np.clip(u, 1e-9, 1 - 1e-9)
    return u


def fit_isotonic_map(u_fit: np.ndarray):
    """
    Fit isotonic regression mapping nominal PIT (x) -> empirical quantile (y).

    Standard PIT recalibration: sort PIT values, empirical quantile is
    (i+1)/n (or i/n per doc) — we use (arange(1,n+1)/n) matching the doc's
    "empirical rank (i/n)" phrasing with 1-indexed rank.

    Returns (isotonic_regressor, u_sorted, y_empirical, y_hat) where
    y_hat = isotonic_regressor.predict(u_sorted) is the fitted monotone values.

    Uses sklearn.isotonic.IsotonicRegression with y_min=0, y_max=1,
    increasing=True, out_of_bounds='clip'. If sklearn is unavailable,
    raises ImportError with guidance (should not happen — scikit-learn is
    already a repo dependency via backend-heavy/requirements.txt).
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as e:
        raise ImportError(
            "scikit-learn IsotonicRegression not available. Install scikit-learn "
            "(already in backend-heavy/requirements.txt: scikit-learn==1.4.2) or "
            "provide a PAV fallback."
        ) from e

    u_fit_a = np.asarray(u_fit, dtype=float)
    n = len(u_fit_a)
    if n == 0:
        raise ValueError("u_fit is empty — cannot fit isotonic map")
    # Sort PIT values
    order = np.argsort(u_fit_a)
    u_sorted = u_fit_a[order]
    # Empirical quantiles: (1..n)/n  (doc: "empirical rank (i/n) against sorted u_i")
    y_emp = np.arange(1, n + 1, dtype=float) / float(n)
    # Fit monotone increasing, clipped to [0,1], clip out-of-bounds
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    ir.fit(u_sorted, y_emp)
    y_hat = ir.predict(u_sorted)
    return ir, u_sorted, y_emp, y_hat


def _invert_isotonic(ir, u_sorted: np.ndarray, y_hat: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    """
    Invert isotonic map f: raw_u -> calibrated_u.
    Given desired calibrated quantile q (e.g. 0.025), find raw_u such that f(raw_u)=q.
    Implements linear interpolation of (y_hat -> u_sorted) with clipping.

    q is in [0,1]. u_sorted and y_hat are training sorted PIT and fitted values
    (both monotone increasing). Uses np.interp(q, y_hat, u_sorted).

    Handles flat plateaus in y_hat by deduplicating in a stable way: we keep
    y_hat strictly increasing for interpolation by taking the leftmost u for
    each distinct y level. But np.interp already handles non-strictly-increasing
    xp by returning the leftmost — we just ensure y_hat and u_sorted are sorted
    by y_hat (they already are increasing, so fine).

    Also clips q to [y_hat.min(), y_hat.max()] which with out_of_bounds='clip'
    matches IsotonicRegression's clipping behaviour.
    """
    y_hat_a = np.asarray(y_hat, dtype=float)
    u_sorted_a = np.asarray(u_sorted, dtype=float)
    q_a = np.asarray(q, dtype=float)
    # Ensure y_hat is sorted increasing — it is by construction, but guard
    # against tiny floating noise: sort by y_hat
    # Actually y_hat is already monotone and u_sorted monotone, so joint sort by y_hat keeps correspondence
    # If y_hat has duplicates, interp will use the last occurrence's u — we prefer
    # averaging across plateaus: take the mean u per y_hat level. Simpler: use
    # unique via grouping.
    # Build unique y_hat -> mean u mapping to avoid ambiguous interpolation at plateaus.
    # This is a small refinement; raw np.interp(q, y_hat, u_sorted) also works but
    # can be biased at exact plateau boundaries. We deduplicate by averaging.
    # For speed, only deduplicate if duplicates exist.
    if len(y_hat_a) != len(np.unique(y_hat_a)):
        # Group by y_hat rounded to avoid floating duplicates? Use exact.
        # Use dict: map y->list of u, then mean.
        # More efficient: use np.unique with return_inverse
        uniq_y, inv, counts = np.unique(y_hat_a, return_inverse=True, return_counts=True)
        # Compute mean u per uniq_y
        sum_u = np.zeros_like(uniq_y, dtype=float)
        np.add.at(sum_u, inv, u_sorted_a)
        mean_u = sum_u / counts
        y_for_interp = uniq_y
        u_for_interp = mean_u
    else:
        y_for_interp = y_hat_a
        u_for_interp = u_sorted_a

    # Clip q to interpolation domain
    lo = float(y_for_interp.min())
    hi = float(y_for_interp.max())
    # np.interp left/right handle out-of-bounds
    result = np.interp(q_a, y_for_interp, u_for_interp, left=float(u_for_interp[0]), right=float(u_for_interp[-1]))
    # Also clip raw_u to valid PIT range [1e-9, 1-1e-9] for subsequent ppf
    result = np.clip(result, 1e-9, 1 - 1e-9)
    return result


def compute_calibrated_interval_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: np.ndarray,
    theta: np.ndarray,
    ir,
    u_sorted: np.ndarray,
    y_hat: np.ndarray,
) -> tuple[dict, float]:
    """
    Compute interval coverage after isotonic recalibration, on HELD-OUT data.

    For each nominal level in (0.50, 0.80, 0.90, 0.95):
      lo_q, hi_q = 0.5 - level/2, 0.5 + level/2  (central interval quantiles)
      raw_lo, raw_hi = f^{-1}(lo_q), f^{-1}(hi_q) via inverse isotonic map
      lo, hi = pred + ppf(raw_lo)*std, pred + ppf(raw_hi)*std
      empirical = mean(true in [lo, hi])

    Returns (coverage_dict, interval_calibration_error) in SAME format as
    eval_interval_calibration.compute_interval_coverage.

    ir, u_sorted, y_hat are from fit_isotonic_map on the FIT split only —
    caller must ensure they are not refitted on the test data.
    """
    from scipy.stats import norm

    y_true_a = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred, dtype=float)
    k_a = np.asarray(k, dtype=float)
    theta_a = np.asarray(theta, dtype=float)
    var = k_a * (theta_a ** 2)
    std = np.sqrt(np.maximum(var, 1e-6))

    coverage: dict = {}
    for level in (0.50, 0.80, 0.90, 0.95):
        lo_q = 0.5 - level / 2
        hi_q = 0.5 + level / 2
        raw_lo = float(_invert_isotonic(ir, u_sorted, y_hat, lo_q))
        raw_hi = float(_invert_isotonic(ir, u_sorted, y_hat, hi_q))
        # Convert raw PIT quantiles back to y via Normal ppf
        z_lo = norm.ppf(raw_lo)
        z_hi = norm.ppf(raw_hi)
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


def fit_temperature(
    y_true: np.ndarray, y_pred: np.ndarray, k: np.ndarray, theta: np.ndarray
) -> tuple[float, float]:
    """
    Fit a scalar T minimizing interval_calibration_error when std is scaled
    by T (i.e. var_scaled = T^2 * k * theta^2). Coarse grid over [0.05, 2.0]
    then a local refinement pass around the best coarse point.

    Identical to training/kaggle/fit_interval_temperature.py:fit_temperature
    — duplicated here so this script is self-contained for import in tests,
    and to allow a fair from-scratch temperature baseline on the SAME fit
    split as the isotonic map (rather than reusing a stale T=0.21 fitted on a
    different protein split).

    Returns (best_T, interval_calibration_error_at_best_T) on the SAME data
    passed in — caller is responsible for only passing the fit split here.
    """
    def error_at(t: float) -> float:
        scaled_theta = np.asarray(theta) * t
        _, err = compute_interval_coverage(y_pred, y_true, k, scaled_theta)
        return err

    coarse = np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(1.0, 2.0, 11)[1:]])
    coarse_errs = [error_at(t) for t in coarse]
    best_idx = int(np.argmin(coarse_errs))
    best_t = float(coarse[best_idx])

    lo = max(0.01, best_t - 0.1)
    hi = best_t + 0.1
    fine = np.linspace(lo, hi, 41)
    fine_errs = [error_at(t) for t in fine]
    fbest_idx = int(np.argmin(fine_errs))
    best_t = float(fine[fbest_idx])
    best_err = float(fine_errs[fbest_idx])

    return best_t, best_err


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit interval-calibration isotonic recalibration for FusionUncertaintyNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="eval_results/interval_isotonic.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-samples", type=int, default=841)
    parser.add_argument("--encoder-mode", default=None, choices=["small", "full"])
    parser.add_argument("--hf-repo", default=HF_CHECKPOINT_REPO)
    args = parser.parse_args()

    repo_root = _ensure_repo_on_path()
    _try_load_kaggle_token()

    import torch

    device = args.device if args.device != "auto" else _pick_device()

    ckpt = args.checkpoint
    if ckpt is not None and os.path.isfile(os.path.join(ckpt, "pytorch_model.bin")):
        print(f"[fit-iso] Using explicit local checkpoint {ckpt}")
    else:
        ckpt = download_checkpoint(
            repo_id=args.hf_repo,
            local_dir="/kaggle/working/checkpoints/best-v2-leakfree" if Path("/kaggle/working").exists() else "./checkpoints/best-v2-leakfree",
        )

    manifest = resolve_manifest(args.manifest, repo_root)
    encoder_mode = args.encoder_mode or _detect_encoder_mode(ckpt)
    print(f"[fit-iso] device={device} encoder_mode={encoder_mode} checkpoint={ckpt}")

    from dataset import ProteinQualityDataset

    ds = ProteinQualityDataset(manifest, synthetic_fallback=True)
    n = len(ds)
    val_items = []
    for i, it in enumerate(ds.items):
        acc = it.get("accession", f"idx{i:06d}")
        if int(hashlib.md5(acc.encode()).hexdigest(), 16) % 10 == 9:
            val_items.append(it)
    if len(val_items) < 50:
        test_n = min(args.max_samples or 200, n // 10)
        val_items = ds.items[n - test_n:]
    if args.max_samples and len(val_items) > args.max_samples:
        val_items = val_items[: args.max_samples]
    print(f"[fit-iso] held-out val set: {len(val_items)}/{n} proteins")

    # Protein-level fit/test split, deterministic via salted md5 (distinct salt
    # from the temperature script so splits are independent, but same convention
    # — no protein leaks across fit/test).
    fit_items, test_items = [], []
    for it in val_items:
        acc = it.get("accession", it.get("sequence", "")[:32])
        h = int(hashlib.md5((acc + "|isotonic-fit-split").encode()).hexdigest(), 16)
        (fit_items if h % 2 == 0 else test_items).append(it)
    print(f"[fit-iso] protein-level split: fit={len(fit_items)} test={len(test_items)}")

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../backend-heavy"))
    from fusionuncertaintynet.model import FusionUncertaintyNet

    ckpt_dir = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
    model = FusionUncertaintyNet.from_pretrained(ckpt_dir, device=device)
    model.eval()

    print("[fit-iso] running inference on FIT split...")
    fy_true, fy_pred, fk, ftheta, _ = collect_predictions(model, fit_items, device=device, encoder_mode=encoder_mode)
    print(f"[fit-iso] FIT split: {len(fy_true)} residues from {len(fit_items)} proteins")

    print("[fit-iso] running inference on TEST split...")
    ty_true, ty_pred, tk, ttheta, _ = collect_predictions(model, test_items, device=device, encoder_mode=encoder_mode)
    print(f"[fit-iso] TEST split: {len(ty_true)} residues from {len(test_items)} proteins")

    fy_true_a, fy_pred_a, fk_a, ftheta_a = map(np.array, (fy_true, fy_pred, fk, ftheta))
    ty_true_a, ty_pred_a, tk_a, ttheta_a = map(np.array, (ty_true, ty_pred, tk, ttheta))

    # --- pre-recalibration (raw) ---
    fit_coverage_raw, fit_err_raw = compute_interval_coverage(fy_pred_a, fy_true_a, fk_a, ftheta_a)
    test_coverage_raw, test_err_raw = compute_interval_coverage(ty_pred_a, ty_true_a, tk_a, ttheta_a)
    print(f"[fit-iso] RAW FIT  interval_calibration_error={fit_err_raw:.4f} coverage={ {k: round(v['empirical_coverage'],3) for k,v in fit_coverage_raw.items()}}")
    print(f"[fit-iso] RAW TEST interval_calibration_error={test_err_raw:.4f} coverage={ {k: round(v['empirical_coverage'],3) for k,v in test_coverage_raw.items()}}")

    # --- temperature-only baseline (refit on SAME fit split for fair comparison) ---
    best_t, fit_err_temp = fit_temperature(fy_true_a, fy_pred_a, fk_a, ftheta_a)
    fit_coverage_temp, _ = compute_interval_coverage(fy_pred_a, fy_true_a, fk_a, ftheta_a * best_t)
    test_coverage_temp, test_err_temp = compute_interval_coverage(ty_pred_a, ty_true_a, tk_a, ttheta_a * best_t)
    print(f"[fit-iso] TEMP  best T={best_t:.4f} FIT err {fit_err_raw:.4f}->{fit_err_temp:.4f}  TEST err {test_err_raw:.4f}->{test_err_temp:.4f}")
    # also note canned T=0.21 from the Modal run for audit
    _, canned_temp_err_on_fit = compute_interval_coverage(fy_pred_a, fy_true_a, fk_a, ftheta_a * 0.21)
    test_coverage_canned, test_err_canned = compute_interval_coverage(ty_pred_a, ty_true_a, tk_a, ttheta_a * 0.21)
    print(f"[fit-iso] CANNED T=0.21 (Modal) FIT err-> {canned_temp_err_on_fit:.4f}  TEST err->{test_err_canned:.4f} (reference, different split)")

    # --- isotonic recalibration (fit on FIT split PIT) ---
    from scipy.stats import norm as _norm  # noqa: F401

    u_fit = compute_pit(fy_true_a, fy_pred_a, fk_a, ftheta_a)
    # Sanity: PIT mean should be ~0.5 if calibrated; here likely ~0.5 but with over-coverage the PIT distribution is too peaked.
    print(f"[fit-iso] FIT PIT stats: mean={u_fit.mean():.3f} std={u_fit.std():.3f} (uniform would be 0.5/0.289)  5th={np.percentile(u_fit,5):.3f} 95th={np.percentile(u_fit,95):.3f}")

    ir, u_sorted, y_emp, y_hat = fit_isotonic_map(u_fit)
    # Fit coverage on fit split via isotonic (should be well-calibrated in-sample)
    fit_coverage_iso, fit_err_iso = compute_calibrated_interval_coverage(fy_true_a, fy_pred_a, fk_a, ftheta_a, ir, u_sorted, y_hat)
    # True test: apply to held-out TEST split
    test_coverage_iso, test_err_iso = compute_calibrated_interval_coverage(ty_true_a, ty_pred_a, tk_a, ttheta_a, ir, u_sorted, y_hat)
    print(f"[fit-iso] ISOTONIC FIT  err {fit_err_raw:.4f}->{fit_err_iso:.4f}")
    print(f"[fit-iso] ISOTONIC TEST err {test_err_raw:.4f}->{test_err_iso:.4f}  (honest, TEST never used for fitting)")
    print(f"[fit-iso] TEST coverage detail raw : { {k: round(v['empirical_coverage'],3) for k,v in test_coverage_raw.items()}}")
    print(f"[fit-iso] TEST coverage detail temp : { {k: round(v['empirical_coverage'],3) for k,v in test_coverage_temp.items()}} (T={best_t:.3f})")
    print(f"[fit-iso] TEST coverage detail iso  : { {k: round(v['empirical_coverage'],3) for k,v in test_coverage_iso.items()}}")

    result = {
        "checkpoint": args.hf_repo,
        "manifest": manifest,
        "device": device,
        "encoder_mode": encoder_mode,
        "n_fit_proteins": len(fit_items),
        "n_fit_residues": len(fy_true),
        "n_test_proteins": len(test_items),
        "n_test_residues": len(ty_true),
        "fit_split": {
            "interval_calibration_error_raw": fit_err_raw,
            "interval_calibration_error_temperature_refit": fit_err_temp,
            "interval_calibration_error_isotonic": fit_err_iso,
            "coverage_raw": fit_coverage_raw,
            "coverage_temperature_refit": fit_coverage_temp,
            "coverage_isotonic": fit_coverage_iso,
            "fitted_temperature_refit": best_t,
            "pit_mean": float(u_fit.mean()),
            "pit_std": float(u_fit.std()),
        },
        "test_split_never_used_for_fitting": {
            "interval_calibration_error_raw": test_err_raw,
            "interval_calibration_error_temperature_refit": test_err_temp,
            "interval_calibration_error_temperature_canned_0_21": test_err_canned,
            "interval_calibration_error_isotonic": test_err_iso,
            "coverage_raw": test_coverage_raw,
            "coverage_temperature_refit": test_coverage_temp,
            "coverage_temperature_canned_0_21": test_coverage_canned,
            "coverage_isotonic": test_coverage_iso,
        },
        # Chain summary for audit: raw -> temp -> iso on the same honest TEST split
        "improvement_chain_test": {
            "raw": test_err_raw,
            "temperature_refit_T": best_t,
            "temperature_refit": test_err_temp,
            "temperature_canned_0_21": test_err_canned,
            "isotonic": test_err_iso,
            "temperature_note": "T refitted on SAME fit split as isotonic for fair comparison; canned 0.21 is from a different protein split (Modal T=0.21 result) included only as reference",
            "isotonic_note": "Isotonic map fitted on FIT PIT only, inverted to calibrated intervals on TEST; honest held-out estimate.",
        },
        "method": {
            "raw": "Normal(pred, var=k*theta^2) central intervals at 50/80/90/95, mean|empirical-nominal| — identical to eval_interval_calibration.compute_interval_coverage / training/scripts/evaluate.py:141-152",
            "temperature": "scalar T on theta (std scaled by T), grid+refine on FIT split minimizing interval_calibration_error, applied to TEST — same as fit_interval_temperature.py",
            "isotonic": "PIT u_i = Phi((y_true-pred)/std) on FIT split, sort u_i, empirical rank (i/n), IsotonicRegression(y_min=0,y_max=1, increasing, clip) mapping nominal->empirical, invert via interp(mean_u per y_hat plateau) to get f^{-1}(lo_q),f^{-1}(hi_q), then y intervals via ppf; TEST coverage measured there",
        },
        "gate": {
            "bar": 0.07,
            "test_raw_passes": bool(test_err_raw < 0.07),
            "test_temperature_refit_passes": bool(test_err_temp < 0.07),
            "test_isotonic_passes": bool(test_err_iso < 0.07),
        },
        "caveats": (
            "Isotonic recalibration is more flexible than a single scalar T and can overfit the calibration map "
            "itself when the FIT split is small. With ~407 proteins / tens of thousands of residues (fit split), "
            "the map has ample data for a 1D monotone fit, but if FIT residues are <~5k or proteins are highly "
            "correlated, the map may chase noise. Clip+monotone constraints mitigate but do not eliminate this. "
            "Report TEST error (never used for fitting) as the honest number; FIT error is optimistic. "
            "Also, the Normal(pred,var) predictive approximation is itself a model choice — true predictive may be "
            "non-Gaussian; isotonic PIT recalibration corrects marginal CDF shape but not conditional structure."
        ),
    }

    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fit-iso] Saved -> {args.out}")


if __name__ == "__main__":
    main()
