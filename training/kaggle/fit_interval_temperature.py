#!/usr/bin/env python3
"""
FusionUncertaintyNet — Interval-Calibration Temperature Scaling.

Follow-up to training/kaggle/eval_interval_calibration.py's real, post-bugfix
result (docs/interval-calibration-result-2026-08-30.md): point predictions are
strong (Pearson 0.881) but the (k, theta) uncertainty head is systematically
over-conservative at every nominal level (interval_calibration_error=0.181 vs
a <0.07 target) — the classic signature calling for temperature/variance
scaling, not a retrain.

What this does
  1) Loads the same real checkpoint + real held-out split as
     eval_interval_calibration.py (same md5(accession)%10==9 val set, same
     encoder-mode auto-detection, same plddt-withheld/phi-psi-forwarded
     extraction — reuses collect_predictions() directly, no duplicated logic).
  2) Splits the held-out set PROTEIN-LEVEL (not residue-level, to avoid
     within-protein leakage) into a fit half and a final test half, via
     md5(accession) parity — deterministic, no shuffling needed.
  3) Fits a single scalar temperature T (multiplier on the predicted std) on
     the fit half only, via 1D grid + local refinement, minimizing
     interval_calibration_error.
  4) Applies the fitted T to the held-out TEST half (never seen during fitting)
     and reports both pre- and post-scaling interval_calibration_error there —
     this is the honest, un-leaked number.
  5) Saves calibration.json with T and both halves' metrics.

Usage (same environment as eval_interval_calibration.py — Kaggle T4, real
ESM2/ProtT5 encoders, internet + disk):
  python training/kaggle/fit_interval_temperature.py \
      --out eval_results/interval_temperature.json
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


def fit_temperature(
    y_true: np.ndarray, y_pred: np.ndarray, k: np.ndarray, theta: np.ndarray
) -> tuple[float, float]:
    """
    Fit a scalar T minimizing interval_calibration_error when std is scaled
    by T (i.e. var_scaled = T^2 * k * theta^2). Coarse grid over [0.05, 2.0]
    then a local refinement pass around the best coarse point.

    Returns (best_T, interval_calibration_error_at_best_T) on the SAME data
    passed in — caller is responsible for only passing the fit split here,
    never the held-out test split.
    """
    def error_at(t: float) -> float:
        scaled_theta = theta * t
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
        description="Fit interval-calibration temperature scaling for FusionUncertaintyNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="eval_results/interval_temperature.json")
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
        print(f"[fit-temp] Using explicit local checkpoint {ckpt}")
    else:
        ckpt = download_checkpoint(
            repo_id=args.hf_repo,
            local_dir="/kaggle/working/checkpoints/best-v2-leakfree" if Path("/kaggle/working").exists() else "./checkpoints/best-v2-leakfree",
        )

    manifest = resolve_manifest(args.manifest, repo_root)
    encoder_mode = args.encoder_mode or _detect_encoder_mode(ckpt)
    print(f"[fit-temp] device={device} encoder_mode={encoder_mode} checkpoint={ckpt}")

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
    print(f"[fit-temp] held-out val set: {len(val_items)}/{n} proteins")

    # Protein-level fit/test split, deterministic via a second, independent
    # md5 hash pass (salted) so it doesn't correlate with the val-set selection
    # hash above — that would bias which proteins land in fit vs test.
    fit_items, test_items = [], []
    for it in val_items:
        acc = it.get("accession", it.get("sequence", "")[:32])
        h = int(hashlib.md5((acc + "|temp-fit-split").encode()).hexdigest(), 16)
        (fit_items if h % 2 == 0 else test_items).append(it)
    print(f"[fit-temp] protein-level split: fit={len(fit_items)} test={len(test_items)}")

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../backend-heavy"))
    from fusionuncertaintynet.model import FusionUncertaintyNet

    ckpt_dir = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
    model = FusionUncertaintyNet.from_pretrained(ckpt_dir, device=device)
    model.eval()

    print("[fit-temp] running inference on FIT split...")
    fy_true, fy_pred, fk, ftheta, _ = collect_predictions(model, fit_items, device=device, encoder_mode=encoder_mode)
    print(f"[fit-temp] FIT split: {len(fy_true)} residues from {len(fit_items)} proteins")

    print("[fit-temp] running inference on TEST split...")
    ty_true, ty_pred, tk, ttheta, _ = collect_predictions(model, test_items, device=device, encoder_mode=encoder_mode)
    print(f"[fit-temp] TEST split: {len(ty_true)} residues from {len(test_items)} proteins")

    fy_true_a, fy_pred_a, fk_a, ftheta_a = map(np.array, (fy_true, fy_pred, fk, ftheta))
    ty_true_a, ty_pred_a, tk_a, ttheta_a = map(np.array, (ty_true, ty_pred, tk, ttheta))

    fit_coverage_pre, fit_err_pre = compute_interval_coverage(fy_pred_a, fy_true_a, fk_a, ftheta_a)
    best_t, fit_err_post = fit_temperature(fy_true_a, fy_pred_a, fk_a, ftheta_a)
    print(f"[fit-temp] best T={best_t:.4f} FIT interval_calibration_error {fit_err_pre:.4f} -> {fit_err_post:.4f}")

    test_coverage_pre, test_err_pre = compute_interval_coverage(ty_pred_a, ty_true_a, tk_a, ttheta_a)
    test_coverage_post, test_err_post = compute_interval_coverage(ty_pred_a, ty_true_a, tk_a, ttheta_a * best_t)
    print(f"[fit-temp] HELD-OUT TEST interval_calibration_error {test_err_pre:.4f} -> {test_err_post:.4f} (never used for fitting T)")

    result = {
        "checkpoint": args.hf_repo,
        "manifest": manifest,
        "device": device,
        "encoder_mode": encoder_mode,
        "n_fit_proteins": len(fit_items),
        "n_fit_residues": len(fy_true),
        "n_test_proteins": len(test_items),
        "n_test_residues": len(ty_true),
        "fitted_temperature": best_t,
        "fit_split": {
            "interval_calibration_error_pre": fit_err_pre,
            "interval_calibration_error_post": fit_err_post,
            "coverage_pre": fit_coverage_pre,
        },
        "test_split_never_used_for_fitting": {
            "interval_calibration_error_pre": test_err_pre,
            "interval_calibration_error_post": test_err_post,
            "coverage_pre": test_coverage_pre,
            "coverage_post": test_coverage_post,
        },
        "method": "scalar multiplier T on predicted theta (std = sqrt(k)*theta*T), fit by grid+local-refine search on FIT split minimizing mean|empirical_coverage-nominal| across (50,80,90,95)%, applied to a disjoint held-out TEST split for an unbiased post-scaling estimate.",
        "gate": {
            "bar": 0.07,
            "test_pre_passes": bool(test_err_pre < 0.07),
            "test_post_passes": bool(test_err_post < 0.07),
        },
    }

    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fit-temp] Saved -> {args.out}")


if __name__ == "__main__":
    main()
