# Interval Calibration Result — 2026-08-30

**Closes the gap flagged in `docs/promotion-decision-v2-2026-08-28.md` §4**: "Uncertainty `interval_calibration_error` — not measured for this checkpoint — NOT PASS." It has now been measured, twice.

## Attempt 1 (2026-08-28, Kaggle kernel `fusionuncertaintynet-interval-calibration-eval`): invalid, do not use

The checkpoint genuinely loaded and real ESM2/ProtT5 extraction ran (841 real proteins, 271,311 real residues, no synthetic-manifest fallback), but the standalone `training/kaggle/eval_interval_calibration.py` always ran `encoder_mode=full` (ESM2 t33 650M + ProtT5-XL) regardless of what the checkpoint was actually trained with. `retrain_checkpoints_v2/best/metrics.json` records `encoder_mode=small` (ESM2 t12 35M padded to 1280d + ESM2 t30 150M padded to 1024d, per `train_real.py:146-147`). Feeding full-encoder embeddings into a fusion head trained on zero-padded small-encoder inputs produced garbage: **Pearson 0.0072** (vs. the real training-time point-prediction result of 0.8622) and `aleatoric_mean=707.7` (the model treating everything as maximally uncertain). The resulting `interval_calibration_error=0.0996` from that run is an artifact of this bug, not a real measurement, and must not be used for any promotion decision.

A second bug in the same script (`af_kwargs={"plddt": None}` without forwarding `phi`/`psi`) also silently zeroed out real geometric context that `train_real.py:134` does retain, compounding the mismatch.

Both bugs were found and fixed on 2026-08-30 (commit `157318f`, verified by 9 new regression tests against synthetic fixtures — see `tests/test_eval_interval_calibration_fix.py`).

## Attempt 2 (2026-08-30, Modal T4, post-fix): real result

Same checkpoint (`bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree`), same real held-out split (841 proteins / 271,311 residues, `md5(accession)%10==9`), `encoder_mode=small` correctly auto-detected from `metrics.json`, `phi`/`psi` correctly forwarded.

```json
{
  "pearson": 0.8812946389733627,
  "aleatoric_mean": 665.6142145728583,
  "epistemic_mean": 0.08747636079556517,
  "interval_calibration_error": 0.18085550346281573,
  "interval_calibration_at": {
    "50": 0.9178544179926358,
    "80": 0.9762486592876809,
    "90": 0.9872839656335349,
    "95": 0.9920349709374113
  }
}
```

**Point predictions**: Pearson 0.881 — confirms (slightly exceeds) the training-time `0.8622` claim. The bug fix, not the model, was the problem.

**Uncertainty calibration**: still does not clear the `<0.05` (tight) / `<0.07` (suggested minimum) bar from the promotion-decision doc's §8.2. `interval_calibration_error=0.181` is real and driven by systematic **over-coverage at every nominal level** — the model's predicted intervals are far wider than they need to be (a 50%-nominal interval actually captures 91.8% of true values; even the 95%-nominal interval overshoots to 99.2%). This is a genuine, reproducible finding, not noise or a leftover bug: the point predictions are excellent (r=0.88) but the Gamma `(k, θ)` uncertainty head is materially over-conservative.

## Verdict: still CONDITIONAL / STAGING ONLY — uncertainty gate remains NOT PASSED

| Criterion (from 2026-08-28 doc) | Bar | Observed (real, post-fix) | Pass? |
|---|---|---|---|
| Point Pearson | `>0.7` strong | **0.881** | **PASS** |
| Interval calibration error | `<0.07` suggested, `<0.05` preferred | **0.181** | **NOT PASS** |

Do not flip `MODEL_RELEASE_APPROVED` on the strength of this result alone. The point-prediction promotion path (§8) remains valid on its own terms; a "calibrated uncertainty" claim specifically is not yet supportable.

## Recommended next step

Systematic over-coverage across all four nominal levels (50/80/90/95, monotonically improving gap as nominal increases) is the classic signature of an uncertainty head that needs **temperature/variance scaling**, not a redesign: fit a single scalar multiplier on `theta` (or divisor on the implied variance) against this same held-out split to shrink predicted intervals until empirical coverage matches nominal, then re-run this exact evaluation to confirm. This is a small, well-scoped follow-up — do not re-attempt a blind retrain.
