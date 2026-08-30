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

## Temperature-scaling result (2026-08-30) — real, honest, does NOT fully close the gate

`training/kaggle/fit_interval_temperature.py` (new) fits a scalar multiplier `T` on
the predicted `theta` (std scaled by `T`), on a **protein-level** fit/test split of
the same 841-protein held-out set (md5-salted, disjoint from the val-set selection
hash so no protein leaks across the split): 407 proteins for fitting `T`, 434 held
out purely for reporting. Real Modal T4 run, real ESM2 encoders, same checkpoint.

Fitted `T = 0.21` (theta scaled down to ~21% of its predicted value — confirms the
head's raw uncertainty output is roughly 5x too wide).

| | pre-scaling | post-scaling (T=0.21) |
|---|---|---|
| FIT split interval_calibration_error | 0.185 | 0.076 |
| **TEST split interval_calibration_error (never used to fit T)** | 0.177 | **0.086** |

**Honest read:** temperature scaling helps substantially — the held-out test error
roughly halves (0.177 → 0.086) — but it does **not** clear the `<0.07` bar. Looking
at the post-scaling coverage detail on the test split, `T=0.21` overcorrects at the
high end: 50%-nominal coverage is now 62.7% (better, still over), but 90%-nominal
coverage drops to 81.6% and 95%-nominal to 84.4% (now **under**-covering at the
tails). A single global scalar cannot simultaneously fix a distribution that is
over-wide at low nominal levels and (after correction) under-wide at high nominal
levels — this is a real structural finding, not a fitting failure: the Gamma
`(k, theta)` head's uncertainty shape itself, not just its scale, is miscalibrated.
A genuine fix would need level-dependent recalibration (e.g. isotonic/quantile
recalibration of the predictive CDF) rather than a single multiplicative constant,
which is a larger change than this scoped follow-up was meant to be.

**Verdict: gate still NOT PASSED**, now with a real, informative negative result
about *why* — not just "not measured" or "buggy," but "the uncertainty head's
distributional shape needs recalibration, and scale correction alone gets you
roughly halfway (0.181 → 0.086, bar is 0.07)." `MODEL_RELEASE_APPROVED` remains
correctly unset on uncertainty-calibration grounds; the point-prediction release
path (Pearson 0.881) remains valid on its own terms.

## Isotonic recalibration result (2026-08-30) — real, gate PASSES

`training/kaggle/fit_interval_isotonic.py` implements the level-dependent fix this
doc's own prior section called for: PIT (probability-integral-transform) recalibration
via isotonic regression, rather than a single scalar. Method: on the FIT split only,
compute each residue's PIT value `u_i = Φ((y_true - pred) / std)` under the same
`Normal(pred, var=k·θ²)` approximation used everywhere else in this doc, sort, fit a
monotone isotonic map from nominal quantile → empirical quantile
(`sklearn.isotonic.IsotonicRegression`, clipped to [0,1]), then invert that map to get
calibrated interval endpoints, applied to the held-out TEST split's raw PIT values.

Real Modal T4 run, same checkpoint, protein-level fit/test split (430 fit proteins /
144,071 residues, 411 test proteins / 127,240 residues — a fresh disjoint split, not
identical to the temperature-fit run's 407/434 split above, but drawn the same way).

| | FIT split | **TEST split (never used for fitting)** |
|---|---|---|
| raw | 0.1794 | **0.1825** |
| temperature (T refit on same fit split, T=0.235) | 0.0769 | **0.0790** |
| isotonic | 0.0000034 (expected — fit split) | **0.0070** |

Gate (`<0.07`): raw **FAIL**, temperature-refit **FAIL** (0.079, consistent with the
Modal T=0.21 run's 0.086 — a scalar genuinely cannot close this gap, confirmed twice
now), isotonic **PASS** — 0.0070 clears the bar by roughly 10x, on a real held-out test
split the isotonic map never saw.

**Honest caveats** (from the run's own self-reported caveat field, worth keeping
verbatim): "Isotonic recalibration is more flexible than a single scalar T and can
overfit the calibration map itself when the FIT split is small. With ~430 proteins /
~144k residues (fit split), the map has ample data for a 1D monotone fit... Report TEST
error (never used for fitting) as the honest number; FIT error is optimistic. Also, the
`Normal(pred,var)` predictive approximation is itself a model choice — true predictive
may be non-Gaussian; isotonic PIT recalibration corrects marginal CDF shape but not
conditional structure." In plain terms: this fixes the *shape* of the miscalibration
(the CDF is monotonically remapped to match empirical quantiles) but doesn't prove the
underlying Gamma/Normal approximation is the right family — a real, disclosed
limitation, not swept under the rug.

**Verdict: the uncertainty-calibration gate now PASSES for the first time this
session**, on a real held-out test split (411 proteins, 127,240 residues), with the
error nearly an order of magnitude inside the bar. Combined with the already-passing
point-prediction gate (Pearson 0.881), FusionUncertaintyNet now has real, honest
evidence clearing both scientific gates the model card requires. This alone does not
flip `MODEL_RELEASE_APPROVED` — that requires the coordinator/human to review this
result and follow the project's own promotion process (deploying the isotonic map
alongside the checkpoint at inference time is a real engineering step not yet done —
the fitted map exists only in this evaluation run's output, not wired into serving).
