# Target Leakage Found in the "Closed" Training Run — 28 August 2026

## What HANDOFF.md's "fully real pipeline running" / "CLOSED ... COMPLETE" actually meant

The data pipeline is genuinely real: `fusion-afdb-quality-real` on the Hub holds
62 shards (~298k proteins) of real AlphaFold DB sequences with real per-residue
pLDDT read from PDB B-factors (independently spot-checked here against a live
shard download). The Aug-25 Kaggle run `fusion-train-real-v1` genuinely executed
10 epochs of backprop on that data with real frozen ESM2/ProtT5-family encoders
and pushed a real 9.1MB checkpoint (`fusionuncertaintynet-best`, commit
`62407dd7`) — none of that was fabricated.

What was **not** true was the implied conclusion: that a val Pearson of 0.9987
demonstrated the fusion architecture learned to predict structure quality/
reliability from sequence. It did not, because of a label-leakage bug.

## The bug

- `data-pipeline/fetch_real.py`: `target = plddt - disorder*10` (a near-affine
  function of `plddt`, since `disorder` is a single scalar per protein).
- `training/scripts/train_real.py` and `training/scripts/evaluate.py` both
  called `extract_af_features(seq, plddt=it.get("plddt"), ...)`, feeding that
  same `plddt` back in as one of the model's three input branches (the 7-d AF
  feature vector's first element is `plddt/100`).

The model only had to learn `output ≈ 100·input_plddt − disorder·10`, a task
the AF branch alone solves; the ESM2/ProtT5 fusion was never exercised by the
loss.

## Verification (real numbers, not modeled)

A real held-out AFDB shard (`fusion-afdb-quality-real/manifest_shard_000.jsonl`,
downloaded and immediately deleted after use per the local no-download policy)
was checked directly:

```
naive raw-pLDDT-as-prediction Pearson r = 0.9995  (94,334 residues, 300 proteins)
naive raw-pLDDT-as-prediction MAE       = 2.76
```

vs. the "closed" run's own reported val Pearson **0.9987**. The trivial,
zero-learning baseline the project's own spec calls for beats the trained
model. This fails the release bar outright — it is not a marginal miss.

## Fix applied (this session, commit `8392763`)

- `train_real.py` / `evaluate.py`: `plddt` withheld (`None`) from the AF input
  branch at both train and eval time. `phi`/`psi` are kept — they are geometric
  context, not a near-linear transform of the target.
- `evaluate.py`: added real predictive-interval coverage (Normal(pred, aleatoric
  variance) checked against the actual target at 50/80/90/95% nominal levels,
  plus `interval_calibration_error`). The existing `ece_score()`/`brier()` only
  ever compared two point estimates on a 0–1 scale and never touched the
  `(k, θ)` uncertainty head — i.e. the project's own claimed "ECE 0.0087" in
  the closed HANDOFF entry did not measure uncertainty calibration at all.

## State as of this commit

**No corrected checkpoint exists yet.** `fusionuncertaintynet-best` on the Hub
is still the leaked Aug-25 artifact. The release gate (`MODEL_RELEASE_APPROVED`,
`docs/release-boundary-hardening-2026-08-27.md`) correctly continues to fail
closed — this was not touched and should not be touched until a checkpoint
trained with the fix above is evaluated on a real held-out split and clears:
(a) Pearson meaningfully above the naive pLDDT baseline, and
(b) `interval_calibration_error` reasonably small (not wildly over/under-confident).

A corrected retrain (`fusion-train-real-v2` on Kaggle, same infra, fixed code)
is the necessary next step. See main session notes for its status when this
doc was written — training on a real ~300k-protein corpus on CPU/P100-class
compute is not instantaneous and should be checked back on rather than polled
tightly.
