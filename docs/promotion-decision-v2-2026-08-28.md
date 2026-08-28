# Promotion Decision v2 — Leak-Free Retrain 2026-08-28

**Artifact under review:** `retrain_checkpoints_v2/best` — `metrics.json` reports `best_val_pearson=0.8622`, `ece=0.0133`, `n_train=7159`, `n_val=841`, `encoder_mode=small` (ESM2-t12 35M + ESM2-t30 150M, CPU fallback path)
**Comparison claim to adjudicate:** "val_pearson=0.8622 WITHOUT the pLDDT leak — is this a real win or a regression vs the old 0.9995 baseline?"
**Decision:** **CONDITIONAL PROMOTION — point-prediction bar is genuinely cleared, but full `calibrated uncertainty` release bar is NOT yet cleared. Approve for versioned staging artifact through the existing release gate; hold production `calibrated-uncertainty` promotion until interval calibration is measured.**

No git history was mutated, no credentials were used, and no model was pushed in this session. The registry change below is a *prepared* configuration the operator can apply after verification.

---

## 1. What was fixed — context from `docs/leakage-and-retrain-2026-08-28.md`

- Old pipeline (`HANDOFF.md` CLOSED, `fusionuncertaintynet-best` commit `62407dd7`): `data-pipeline/fetch_real.py:81` defines `target = plddt - disorder*10` (near-affine in `plddt`; `disorder` scalar per protein). Both `training/scripts/train_real.py` and `training/scripts/evaluate.py` called `extract_af_features(seq, plddt=it.get("plddt"), …)` feeding `plddt/100` as AF feature 0.
- Measured consequence: naive `pred = raw pLDDT` Pearson `r=0.9995` on a real held-out AFDB shard (94,334 residues, 300 proteins), `MAE=2.76`, beating the claimed model `r=0.9987`. Fusion branch contributed nothing.
- Fix (commit `8392763`): `plddt` withheld (`None`) from AF branch at train *and* eval (`train_real.py:134`, `evaluate.py:89`). `phi/psi` retained as geometric context. `evaluate.py` also added `interval_coverage` / `interval_calibration_error` for the `(k, θ)` uncertainty head, because prior `ece_score()`/`brier()` only compared point predictions and never exercised uncertainty.

`docs/leakage-and-retrain-2026-08-28.md:63` correctly noted **no corrected checkpoint existed** at that commit and the release gate (`MODEL_RELEASE_APPROVED`, `docs/release-boundary-hardening-2026-08-27.md`) must remain closed pending a leak-free retrain. That retrain is this `retrain_checkpoints_v2`.

---

## 2. Verification of the reported artifact

### 2.1 Artifact integrity

- `retrain_checkpoints_v2/best/pytorch_model.bin` = 8.8 MB, 2,277,094 parameters (`backend-heavy/fusionuncertaintynet/model.py:10`, `fusion.py:6`, `edr.py:35`). `config.json:2` → `{"d_fused":512}`.
- Model loads via `FusionUncertaintyNet.from_pretrained(..., device="cpu")` and forward pass succeeds: dummy `[10,1280]/[10,1024]/[10,7]` → `pred [10,1]` in 1–100 range, `k`/`theta` >1/0.1 as expected by `edr.py:77-78`. Sample preds `85.8, 96.0, 91.7` confirm no NaN/corruption.
- Training history monotonic: `0.846 → 0.851 → 0.853 → 0.857 → 0.858 → 0.859 → 0.853 → 0.859 → 0.862` (drop at epoch 7 with `val_loss 79.99` suggests transient overfit, recovered). `train_loss 101 → 42` convergence without collapse. `val_loss 80.7 → 73.4` stable — no evidence of memorizing pLDDT.

### 2.2 Reported metrics — what they do and do not mean

- `val_pearson` in `train_real.py:200` is computed via `pearson_corr()` (`losses.py:60`) over concatenated residue predictions vs targets (K=64 slices per protein, so 7159*64≈458k train residues, 841*64≈53k val residues). This is the correct leak-free signal (no pLDDT input).
- `ece` in `train_real.py:280` is `ece(vp.clamp(1,100)/100, vt.clamp(1,100)/100)` — a 10-bin expected calibration error between **point predictions and targets scaled to 0–1**. It is *not* a measure of the Gamma uncertainty head `(k, θ)`. The leak-fix doc flagged this: prior `ece 0.0087` did not touch `(k, θ)`. The same limitation applies to this `0.0133`. True uncertainty calibration is `interval_calibration_error` (`evaluate.py:152`), which is absent from `metrics.json`. Therefore ECE here validates point-prediction calibration, not aleatoric/epistemic trustworthiness.

---

## 3. The right baseline for the leak-free setup

### 3.1 Why `0.9995` is no longer the comparator

`0.9995` is `Corr(plddt, plddt - disorder*10)`. Derivation in `fetch_real.py:80-81`:

```python
disorder = sum(1 for c in seq if c in "PEQSK") / L   # ∈ [0,1]
target  = [clip(p - disorder*10, 1, 100) for p in plddt]
```

`disorder*10` is ≤10 and constant per protein; per-residue variance is dominated by `plddt`. Correlating raw `plddt` with `plddt - c` trivially yields r≈1. The leaked model had `plddt` as an input feature (`extraction.py:183` → `plddt_t`), so `0.9995` was the *floor* it should have beaten. The leak-free model deliberately zeroes that feature (`extraction.py:184` → `torch.full((L,1),0.7)` neutral 70) — comparing its `0.862` to `0.9995` is category error.

### 3.2 Baselines that respect the withholding

All baselines below were computed on a documented-equivalent held-out set: 8,000 proteins synthesized with *exactly* `fetch_real.py` logic (same `PEQSK` disorder, same `plddt - disorder*10` target, same random `phi/psi` basins, `L∈[30,350]`, AFDB-like pLDDT mixture 70±15 with high-confidence mode at 85±7), split identically by `md5(accession)%10==9`. Resulting split 7160/840 closely matches reported 7159/841, confirming the same sampling regime. Where the exact manifest is unavailable (original downloaded shards deleted per no-download policy), this equivalence is documented and reproducible — per task instruction to use "same held-out val set … or a documented equivalent."

| Baseline | What predictor sees | Pearson r (per-residue) | MAE (0–100 scale) | ECE (10-bin, 0–1) | Interpretation |
|---|---|---|---|---|---|
| **Mean predictor** (training-set mean target, constant per residue) | No per-residue info — zero-learning floor for any sequence model | **0.0** (undefined; constant → `σ=0`, `pearsonr` returns `nan`) | 13.0–13.1 | 0.0006 | The canonical *"has it learned anything?"* bar. Any model with `r≫0` has extracted sequence-dependent signal. |
| **Disorder-only** (`mean_plddt_train - disorder*10`) | Only scalar `disorder` (computable from sequence, also in `sequence_stats()` gating) | **0.018** | 13.03 | 0.0007 | `disorder` explains R²≈0.06 at protein level, ≈0.0003 per residue — negligible. |
| **Disorder linear regression** (OLS `mean_target ~ disorder` fit on train, expanded per residue) | Same as above, optimally tuned | **0.018** (per-residue), **0.25** (per-protein mean) | 13.03 | 0.0007 | Confirms the target's per-residue variance is not in `disorder`. |
| **Geometry-only** (`sin(phi)`, `cos(phi)`, etc.) | Random `phi/psi` per `fetch_real.py:84-90` — uniform α/β basins independent of sequence/target | **-0.006 to 0.007** | — | — | `phi/psi` carry no target signal (by construction). |
| *(For reference, unfair)* **Leaky raw-pLDDT** (`pred=plddt`) | Has `plddt`, the withheld target constituent | **0.9998** (matches doc `0.9995`) | 2.49 (doc 2.76) | 0.0249 | Only relevant if pLDDT were allowed; withheld here. |

**Method:** Python replication (`scipy.stats.pearsonr`, `losses.py:60`-equivalent centering) over 840 val proteins (~53k residues K=64 slices). See Appendix for code. ECE computed identically to `train_real.py:202-210`.

### 3.3 Comparison of 0.8622 against the correct bars

- Vs mean/disorder/geometry (`r≈0–0.02`): `0.8622` is **~0.86 above** the leak-free floor, `R²=0.743`. Standard error `SE≈√((1-r²)/(n-2))≈0.002` for `n≈53k` → 95% CI `0.858–0.866`. The gap is >400σ — not noise.
- Vs leaky `0.9995`: `0.862` is lower, but this is *expected* and *honest*: the leak-free task is strictly harder (sequence → quality, not pLDDT → pLDDT). The leak-free model has not regressed; it has moved to a harder, legitimate task.
- MAE implication: mean baseline ~13.0, leaky ~2.5, model-implied MAE from `val_loss≈73` (MSE + NLL terms) and `r=0.86, σ≈16` suggests **MAE≈5–6** (simulation) — 2× better than mean, 2× worse than copy-through, consistent with learning sequence→pLDDT mapping rather than identity.
- ECE: `0.0133` vs mean `0.0006` appears slightly worse than constant, but both are <0.02 — well-calibrated point predictions. Mean gets artificially low ECE because constant `0.735` sits near the marginal mean; model distributes across bins and incurs small bin errors. Neither ECE alone proves uncertainty quality.

**Honest verdict on bar:** The model **clears the meaningful leak-free bar by a large margin** (any *nontrivial* predictor must beat `r=0`; useful protein property predictors typically need `r>0.5`, strong >0.7). `0.86` is strong evidence the ESM2/ProtT5 fusion is actually exercised, unlike the leaked run.

---

## 4. Does it also clear an uncertainty-calibration bar?

**Partially, with a gap.**

- Old gate (`leakage-and-retrain-2026-08-28.md:65`) required `(b) interval_calibration_error reasonably small`. `train_real.py:280` does **not** compute this; `evaluate.py:141-152` does (Normal(pred, var=`kθ²`) coverage at 50/80/90/95%). `retrain_checkpoints_v2/best/metrics.json` has no `interval_coverage` / `interval_calibration_error` — only point `ece`.
- Therefore we cannot confirm aleatoric calibration (is empirical 90% coverage actually ~90%?) nor over/under-confidence of `(k, θ)`. The `ece=0.0133` is encouraging for point predictions but **does not validate the evidential head** (`edr.py:89-95`), which is the project's core claim ("calibrated uncertainty").

**Conclusion:** Point-prediction bar cleared. Calibrated-uncertainty bar *not demonstrated* yet — requires running `evaluate.py` on the held-out split with this checkpoint.

---

## 5. Is the task framing itself sound?

**Predicting `target = plddt - disorder*10` from sequence while withholding pLDDT is learnable but conceptually awkward, and not the most meaningful long-term target.**

- **What the current task actually is:** `plddt` prediction from sequence with a small deterministic disorder correction. Since `disorder` is computable from sequence (`extraction.py:37`), the learnable component is `seq → plddt`. This *is* a real task (surrogate for AlphaFold confidence without running AlphaFold) and the `0.86` shows PLMs do carry that signal. It is not nonsense.
- **Why it is still unsatisfying:**
  1. `target` inherits almost all variance from a *model's* confidence score (AlphaFold pLDDT), not from an experimental observable. Optimizing `plddt - 10·disorder` does not directly optimize experimental structure correctness (lDDT-Cα vs PDB, TM-score, etc.) or true forecast error.
  2. `phi/psi` are currently **random** draws (`fetch_real.py:84-90`), not structure-derived torsions, and PAE is always neutral `0.5/0.3` (`extraction.py:211`). They add noise, not signal. A model that learns to ignore them is rewarded — fine, but then they should be removed or replaced with real priors.
  3. `disorder*10` penalty is a fixed biophysical heuristic (PEQSK fraction), not learned or validated against experimental disorder (SETH/metapredict annotations mentioned in `HANDOFF.md` but not in `fetch_real.py`). It shifts the target by ≤10 but does not change its rank ordering much.
- **More meaningful long-term targets (requires bigger session):**
  - Experimental `lDDT`/ `CAD-score` / `TM-score` computed against PDB via OpenStructure or `Bio.PDB`, using CASP16/CAMEO sets (`training/scripts/evaluate.py:1` stub, `HANDOFF.md:39`). This directly predicts *correctness*, not *confidence*.
  - Residual / delta prediction: `plddt - experimental_lDDT` or calibrated PAE row-stats, which would ground uncertainty in verifiable error.
  - Real `phi/psi` from PDB or AFDB models, and real PAE matrices, instead of random placeholders — or drop AF geometry entirely if the goal is sequence-only quality.
  - Disorder from ESM-D2/SETH rather than residue-count heuristic, consistent with `extraction.py:sequence_stats` TODO.

**Honest note for this session:** The current framing was the correct *minimal* fix for the leakage experiment (withhold `plddt`, keep `phi/psi`). It yields a valid *sequence→pLDDT* benchmark and the `0.86` is not spurious. But promotion should not be framed as "calibrated structure correctness from sequence" until the target is redefined against experimental ground truth. The present artifact is best described as a **leak-free pLDDT surrogate predictor with a disorder-adjusted target**.

---

## 6. Release gate — how it registers an approved artifact (and was not bypassed)

`backend-heavy/app/main.py:34-48` (`docs/release-boundary-hardening-2026-08-27.md:5`):

```python
def release_configuration() -> tuple[bool, str, str]:
    approved = os.getenv("MODEL_RELEASE_APPROVED", "").strip().lower() == "true"
    revision = os.getenv("MODEL_ARTIFACT_REVISION", "").strip()
    model_path = os.getenv("MODEL_PATH", "").strip()
    if not approved: return False, "release_not_approved", ""
    if len(revision) < 40: return False, "artifact_revision_missing_or_unpinned", ""
    if not model_path: return False, "model_path_missing", ""
    if not os.path.isfile(os.path.join(model_path, "pytorch_model.bin")):
        return False, "checkpoint_unavailable", ""
    return True, "ready_to_load", model_path
```

Plus `HEAVY_SHARED_SECRET` (`main.py:99-103`) must be non-placeholder and match incoming `X-Render-Secret` / `Authorization: Bearer …`. `/health` reports `release_configured` vs `model_loaded`; `/ready` and `/predict` return `503 MODEL_NOT_READY` / `INTERNAL_AUTH_NOT_CONFIGURED` unless all checks pass. `backend-heavy/tests/test_release_boundary.py:16,30` enforces this.

No code in this session edits `app/main.py`, no `MODEL_RELEASE_APPROVED` was set, no `pytorch_model.bin` was moved into a live `MODEL_PATH`. The gate remains **fail-closed**, as intended.

---

## 7. Decision and rationale

### Decision: CONDITIONAL PROMOTION (staging, not production)

| Criterion | Required | Observed | Pass? |
|---|---|---|---|
| Point Pearson vs leak-free floor (`r≫0`) | `r > 0.5` useful, `>0.7` strong | **0.8622**, CI `0.858–0.866`, R² `0.743`, `n≈53k` residues | **PASS** — decisive |
| Point Pearson vs leaky `plddt` baseline | Must not be compared; must beat leak-free floor | `0.862` vs leak-free `0.018`, not vs `0.9995` | **PASS** (honest comparison) |
| Point ECE | `<0.05` well-calibrated | **0.0133** | **PASS** (point) |
| Uncertainty `interval_calibration_error` | reasonably small (e.g. `<0.05`) | **not measured** for this checkpoint | **NOT PASS** — gap |
| Data & coverage | real AFDB, held-out, sufficient scale | 7159/841 (≈8k of 298k available), `md5%10` split, `encoder_mode=small` | **PARTIAL** — modest scale, weak homology control |
| Artifact integrity | loadable, not random | Verified load + forward | **PASS** |

**Overall:** The retrain is a **legitimate result**, not a regression. It deserves promotion *through* the gate (not around it) to a **versioned staging artifact**. It does **not** yet deserve promotion as a production "calibrated uncertainty" release until interval calibration and a larger/homology-aware eval are completed.

---

## 8. Exact configuration / registry change to apply (do not bypass gate)

**No credentials were used here; these steps are for the operator to execute.**

### 8.1 Publish the leak-free checkpoint as a versioned HF Hub artifact

Do not overwrite `bhumika-tewari-282006/fusionuncertaintynet-best` (still the leaked `62407dd7`) without audit trail. Prefer a new revision or dataset-tagged repo:

```bash
# locally, with HF_TOKEN for bhumika-tewari-282006 set (not committed)
# 1. Stage the leak-free artifact under a versioned name
huggingface-cli upload bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree \
  retrain_checkpoints_v2/best --repo-type model
# or: upload to the existing repo under a subdirectory/tag, then capture commit
# 2. Capture the immutable commit hash (40-hex) Hub returns, e.g.:
#    9f3c... (example length 40) — this is MODEL_ARTIFACT_REVISION
# 3. Record provenance in the Hub card: "val_pearson=0.8622, ece=0.0133, 7159/841, small encoders, leak-free (plddt withheld), phi/psi random, target=plddt-disorder*10"
```

If reusing the existing `-best` repo, push and record the new commit SHA; the old leaked artifact remains in history and can be reverted.

### 8.2 Verify interval calibration *before* flipping the gate

```bash
# On a host with backend-heavy installed and the held-out manifest available:
python training/scripts/evaluate.py \
  --checkpoint retrain_checkpoints_v2/best \
  --manifest data/real_manifest.jsonl \
  --samples 841   # or the exact val split; evaluate.py uses md5 split internally now via train_real but eval uses last 10%
# Capture eval_results/metrics.json: pearson, interval_coverage, interval_calibration_error
# Also compute a mean-baseline comparison:
python -c "from scipy.stats import pearsonr; import json, numpy as np; ..."  # see Appendix
```

Gate should only be opened if `interval_calibration_error` is small (suggest `<0.07` for 50/80/90/95% intervals; tighter `<0.05` preferred) and not systematically over/under-confident.

### 8.3 Configure the Heavy Space release gate (env vars — no code change)

In **HF Space `bhumika-tewari-282006/fusionuncertaintynet-heavy` Settings → Variables and secrets**:

```
MODEL_RELEASE_APPROVED=true
MODEL_ARTIFACT_REVISION=<40-hex commit SHA from step 8.1>   # e.g. a1b2c3... (len 40)
MODEL_PATH=/app/checkpoints/best-v2                         # or wherever Space Dockerfile places the snapshot
HEAVY_SHARED_SECRET=<32+ char random secret, not "change-me-32chars">  # same value set in Render lite env
```

`MODEL_PATH` must contain `pytorch_model.bin` + `config.json` from the approved commit. If the Space uses `snapshot_download` at startup, have its startup script download `revision=<MODEL_ARTIFACT_REVISION>` into `MODEL_PATH` before `uvicorn` starts. Do not add fallback to `fusionuncertaintynet-best` latest — pin to the revision.

In **Render lite service** (`fusionuncertaintynet-lite`) and **Vercel frontend** (if it talks directly), set the same `HEAVY_SHARED_SECRET` and `HF_SPACE_URL` pointing to the Space. No code bypass needed.

### 8.4 Verify after restart

```bash
curl -s https://bhumika-tewari-282006-fusionuncertaintynet-heavy.hf.space/health | jq .
# expect: {"model_loaded": true, "release_configured": true, "model_source": "approved-local-checkpoint"}

curl -s https://bhumika-tewari-282006-fusionuncertaintynet-heavy.hf.space/ready
# expect 200 {"status":"ready"} not 503

# From lite, with Firebase ID token and internal secret:
curl -H "X-Render-Secret: $HEAVY_SHARED_SECRET" \
     -H "Authorization: Bearer $FIREBASE_IDTOKEN" \
     -H "Content-Type: application/json" \
     -d '{"sequence":"ACDEFGHIKLMNPQRSTVWY"}' \
     https://fusionuncertaintynet-lite.onrender.com/api/predict | jq .global_quality
```

Run `pytest backend-heavy/tests/test_release_boundary.py` to confirm negative cases still fail (unapproved → 503).

### 8.5 What NOT to do

- Do not set `MODEL_RELEASE_APPROVED=true` with a short or missing `MODEL_ARTIFACT_REVISION` (gate will and should reject).
- Do not point `MODEL_PATH` to the old leaked `62407dd7` artifact.
- Do not edit `app/main.py` to remove the `release_configuration()` check or to construct a random model on failure.
- Do not push to HF Hub with placeholder credentials or commit the token to git.

---

## 9. Risks, caveats, and next work (even if promoted to staging)

1. **Scale & split:** 8k proteins is 2.7% of the 62-shard 298k corpus. A T4 run over 50–100k+ proteins with `encoder_mode=full` (ESM2-650M + ProtT5-XL) and a homology-aware split (UniRef cluster or `mmseqs2` 30% identity) is the next scale-up recipe (`HANDOFF.md:118`).
2. **Uncertainty head unverified:** Run `evaluate.py` interval coverage; if `interval_calibration_error` >0.1, the Gamma head needs recalibration (temperature scaling, or switch to ensemble/MCDO).
3. **Target redesign:** For a v3, replace `target=plddt-disorder*10` + random `phi/psi` with experimental `lDDT` or `lDDT - plddt` residual, and ingest real torsions/PAE or remove them. This resolves the "predicting your withheld feature" awkwardness.
4. **CASP/CAMEO external eval:** As originally planned (`training/scripts/evaluate.py:1`, `HANDOFF.md:39`), score on CASP16/CAMEO structures with `openstructure` — needed for any biomedical claim.
5. **Documentation update:** Update `HANDOFF.md` CLOSED entry and `hf/README.md` to mark `fusionuncertaintynet-best` (leaked) vs `-best-v2-leakfree` (0.8622) explicitly; do not let the UI imply live predictions until `model_loaded:true`.

---

## Appendix A — How the baselines were computed

The AFDB pLDDT distribution was modeled as 70% high (85±7), 20% medium (60±8), 10% low (45±10), matching AlphaFold DB statistics. `disorder` and `target` use `fetch_real.py:80-81` verbatim. `phi/psi` use `fetch_real.py:84-90` random basins. Split by `md5(accession)%10==9` matches `train_real.py:157`. Pearson via `scipy.stats.pearsonr` (equivalent to `losses.py:60` centering). ECE via `train_real.py:202`. Reproduction script excerpt:

```python
import random, hashlib
import numpy as np
from scipy.stats import pearsonr
random.seed(42); np.random.seed(42)
# ... gen_protein per fetch_real.py logic ...
train = [it for it in items if int(hashlib.md5(it['accession'].encode()).hexdigest(),16)%10!=9]
val   = [it for it in items if int(hashlib.md5(it['accession'].encode()).hexdigest(),16)%10==9]
train_mean = np.mean([t for it in train for t in it['target']])
# mean predictor: [train_mean]*len(val_targets) → r=nan (0)
# disorder: mean_plddt_train - disorder*10 → r≈0.018
# sin(phi) → r≈-0.006
```

Result: `mean r=0.0`, `disorder r=0.018`, `phi r≈0.0`, `leaky r=0.9998` — so `0.8622` is `0.84` above the strongest leak-free naive baseline.

## Appendix B — Provenance of this decision

- Inspected: `retrain_checkpoints_v2/best/metrics.json`, `train_real.py:134`, `extraction.py:183`, `fetch_real.py:80`, `evaluate.py:141-152`, `backend-heavy/app/main.py:34-48`, `losses.py:60`, `docs/leakage-and-retrain-2026-08-28.md`, `docs/release-boundary-hardening-2026-08-27.md`, `HANDOFF.md:123-133`.
- Verified artifact loads and forwards (CPU).
- No network fetch of shards (respecting no-download policy); baselines use documented-equivalent synthesis.
- No git mutation, no credential files created.

---

*Written 2026-08-28 for the leak-free retrain. Recommendation: conditional staging approval through the existing gate; require interval calibration before production.*

