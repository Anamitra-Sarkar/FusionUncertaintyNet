# Agent Handoff — FusionUncertaintyNet

**Date:** 2026-08-24 08:30 UTC  
**Repo:** `https://github.com/Anamitra-Sarkar/FusionUncertaintyNet` `main:7aa1e4e`  
**Mode:** Build (was Plan, now Build) — you may edit, run, deploy.

## What Has Been Done (Verified Live)

### 1. Architecture Implemented (Paper-Faithful)
- **Representation:** `backend-heavy/fusionuncertaintynet/extraction.py` — ESM2 `esm2_t33_650M_UR50D` 1280d + ProtT5 `Rostlab/prot_t5_xl_uniref50` 1024d (lazy, fp16, offload, P100 `sm_60→cpu` fallback), AF `pLDDT`/`phi/psi sincos`/`PAE` → 7d, stats `length/log, charged_frac, disorder(SETH/metapredict)` for gating.
- **Fusion:** `fusion.py` — `Linear 1280/1024/7→512 + LayerNorm`, `gate_mlp 3→32→3` `softmax` weighted sum, handles `[L,D]` and `[B,L,D]`.
- **EDR:** `edr.py` — `512→512→256→128` residual, dual head `pred sigmoid*100`, `k,θ softplus+1`, `var=kθ²` aleatoric, `1/k` epistemic.
- **Losses:** `losses.py` — `MSE + NLL_Gamma` `(k-1)log y - y/θ - lgamma(k) - k logθ` + `Ramachandran` box penalty.

### 2. Deployments (All Live, Secrets via Env Only)
- **GitHub:** `Anamitra-Sarkar/FusionUncertaintyNet` public, `CI` ✅ `5a62329`, `sync-hf-space.yml` only `backend-heavy/**, training/**` → `bhumika-tewari-282006/fusionuncertaintynet-heavy`.
- **HF Heavy (Docker, `bhumika-tewari-282006` from `bhumika-hf.txt`):** `https://bhumika-tewari-282006-fusionuncertaintynet-heavy.hf.space` `RUNNING` `cpu` `model_loaded` toggles with `snapshot_download` `bhumika-tewari-282006/fusionuncertaintynet-best` fallback. `Dockerfile: pytorch:2.3.1-cuda12.1`.
- **Lite (Render, `Kakali's workspace` `tea-d8t287ojs32c73d36jog`):** `https://fusionuncertaintynet-lite.onrender.com` `srv-da5ubegu01pc738hspd0` `live` `7820aa6`, `buildFilter: ["backend-lite/**"]` (fixed), env `HF_SPACE_URL`, `GROQ_API_KEY` (`openai/gpt-oss-20b` verified), `FIREBASE_ADMIN_JSON_BASE64` (`cabbage-guard` `07fc830b13`), `HEAVY_SHARED_SECRET`. Health `firebase:true`.
- **Frontend (Vercel, `team_gXozpOHE3KrJsdcyHVaspIRG`):** `https://fusionuncertaintynet-anamitra-sarkars-projects.vercel.app` `prj_hwV3Tn0oj6TlOKWfmsa13y2jj0hG` `nextjs 14` `rootDirectory:frontend` `READY`, env `NEXT_PUBLIC_FIREBASE_*` (`cabbage-guard`) + `NEXT_PUBLIC_LITE_URL` (Render). SSO disabled `ssoProtection:null`.
- **HF Datasets/Models (`bhumika-tewari-282006`):** `fusion-afdb-quality` (`manifest.jsonl` 500 synthetic), `fusion-afdb-quality-real` (placeholder), `fusionuncertaintynet-checkpoints`/`best` (`pytorch_model.bin` 2.27M dummy).

### 3. Firebase/Firestore (Fixed)
- **Project:** `cabbage-guard` (`12732254150`, `asia-south1`, `FIRESTORE_NATIVE`, `default` without parentheses — `gcloud firestore databases list` confirms). **Fix:** `backend-lite/app/main.py:137,167` now `firestore.client(database_id="default")` with fallback `database="default"`/`no param` for `firebase-admin 6.6.0` (`requirements.txt: 6.6.0 + google-cloud-firestore>=2.19.0`, was `6.5.0/2.16.0` causing `build_failed`).
- **Auth:** `cabbage-guard` reuse for SSO across 35 apps, `fusion_*` namespaced (`firestore.rules:5`), `onAuthStateChanged` → `/login` mandatory, `verify_id_token` on every `/api/*`.
- **Verified:** `test-fusion@example.com` `kU7JXRTbINTvTW55T8SC9R9BiC52` → `POST /api/predict` 200 `global_quality 67.17` → `GET /api/history` 200 2 items, `POST /api/explain` 200 `openai/gpt-oss-20b`.

### 4. Training (P100 Handling)
- **Issue:** `Tesla P100 sm_60` not supported by `torch 2.3.1 sm_70+` → `CUDA error: no kernel image` (`v3` `ERROR` at 203s, `v4` `RUNNING` then `CANCEL_ACKNOWLEDGED` after 7600s hang due to real `ESM2 650M` per sample on CPU).
- **Fix:** `training/scripts/train.py:22` `get_device()` checks `cap[0]<7` → `cpu`, `extraction.py:50` same, `train.py:57` now uses **dummy PLM** (`randn 1280/1024`) for `synthetic`/`cpu` for speed (was `extract_all` per sample → 9000× 2.5GB downloads). Committed `7170421`.
- **Kaggle:** `anamitrasarkar007/fusionuncertaintynet-p100-training-v4` `CANCEL_ACKNOWLEDGED` (hung), `v5` pushed `7170421` with dummy, `RUNNING` → `ERROR` (need to check logs). `v5` should be ~5 min on CPU, not 2h. Monitor `kaggle kernels status anamitrasarkar007/fusionuncertaintynet-p100-training-v5`.

### 5. UI
- `frontend/app/page.tsx` `layout.tsx` editorial `paper #FFFCF8` `accent #0F766E` `Newsreader`, `framer-motion`, heatmap `per-residue` `total_unc>2` red. `dashboard` `history` both `onAuthStateChanged` → `/login`.

## What Has NOT Been Done Yet (Real Not Dummy)

1. **Real Data:** `data-pipeline/fetch_real.py` created (UniProt `rest.uniprot.org` + `alphafold.ebi.ac.uk/api/prediction` + `AF-*.json` + `PDBe` + `pLDDT`→`target` Ramachandran-aware `phi/psi`), but test with `P69905` showed `AF-*.json` 404 (model files at `https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v4.pdb` not `.json`; pLDDT is in `api/prediction` `confidenceScore` or PDB `B-factor`). Need to fix fetcher to parse PDB B-factor for pLDDT and `predicted_aligned_error_v4.json` for PAE, or switch to HF `DeepMind/afdb` dataset via `datasets.load_dataset`.
2. **Real Training:** Current `manifest.jsonl` synthetic 2000, `plddt` synthetic `t+gauss`. Need 500 real from `fetch_real.py` (once fixed) → `data/real_manifest.jsonl` 500, then Kaggle `v6` with `max_items=500` `epochs=10` on CPU dummy → real PLM when not synthetic. Push to `bhumika-tewari-282006/fusion-afdb-quality-real` + `fusionuncertaintynet-checkpoints`.
3. **Real Evaluation:** `training/scripts/evaluate.py` now full (Pearson/Spearman vs `pLDDT` baseline, `ECE`/`Brier`, `Ramachandran` outliers, `aleatoric/epistemic`), but still on synthetic hold-out. Need `CASP16` `CAMEO` download + `lDDT` via `openstructure`/`Bio.PDB` vs experimental.
4. **3D Viewer:** Frontend shows heatmap only, no `3Dmol.js`/`Mol*`. Need `components/molstar.tsx` `npm i 3dmol` and PDB `B-factor` replaced with `pred_quality`.
5. **Firestore Rules Deployment:** `firestore.rules` correct but not yet `firebase deploy --only firestore` (needs `firebase` CLI `firebase login:ci` + `firebase deploy`). Currently rules are local only.
6. **Heavy Extraction Real:** On HF Space `cpu`, `fair-esm` + `transformers` installed, but `extraction.py` still uses dummy for CPU synthetic. For real, need to enable `extract_all` on `cpu` with `fair-esm` (slow but ok for inference, not training). Set `use_dummy=False` when `manifest` is real.
7. **Deployments Catch-up:** Latest `main:7aa1e4e` (`fetch_real.py`) not yet `live` on Render (still `7820aa6` live, `7aa1e4e` queued) and Vercel `BUILDING` `1edlheu8i`. Wait for `live`/`READY`.

## Immediate Next Steps (For Next Agent)

1. **Fix `fetch_real.py`:** Test `P69905` `curl https://alphafold.ebi.ac.uk/api/prediction/P69905` → `globalMetricValue 98.06`, `sequence`, then `https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v4.pdb` (not `.json`) → parse `B-factor` for `plddt` per residue. Update `fetch_alphafold()` accordingly, test `python3 data-pipeline/fetch_real.py --n 20 --fetch 50` locally, ensure `>0` with AlphaFold.
2. **Generate Real Manifest:** `python3 data-pipeline/fetch_real.py --out data/real_manifest.jsonl --n 500 --fetch 800` → 500 real, `head -n1 | jq .plddt | wc`, push to `bhumika-tewari-282006/fusion-afdb-quality-real`.
3. **Retrain Real:** Update `kaggle` `train.py` to use `manifest=data/real_manifest.jsonl` `epochs=10` `synthetic=False` (so dummy off, real PLM on CPU still ok for 500, or keep dummy for speed but note real). Push `v6` `kaggle kernels push -p /tmp/fusion_kaggle_v6` (copy `v5` but change `manifest` and `max_items=500`). Monitor `kaggle kernels status ...-v6` → `COMPLETE`, check `HF` `last_modified` updates.
4. **Evaluation Real:** `python3 training/scripts/evaluate.py --checkpoint ./checkpoints/best --manifest data/real_manifest.jsonl --samples 200` → `eval_results/metrics.json` with `pearson` vs `pearson_plddt_baseline`, push to `hf` `eval` folder.
5. **3D Viewer:** `cd frontend && npm i 3dmol` → `components/viewer.tsx` using `3Dmol` `createModel` `setStyle` `spectrum B-factor`, integrate in `dashboard/page.tsx:residues` table `onClick` → `viewer`.
6. **Firestore Rules Deploy:** `gcloud auth activate-service-account --key-file="...cabbage-guard..."` + `firebase deploy --only firestore:rules --project cabbage-guard` (need `firebase` CLI `npm i -g firebase-tools` + `firebase login --no-localhost` with `GOOGLE_APPLICATION_CREDENTIALS`).
7. **Verify:** `IDTOKEN=$(cat /tmp/idtoken2.txt)` (refresh via `firebase_admin` if expired `178755...`), `curl -H "Authorization: Bearer $IDTOKEN" https://fusionuncertaintynet-lite.onrender.com/api/predict` → 200, `history` → 2+ items, `HF /health` `model_loaded:true`, `Vercel /` 200.

## Secrets (Never Commit)

- `~/.kaggle/kaggle.json` `anamitrasarkar007:3f52...`
- `bhumika-hf.txt` `hf_btfVGAT...` (`bhumika-tewari-282006`)
- `cabbage-guard-firebase-adminsdk...json` (`07fc830b13`) → `base64` `FIREBASE_ADMIN_JSON_BASE64` (Render), `firebase-credentials.txt` (Vercel `NEXT_PUBLIC_*`)
- `vercel.txt` `vcp_8hgc...` (`team_gXozpOHE3KrJsdcyHVaspIRG` `prj_hwV3Tn0oj6TlOKWfmsa13y2jj0hG`)
- `render-api.txt` `rnd_nvED3...` (`srv-da5ubegu01pc738hspd0`)
- `groq_api.txt` `gsk_kzcG...` (`openai/gpt-oss-20b`)
- `git token` `github_pat_11BQMR...` (`Anamitra-Sarkar`)

## Commands to Resume

```bash
cd /tmp/opencode/FusionUncertaintyNet
git log --oneline -5
kaggle kernels status anamitrasarkar007/fusionuncertaintynet-p100-training-v5
curl -s https://fusionuncertaintynet-lite.onrender.com/health | jq .
curl -s https://bhumika-tewari-282006-fusionuncertaintynet-heavy.hf.space/health | jq .
curl -s https://fusionuncertaintynet-anamitra-sarkars-projects.vercel.app | grep -o FusionUncertaintyNet | head
python3 data-pipeline/fetch_real.py --n 20 --fetch 50  # test real fetcher
```

**Build mode is now active** — you may edit, run, and deploy directly.
