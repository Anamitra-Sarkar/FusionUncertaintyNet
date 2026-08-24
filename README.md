# FusionUncertaintyNet

**Adaptive Multi-PLM Architecture with Evidential Deep Learning for Calibrated Protein Structure Reliability Prediction**

End-to-end system that fuses ESM-2, ProtT5 and AlphaFold-derived features via adaptive gating, and predicts per-residue quality with calibrated aleatoric/epistemic uncertainty via Evidential Deep Regression (Gamma).

## Architecture

```
Sequence -> [ESM-2 650M] -> 1280d \
         -> [ProtT5-XL]  -> 1024d -> [Adaptive Fusion (Gating MLP)] -> 512d -> [EDR Dual Head] -> pLDDT + (k, θ) -> μ=kθ, var=kθ²
         -> [AF features: pLDDT, φ/ψ, PAE] -> 7d /
                └─ stats: length, charged_frac, disorder -> gating weights
Loss: L = λ_pred·MSE + λ_ev·NLL_Gamma + λ_const·Ramachandran
```

## Repo Layout

- `frontend/` — Next.js 14 on Vercel, Firebase Auth, Firestore `fusion_*`
- `backend-lite/` — FastAPI on Render (auth proxy, Groq, Firestore)
- `backend-heavy/` — FastAPI on HF Spaces Docker (GPU inference)
- `training/` — Kaggle P100 notebooks + scripts, checkpoints -> HF Hub `bhumika-tewari-282006`
- `data-pipeline/` — AFdb + PDB alignment, PAE, SETH disorder

## Secrets — NEVER commit

All keys via env vars. See `.env.example`. GitHub Actions syncs only `backend-heavy/**` to HF Space `bhumika-tewari-282006/fusionuncertaintynet-heavy`.

## Quick Start

```bash
# heavy backend
cd backend-heavy && pip install -r requirements.txt && uvicorn app.main:app --reload --port 7860
# lite backend
cd backend-lite && pip install -r requirements.txt && uvicorn app.main:app --reload --port 8000
# frontend
cd frontend && npm install && npm run dev
```

## Training on Kaggle (P100)

1. Create Kaggle secret `KAGGLE_JSON`, `HF_TOKEN` (bhumika token)
2. Upload `training/kaggle/train.ipynb` as Kaggle Notebook, enable GPU P100, Internet on
3. Run — checkpoints auto-push to `bhumika-tewari-282006/fusionuncertaintynet-checkpoints`

## Deployment

- Frontend: Vercel (`vercel --prod`)
- Heavy: HF Space Docker (`git push hf main`)
- Lite: Render Docker (`render.yaml`)

## Evaluation

CASP16 EMA, CAMEO, Pearson/Spearman, ECE, Brier, Ramachandran Z.

## Security

Firebase `cabbage-guard` reused with `fusion_*` namespacing. Login mandatory. Every API call verifies ID token. Groq proxied via lite backend.

## Authors

Built for client project — 2026.
