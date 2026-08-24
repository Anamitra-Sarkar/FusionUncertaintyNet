# Deployment Wiring

## GitHub -> HF Spaces
- Repo: `Anamitra-Sarkar/FusionUncertaintyNet` (anamitrasarslsn10ab@gmail.com)
- Secret `HF_TOKEN` = HF token for `bhumika-tewari-282006` (set in GitHub Secrets)
- Workflow `sync-hf-space.yml` triggers only on `backend-heavy/**` changes
- Space: `bhumika-tewari-282006/fusionuncertaintynet-heavy` (Docker, port 7860, T4-small)

## Vercel
- Project `fusionuncertaintynet-frontend` linked to `Anamitra-Sarkar/FusionUncertaintyNet` repo, root `frontend`
- Auto-deploy on `frontend/**` push
- Env: `NEXT_PUBLIC_FIREBASE_*` + `NEXT_PUBLIC_LITE_URL`
- Domain: vercel-provided `*.vercel.app`

## Render
- Service `fusionuncertaintynet-lite` from `backend-lite/Dockerfile`
- Auto-deploy on `backend-lite/**` (set in Render dashboard filter)
- Env: `HF_SPACE_URL`, `GROQ_API_KEY`, `FIREBASE_ADMIN_JSON_BASE64`, `HEAVY_SHARED_SECRET`

## Kaggle
- Notebook `training/kaggle/train.ipynb`, secrets `HF_TOKEN`, GPU P100, Internet on
- Pushes checkpoints to `bhumika-tewari-282006/fusionuncertaintynet-checkpoints`

## Groq
- Model `openai/gpt-oss-20b` (verified available for this key, 2026-08-24)
- Alternatives: `openai/gpt-oss-120b`, `groq/compound-mini`, `qwen/qwen3.6-27b`
- Not using deprecated `llama-3` etc. — verified via live API check.

## Firebase
- Reuse `cabbage-guard` project to allow SSO across 35 apps
- Web SDK uses public apiKey; Admin SDK only on Render via base64
- Namespaced collections prevent cross-app collision
