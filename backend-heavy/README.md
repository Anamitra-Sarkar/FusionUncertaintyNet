---
title: FusionUncertaintyNet Heavy
emoji: 🧬
colorFrom: teal
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# FusionUncertaintyNet Heavy — HF Space

Docker FastAPI for GPU inference. Triggered by GitHub Actions on `backend-heavy/**`.

- `GET /health`
- `POST /predict` {sequence, plddt, phi, psi, pae}

Env: `HEAVY_SHARED_SECRET`, `MODEL_PATH=./checkpoints`
