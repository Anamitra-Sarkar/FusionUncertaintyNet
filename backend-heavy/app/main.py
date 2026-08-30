"""Heavy Backend — private, release-gated HF Spaces inference."""
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import os

app = FastAPI(title="FusionUncertaintyNet Heavy", version="0.1.0")


def configured_origins() -> list[str]:
    return [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy model singleton. An unloaded model is intentionally not a prediction-capable model.
_model = None
_model_source = "not-loaded"
_model_load_error: Optional[str] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy isotonic calibration artifact (opt-in, fail-closed).  Disabled by default
# — when CALIBRATION_ARTIFACT_ENABLED != "true", behavior is byte-identical to
# current production (no download, no rescaling).
_calibration_breakpoints = None
_calibration_breakpoints_error: Optional[str] = None
_calibration_breakpoints_path: Optional[str] = None


def _get_calibration_breakpoints():
    """Lazily download and cache calibration breakpoints (opt-in path).

    Env:
      CALIBRATION_ARTIFACT_ENABLED=true to enable
      CALIBRATION_ARTIFACT_REPO (default bhumi.../fusionuncertaintynet-best-v2-leakfree)
      CALIBRATION_ARTIFACT_PATH (default calibration/calibration-isotonic-v1.json)
      CALIBRATION_ARTIFACT_REVISION (optional pinned HF revision)
      CALIBRATION_ARTIFACT_LOCAL_PATH (optional local JSON override, for tests)
      HF_TOKEN (optional for private repo; public read uses anonymous)

    Fail-closed: any download/parse error is cached as _calibration_breakpoints_error
    and re-raised so the caller can fall back to raw predictions.
    """
    global _calibration_breakpoints, _calibration_breakpoints_error, _calibration_breakpoints_path
    if _calibration_breakpoints is not None:
        return _calibration_breakpoints
    if _calibration_breakpoints_error is not None:
        raise RuntimeError(_calibration_breakpoints_error)

    # Local override for tests / offline dev
    local_override = os.getenv("CALIBRATION_ARTIFACT_LOCAL_PATH", "").strip()
    if local_override and os.path.isfile(local_override):
        try:
            import json as _json

            data = _json.loads(open(local_override).read())
            bp = data.get("breakpoints", data)
            # normalize: expect dict with x/y
            if isinstance(bp, dict) and ("x" in bp or "nominal" in bp):
                _calibration_breakpoints = bp
            elif isinstance(data, dict) and "x" in data:
                _calibration_breakpoints = data
            else:
                _calibration_breakpoints = bp
            _calibration_breakpoints_path = local_override
            print(f"[heavy] calibration breakpoints loaded from local override {local_override}")
            return _calibration_breakpoints
        except Exception as e:
            _calibration_breakpoints_error = f"local calibration load failed: {e}"
            raise RuntimeError(_calibration_breakpoints_error) from e

    repo = os.getenv("CALIBRATION_ARTIFACT_REPO", "").strip() or "bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree"
    path_in_repo = os.getenv("CALIBRATION_ARTIFACT_PATH", "").strip() or "calibration/calibration-isotonic-v1.json"
    revision = os.getenv("CALIBRATION_ARTIFACT_REVISION", "").strip() or None
    token = os.getenv("HF_TOKEN", "").strip() or None

    try:
        from huggingface_hub import hf_hub_download

        kwargs = dict(repo_id=repo, filename=path_in_repo, repo_type="model")
        if revision:
            kwargs["revision"] = revision
        if token:
            kwargs["token"] = token
        local_path = hf_hub_download(**kwargs)
        import json as _json

        data = _json.loads(open(local_path).read())
        bp = data.get("breakpoints", data)
        if isinstance(bp, dict) and ("x" in bp or "nominal" in bp):
            _calibration_breakpoints = bp
        elif isinstance(data, dict) and "x" in data:
            _calibration_breakpoints = data
        else:
            _calibration_breakpoints = bp
        _calibration_breakpoints_path = local_path
        print(f"[heavy] calibration breakpoints loaded from HF {repo}:{path_in_repo}")
        return _calibration_breakpoints
    except Exception as e:
        _calibration_breakpoints_error = str(e)
        print(f"[heavy] calibration breakpoints download failed: {e}")
        raise


def _reset_calibration_cache():
    """For tests: clear cached breakpoints so env changes take effect."""
    global _calibration_breakpoints, _calibration_breakpoints_error, _calibration_breakpoints_path
    _calibration_breakpoints = None
    _calibration_breakpoints_error = None
    _calibration_breakpoints_path = None


class ModelNotReady(RuntimeError):
    """Raised when the deployed release is not independently approved and loadable."""


def release_configuration() -> tuple[bool, str, str]:
    """Return whether a specific immutable local checkpoint is eligible for loading."""
    approved = os.getenv("MODEL_RELEASE_APPROVED", "").strip().lower() == "true"
    revision = os.getenv("MODEL_ARTIFACT_REVISION", "").strip()
    model_path = os.getenv("MODEL_PATH", "").strip()
    if not approved:
        return False, "release_not_approved", ""
    if len(revision) < 40:
        return False, "artifact_revision_missing_or_unpinned", ""
    if not model_path:
        return False, "model_path_missing", ""
    if not os.path.isfile(os.path.join(model_path, "pytorch_model.bin")):
        return False, "checkpoint_unavailable", ""
    return True, "ready_to_load", model_path


def get_model():
    global _model, _model_source, _model_load_error
    if _model is None:
        eligible, reason, model_path = release_configuration()
        if not eligible:
            raise ModelNotReady(reason)
        from fusionuncertaintynet.model import FusionUncertaintyNet
        try:
            _model = FusionUncertaintyNet.from_pretrained(model_path, device=_device)
            _model_source = "approved-local-checkpoint"
            _model_load_error = None
            print("[heavy] approved checkpoint loaded")
        except Exception:
            _model = None
            _model_source = "not-loaded"
            _model_load_error = "checkpoint_load_failed"
            print("[heavy] approved checkpoint load failed")
            raise ModelNotReady("checkpoint_load_failed")
    return _model

class PredictRequest(BaseModel):
    sequence: str = Field(..., description="Amino acid sequence, up to 1022", min_length=5, max_length=5000)
    plddt: Optional[List[float]] = None
    phi: Optional[List[float]] = None
    psi: Optional[List[float]] = None
    pae: Optional[List[List[float]]] = None
    disorder_score: Optional[float] = Field(None, ge=0, le=1)

class ResidueResult(BaseModel):
    index: int
    aa: str
    pred_quality: float
    aleatoric: float
    epistemic: float
    total_unc: float
    k: float
    theta: float

class PredictResponse(BaseModel):
    sequence: str
    length: int
    global_quality: float
    global_uncertainty: float
    gates: List[float]
    residues: List[ResidueResult]
    ramachandran_outliers: int
    model_version: str

def verify_shared_secret(x_render_secret: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    expected = os.getenv("HEAVY_SHARED_SECRET", "").strip()
    if not expected or expected == "change-me-32chars":
        raise HTTPException(status_code=503, detail={"code": "INTERNAL_AUTH_NOT_CONFIGURED"})
    if x_render_secret != expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=403, detail={"code": "INTERNAL_AUTH_DENIED"})
    return True

@app.get("/health")
def health():
    configured, _, _ = release_configuration()
    return {
        "status": "ok",
        "device": _device,
        "model_loaded": _model is not None,
        "release_configured": configured,
        "model_source": _model_source,
    }


@app.get("/ready")
def ready():
    configured, reason, _ = release_configuration()
    if not configured or _model is None:
        raise HTTPException(status_code=503, detail={"code": "MODEL_NOT_READY", "reason": reason})
    return {"status": "ready", "model_source": _model_source}

@app.get("/")
def root():
    return {"service": "FusionUncertaintyNet Heavy", "docs": "/docs", "health": "/health"}

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_shared_secret)])
def predict(req: PredictRequest):
    if not req.sequence or len(req.sequence.strip()) < 5:
        raise HTTPException(status_code=400, detail="Sequence too short (min 5)")
    seq = req.sequence.strip().upper().replace(" ", "").replace("\n", "")
    if len(seq) > 1022:
        # truncate but warn via header? just truncate for model limit
        seq = seq[:1022]
    # validate AA
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    # map invalid to A internally, but warn if too many
    invalid_count = sum(1 for c in seq if c not in valid)
    if invalid_count > len(seq) * 0.3:
        raise HTTPException(status_code=400, detail=f"Sequence contains too many non-standard AAs ({invalid_count}/{len(seq)})")

    try:
        model = get_model()
    except ModelNotReady as error:
        raise HTTPException(status_code=503, detail={"code": "MODEL_NOT_READY", "reason": str(error)})

    af_kwargs = {}
    if req.plddt:
        af_kwargs["plddt"] = req.plddt
    if req.phi:
        af_kwargs["phi"] = req.phi
    if req.psi:
        af_kwargs["psi"] = req.psi
    if req.pae:
        af_kwargs["pae"] = req.pae

    try:
        from fusionuncertaintynet.extraction import extract_all
        feats = extract_all(seq, device=_device, af_kwargs=af_kwargs)
        if req.disorder_score is not None:
            feats["stats"]["disorder"] = req.disorder_score
        model.eval()
        esm = feats["esm"].to(_device)
        prott5 = feats["prott5"].to(_device)
        af = feats["af"].to(_device)
        stats = torch.tensor([feats["stats"]["length"], feats["stats"]["charged_frac"], feats["stats"]["disorder"]], dtype=torch.float32, device=_device)
        with torch.no_grad():
            out = model(esm, prott5, af, stats, seq=seq)
    except Exception:
        print("[heavy] inference pipeline unavailable")
        raise HTTPException(status_code=503, detail={"code": "INFERENCE_UNAVAILABLE"})

    pred = out["pred"]  # [L,1]
    k = out["k"]
    theta = out["theta"]
    ale = out["aleatoric"]
    epi = out["epistemic"]
    tot = out["total_unc"]
    gates = out["gates"].tolist() if isinstance(out["gates"], torch.Tensor) else out["gates"]

    # flatten
    L = len(seq)
    pred_list = pred.squeeze(-1).cpu().tolist()
    ale_list = ale.squeeze(-1).cpu().tolist()
    epi_list = epi.squeeze(-1).cpu().tolist()
    tot_list = tot.squeeze(-1).cpu().tolist()
    k_list = k.squeeze(-1).cpu().tolist()
    th_list = theta.squeeze(-1).cpu().tolist()

    # Opt-in isotonic calibration: strictly gated, fail-closed, byte-identical when disabled.
    # When CALIBRATION_ARTIFACT_ENABLED=true, breakpoints are lazily downloaded
    # (cached) and theta/aleatoric/total_unc are rescaled via the shared
    # pure-numpy helper (var = k*theta^2, ale = var, epi = 1/k, tot = ale + 50*epi,
    # matching edr.py).  Any failure falls back to raw values.
    if os.getenv("CALIBRATION_ARTIFACT_ENABLED", "").strip().lower() == "true":
        try:
            # lazy import keeps default path free of calibration dependency
            from fusionuncertaintynet.calibration import apply_isotonic_calibration

            import numpy as _np

            bp = _get_calibration_breakpoints()
            k_arr = _np.array(k_list, dtype=float)
            th_arr = _np.array(th_list, dtype=float)
            pred_arr = _np.array(pred_list, dtype=float)
            k_cal, th_cal, ale_cal, epi_cal, tot_cal = apply_isotonic_calibration(
                pred_arr, k_arr, th_arr, bp
            )
            # overwrite reported uncertainty (k unchanged, theta/ale/tot calibrated)
            th_list = th_cal.tolist()
            ale_list = ale_cal.tolist()
            epi_list = epi_cal.tolist()
            tot_list = tot_cal.tolist()
            # k_list intentionally unchanged (isotonic calibrates scale, not shape param)
            print("[heavy] isotonic calibration applied")
        except Exception as e:
            print(f"[heavy] calibration skipped (fail-closed): {e}")
            # keep original raw values — byte-identical fallback
            pass

    residues = []
    outliers = 0
    for i, aa in enumerate(seq):
        # count high uncertainty as outlier proxy
        if tot_list[i] > 2.0:
            outliers += 1
        residues.append(ResidueResult(
            index=i+1,
            aa=aa,
            pred_quality=float(max(0, min(100, pred_list[i]))),
            aleatoric=float(ale_list[i]),
            epistemic=float(epi_list[i]),
            total_unc=float(tot_list[i]),
            k=float(k_list[i]),
            theta=float(th_list[i])
        ))

    global_q = float(sum(pred_list)/len(pred_list)) if pred_list else 0.0
    global_u = float(sum(tot_list)/len(tot_list)) if tot_list else 0.0

    return PredictResponse(
        sequence=seq,
        length=L,
        global_quality=global_q,
        global_uncertainty=global_u,
        gates=gates,
        residues=residues,
        ramachandran_outliers=outliers,
        model_version=f"0.1.0-{_model_source}"
    )

@app.post("/batch", dependencies=[Depends(verify_shared_secret)])
def batch_predict(seqs: List[PredictRequest]):
    # simple batch wrapper
    if len(seqs) > 5:
        raise HTTPException(status_code=400, detail="Batch max 5 sequences")
    return [predict(s) for s in seqs]
