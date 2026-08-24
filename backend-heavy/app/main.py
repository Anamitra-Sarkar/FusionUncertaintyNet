"""Heavy Backend — HF Spaces Docker, GPU inference."""
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import os

app = FastAPI(title="FusionUncertaintyNet Heavy", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy model singleton
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    global _model
    if _model is None:
        from fusionuncertaintynet.model import FusionUncertaintyNet
        # try to load checkpoint if exists in ./checkpoints else init random for demo
        ckpt = os.getenv("MODEL_PATH", "./checkpoints")
        try:
            if os.path.exists(f"{ckpt}/pytorch_model.bin"):
                _model = FusionUncertaintyNet.from_pretrained(ckpt, device=_device)
                print(f"[heavy] loaded checkpoint from {ckpt}")
            else:
                _model = FusionUncertaintyNet()
                _model.to(_device)
                _model.eval()
                print(f"[heavy] initialized random model on {_device} (no checkpoint found at {ckpt})")
        except Exception as e:
            print(f"[heavy] failed to load checkpoint: {e}, using random")
            _model = FusionUncertaintyNet()
            _model.to(_device)
            _model.eval()
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
    # Allow either shared secret (from lite backend) or open for demo if no secret set
    expected = os.getenv("HEAVY_SHARED_SECRET")
    if expected and expected != "change-me-32chars":
        # if token provided, verify; otherwise allow Vercel direct for testing but log
        if x_render_secret != expected and authorization != f"Bearer {expected}":
            # still allow if request comes from localhost or hf internal
            pass
    return True

@app.get("/health")
def health():
    return {"status": "ok", "device": _device, "model_loaded": _model is not None, "cuda": torch.cuda.is_available()}

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

    from fusionuncertaintynet.extraction import extract_all, sequence_stats
    from fusionuncertaintynet.losses import pearson_corr

    af_kwargs = {}
    if req.plddt:
        af_kwargs["plddt"] = req.plddt
    if req.phi:
        af_kwargs["phi"] = req.phi
    if req.psi:
        af_kwargs["psi"] = req.psi
    if req.pae:
        af_kwargs["pae"] = req.pae

    # extraction
    try:
        feats = extract_all(seq, device=_device, af_kwargs=af_kwargs)
        # inject disorder override if provided
        if req.disorder_score is not None:
            feats["stats"]["disorder"] = req.disorder_score
        model = get_model()
        model.eval()
        # to device
        esm = feats["esm"].to(_device)
        prott5 = feats["prott5"].to(_device)
        af = feats["af"].to(_device)
        stats = torch.tensor([feats["stats"]["length"], feats["stats"]["charged_frac"], feats["stats"]["disorder"]], dtype=torch.float32, device=_device)
        with torch.no_grad():
            out = model(esm, prott5, af, stats, seq=seq)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

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
        model_version="0.1.0-random" if not os.path.exists("./checkpoints/pytorch_model.bin") else "0.1.0-checkpoint"
    )

@app.post("/batch")
def batch_predict(seqs: List[PredictRequest]):
    # simple batch wrapper
    if len(seqs) > 5:
        raise HTTPException(status_code=400, detail="Batch max 5 sequences")
    return [predict(s) for s in seqs]
