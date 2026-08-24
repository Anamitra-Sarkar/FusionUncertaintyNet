"""
Representation Extraction Module
- ESM-2 t33 650M (1280d) frozen
- ProtT5-XL (1024d) frozen
- AF-derived: pLDDT, phi/psi sincos, PAE row stats
P100-friendly: sequential loading, offload to CPU, fp16, truncation at 1022.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

# lazy imports to avoid hard dep at import time
ESM2_NAME = "facebook/esm2_t33_650M_UR50D"
PROTT5_NAME = "Rostlab/prot_t5_xl_uniref50"

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

def _clean_sequence(seq: str) -> str:
    seq = seq.strip().upper().replace(" ", "").replace("\n", "")
    # map ambiguous to X->A, keep only standard; unknown -> A
    return "".join([c if c in AA_ORDER else "A" for c in seq])

def sequence_stats(seq: str, disorder_score: Optional[float] = None) -> Dict[str, float]:
    """Compute gating stats: length (log), charged_frac, disorder."""
    n = len(seq)
    if n == 0:
        return {"length": 0.0, "charged_frac": 0.0, "disorder": 0.5}
    charged = sum(1 for c in seq if c in "RKDE")
    charged_frac = charged / n
    # length log-norm: log(n)/log(2000) clipped 0-1, then scaled
    length_norm = min(math.log(max(n, 1)) / math.log(2000), 1.0)
    if disorder_score is None:
        # cheap fallback: disorder propensity via fraction of disorder-promoting residues
        disorder_promoting = sum(1 for c in seq if c in "PEQSK") / n
        disorder_score = float(min(max(disorder_promoting, 0.0), 1.0))
    return {"length": float(length_norm), "charged_frac": float(charged_frac), "disorder": float(disorder_score)}

class ESM2Extractor:
    """Lazy ESM-2 extractor; loads on first call, keeps on GPU if available."""
    def __init__(self, device: str = "auto", use_fp16: bool = True):
        self.device = self._resolve_device(device)
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.model = None
        self.alphabet = None
        self.batch_converter = None

    def _resolve_device(self, d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _load(self):
        if self.model is not None:
            return
        try:
            import esm  # type: ignore
        except ImportError:
            raise ImportError("esm not installed: pip install fair-esm")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().to(self.device)
        if self.use_fp16:
            model = model.half()
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    @torch.no_grad()
    def extract(self, seq: str) -> torch.Tensor:
        """Returns [L, 1280] per-residue (mean if batched). Truncate to 1022 for ESM limit."""
        self._load()
        seq = _clean_sequence(seq)
        # ESM limit 1022 excluding BOS/EOS; truncate
        if len(seq) > 1022:
            seq = seq[:1022]
        assert self.batch_converter is not None and self.model is not None and self.alphabet is not None
        _, _, tokens = self.batch_converter([("prot", seq)])
        tokens = tokens.to(self.device)
        # autocast for fp16
        if self.use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(tokens, repr_layers=[33])
                reps = out["representations"][33]  # [1, L+2, 1280]
        else:
            out = self.model(tokens, repr_layers=[33])
            reps = out["representations"][33]
        # remove BOS/EOS
        reps = reps[0, 1: len(seq)+1, :]  # [L,1280]
        return reps.float().cpu()  # return CPU to save GPU mem

    def offload(self):
        if self.model is not None:
            self.model = self.model.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

class ProtT5Extractor:
    """Lazy ProtT5 extractor. Falls back gracefully if OOM."""
    def __init__(self, device: str = "auto", use_fp16: bool = True):
        self.device = self._resolve_device(device)
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.tokenizer = None
        self.model = None

    def _resolve_device(self, d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _load(self):
        if self.model is not None:
            return
        from transformers import T5Tokenizer, T5EncoderModel  # type: ignore
        self.tokenizer = T5Tokenizer.from_pretrained(PROTT5_NAME, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(PROTT5_NAME)
        self.model = self.model.eval().to(self.device)
        if self.use_fp16:
            self.model = self.model.half()

    @torch.no_grad()
    def extract(self, seq: str) -> torch.Tensor:
        """Returns [L, 1024]. Space-separated AA as required by ProtT5."""
        self._load()
        seq = _clean_sequence(seq)
        if len(seq) > 1022:
            seq = seq[:1022]
        spaced = " ".join(list(seq))
        # ProtT5: replace U/Z/O/B with X handled in _clean
        assert self.tokenizer is not None and self.model is not None
        enc = self.tokenizer(spaced, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        if self.use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state[0]  # [seq_len, 1024]
        # remove padding and special tokens: ProtT5 adds </s> at end, we slice to len(seq)
        # last corresponds to spaced tokens + eos; take first L
        reps = last[: len(seq), :]
        return reps.float().cpu()

    def offload(self):
        if self.model is not None:
            self.model = self.model.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

def extract_af_features(
    seq: str,
    plddt: Optional[List[float]] = None,
    phi: Optional[List[float]] = None,
    psi: Optional[List[float]] = None,
    pae: Optional[List[List[float]]] = None,
) -> torch.Tensor:
    """
    Build per-residue AF feature vector [L, 7]:
    [pLDDT_norm, sin(phi), cos(phi), sin(psi), cos(psi), pae_row_mean_norm, pae_row_min_norm]
    Missing values are imputed with neutral defaults.
    """
    L = len(seq)
    device = "cpu"
    # pLDDT 0-100 -> 0-1
    if plddt is None:
        plddt_t = torch.full((L, 1), 0.7)  # neutral 70
    else:
        # pad/truncate
        arr = plddt[:L] + [70.0] * max(0, L - len(plddt))
        plddt_t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1) / 100.0

    if phi is None:
        phi_t = torch.zeros(L, 1)
        # use uniform angle 0 -> sin0=0 cos0=1 but we output sin/cos
        sin_phi = torch.zeros(L, 1)
        cos_phi = torch.ones(L, 1)
    else:
        arr = phi[:L] + [0.0] * max(0, L - len(phi))
        rad = torch.tensor(arr, dtype=torch.float32) * math.pi / 180.0
        sin_phi = torch.sin(rad).unsqueeze(1)
        cos_phi = torch.cos(rad).unsqueeze(1)

    if psi is None:
        sin_psi = torch.zeros(L, 1)
        cos_psi = torch.ones(L, 1)
    else:
        arr = psi[:L] + [0.0] * max(0, L - len(psi))
        rad = torch.tensor(arr, dtype=torch.float32) * math.pi / 180.0
        sin_psi = torch.sin(rad).unsqueeze(1)
        cos_psi = torch.cos(rad).unsqueeze(1)

    if pae is None:
        pae_mean = torch.full((L, 1), 0.5)  # normalized by 30 Ang
        pae_min = torch.full((L, 1), 0.3)
    else:
        # pae is LxL
        import numpy as np
        arr = np.array(pae, dtype=float)
        if arr.shape[0] >= L and arr.shape[1] >= L:
            arr = arr[:L, :L]
        elif arr.shape[0] < L or arr.shape[1] < L:
            # pad with 15
            padded = np.full((L, L), 15.0)
            padded[: arr.shape[0], : arr.shape[1]] = arr
            arr = padded
        row_mean = arr.mean(axis=1) / 30.0  # approx max PAE 30
        row_min = arr.min(axis=1) / 30.0
        pae_mean = torch.tensor(row_mean, dtype=torch.float32).unsqueeze(1)
        pae_min = torch.tensor(row_min, dtype=torch.float32).unsqueeze(1)

    af = torch.cat([plddt_t, sin_phi, cos_phi, sin_psi, cos_psi, pae_mean, pae_min], dim=1)  # [L,7]
    # clamp
    af = torch.nan_to_num(af, nan=0.0, posinf=1.0, neginf=0.0)
    return af

# ---- convenience pooled for sequence-level baseline ----
@torch.no_grad()
def extract_all(seq: str, device="auto", use_esm=True, use_prott5=True, af_kwargs=None):
    """High-level: returns dict with per-residue tensors (CPU)."""
    seq = _clean_sequence(seq)
    af_kwargs = af_kwargs or {}
    out = {}
    if use_esm:
        try:
            esm_ext = ESM2Extractor(device=device)
            out["esm"] = esm_ext.extract(seq)
            esm_ext.offload()
        except Exception as e:
            print(f"[extraction] ESM2 failed: {e}")
            out["esm"] = torch.zeros(len(seq), 1280)
    if use_prott5:
        try:
            pt5_ext = ProtT5Extractor(device=device)
            out["prott5"] = pt5_ext.extract(seq)
            pt5_ext.offload()
        except Exception as e:
            print(f"[extraction] ProtT5 failed (OOM fallback): {e}")
            # try smaller fallback: zero
            out["prott5"] = torch.zeros(len(seq), 1024)
    out["af"] = extract_af_features(seq, **af_kwargs)
    out["stats"] = sequence_stats(seq)
    return out
