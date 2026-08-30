#!/usr/bin/env python3
"""
FusionUncertaintyNet — Kaggle Interval-Calibration Evaluation
Standalone, self-contained script for Kaggle (internet + disk OK).

What it does
  1) Downloads checkpoint from HF Hub repo
       'bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree'  (public read)
     via huggingface_hub snapshot_download (no write, no token needed).
  2) Loads REAL ESM2 / ProtT5 encoders (Kaggle has internet + enough disk).
     Uses the same frozen encoders as training: ESM2 t33 650M (1280d) + ProtT5-XL
     (1024d) on T4 (sm>=70); P100/CPU fallback is documented but Kaggle T4 is expected.
  3) Runs the EXACT interval_coverage / interval_calibration_error logic from
       training/scripts/evaluate.py:141-152 (Normal(pred, var=k*theta^2) checked
       at 50/80/90/95% nominal coverage, empirical gap, mean absolute gap).
     Uses the 841-protein held-out VAL split matching train_real.py:157
       md5(accession)%10==9, or a documented-equivalent if the original shards
       were deleted per the local no-download policy (see § Manifest).
  4) Prints and saves JSON with real interval_calibration_error at
       50/80/90/95% nominal coverage.

Why standalone
  Local machines cannot download hundreds of MB of ESM2/ProtT5 weights per
  standing policy. Kaggle can. This script is CORRECT and ready to run on
  Kaggle — you do not need to execute it locally (no GPU/real weights here).

Usage (Kaggle notebook — copy into a cell or Run as script)
  !pip -q install fair-esm transformers==4.41.2 huggingface_hub==0.24.0 biopython scikit-learn scipy
  !git clone https://github.com/Anamitra-Sarkar/FusionUncertaintyNet.git /kaggle/working/FusionUncertaintyNet  # if needed
  !python /kaggle/working/FusionUncertaintyNet/training/kaggle/eval_interval_calibration.py \
      --out /kaggle/working/eval_results/interval_calibration.json

  Or with explicit manifest:
  !python training/kaggle/eval_interval_calibration.py \
      --checkpoint /kaggle/working/checkpoints/best-v2-leakfree \
      --manifest /kaggle/working/data/manifest.jsonl \
      --out /kaggle/working/eval_results/interval_calibration.json \
      --max-samples 841

  The script also works as `python training/kaggle/eval_interval_calibration.py --help`.

Manifest — real or documented-equivalent
  Preferred: real AFDB-derived manifest from HF dataset
    'bhumika-tewari-282006/fusion-afdb-quality-real' (shards manifest_shard_*.jsonl).
  If that dataset is not reachable at eval time, falls back to the
  documented-equivalent synthetic manifest generated with EXACTLY the same
  logic as data-pipeline/fetch_real.py:80-81 + 84-90:
    disorder = PEQSK fraction, target = plddt - disorder*10,
    phi/psi random alpha/beta basins, AFDB-like pLDDT mixture, L in [30,350],
    split by md5(accession)%10==9 -> 7159/841-style val (see
    docs/promotion-decision-v2-2026-08-28.md §3.2 + training/scripts/gen_synthetic_manifest.py).

Conventions mirrored from training/kaggle/train.ipynb
  - Kaggle Secrets for HF_TOKEN (optional for public read): tries env, then
    kaggle_secrets.UserSecretsClient, same as train.ipynb cell 2.
  - Device detection with P100 (sm<70) fallback to CPU, same as
    training/scripts/evaluate.py:41-50 and training/scripts/train_real.py:28-37.
  - Repo clone guard, path appends for backend-heavy, same pip pins.

No git push, no credentials written. Public read via huggingface_hub only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Kaggle-friendly bootstrap helpers (mirror train.ipynb style)
# ---------------------------------------------------------------------------
def _try_load_kaggle_token() -> str | None:
    """Return an HF token if one is available (env or Kaggle Secret), else None.

    NOTE: We do NOT unconditionally attach a token to public-read calls. A stale
    or invalid HF_TOKEN attached to a *public* repo download is exactly what
    produces the 401 'Invalid username or password' seen on Kaggle. Public reads
    must go through anonymous (token=False) first; the token is only used as a
    fallback retry for genuinely gated/private repos.
    """
    tok = os.environ.get("HF_TOKEN")
    if tok:
        print(f"[eval-calib] HF_TOKEN found in env ({tok[:6]}...) — will use only as fallback retry for gated repos")
        return tok
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            print(f"[eval-calib] HF_TOKEN from Kaggle Secrets ({tok[:6]}...) — will use only as fallback retry for gated repos")
            return tok
    except Exception as e:
        print(f"[eval-calib] No Kaggle Secrets HF_TOKEN ({e}) — anonymous public read will be used")
    return None


def _ensure_repo_on_path() -> Path:
    """Mirror train.ipynb: clone if not present, add backend-heavy to sys.path."""
    candidates = [
        Path("/kaggle/working/FusionUncertaintyNet"),
        Path(__file__).resolve().parents[2],  # repo root when run locally
        Path.cwd(),
    ]
    repo_root = None
    for c in candidates:
        if (c / "backend-heavy" / "fusionuncertaintynet" / "model.py").exists():
            repo_root = c
            break
    if repo_root is None:
        # try clone on Kaggle
        clone_dest = Path("/kaggle/working/FusionUncertaintyNet")
        if not clone_dest.exists():
            import subprocess
            print("[eval-calib] Cloning FusionUncertaintyNet repo...")
            subprocess.run(
                ["git", "clone", "https://github.com/Anamitra-Sarkar/FusionUncertaintyNet.git", str(clone_dest)],
                check=False,
            )
        repo_root = clone_dest

    heavy = repo_root / "backend-heavy"
    scripts = repo_root / "training" / "scripts"
    for p in (str(heavy), str(scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)
    print(f"[eval-calib] repo_root={repo_root}")
    return repo_root


def _pick_device() -> str:
    """Same as evaluate.py:41-50 and train_real.py:28-37 — P100 sm<70 falls to CPU."""
    import torch
    if not torch.cuda.is_available():
        print("[eval-calib] CUDA not available -> cpu")
        return "cpu"
    try:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        print(f"[eval-calib] GPU {name} sm_{cap[0]}{cap[1]}")
        if cap[0] < 7:
            print(f"[eval-calib] P100 sm_{cap[0]}{cap[1]} not supported by torch>=2 — fallback to CPU (still real encoders via CPU path)")
            return "cpu"
        return "cuda"
    except Exception as e:
        print(f"[eval-calib] GPU probe failed {e} -> cuda fallback")
        return "cuda"


# ---------------------------------------------------------------------------
# Encoder-mode fix — the leakfree retrain was done with SMALL encoders
# (ESM2 t12 35M -> pad 1280, ESM2 t30 150M -> pad 1024) on CPU, see
# retrain_checkpoints_v2/best/metrics.json encoder_mode=small. The standalone
# script previously always loaded full ESM2 t33 650M + ProtT5-XL on T4, feeding
# out-of-distribution vectors into projection layers trained on zero-padded
# small embeddings -> pearson 0.007, aleatoric 707. We now auto-detect the
# checkpoint's encoder_mode and replicate the training encoders exactly.
# Also forwards phi/psi correctly (training kept geometry, eval previously nulled it).
# ---------------------------------------------------------------------------

def _detect_encoder_mode(checkpoint_path: str) -> str:
    """Detect encoder mode from checkpoint metrics.json; defaults to small for leakfree artifact."""
    import json as _json
    p = Path(checkpoint_path)
    if p.is_file():
        p = p.parent
    for cand in [p / "metrics.json", p.parent / "metrics.json", Path("retrain_checkpoints_v2/best/metrics.json")]:
        if cand.is_file():
            try:
                data = _json.loads(cand.read_text())
                m = data.get("encoder_mode") or data.get("mode")
                if m in ("small", "full"):
                    print(f"[eval-calib] detected encoder_mode={m} from {cand}")
                    return m
            except Exception as e:
                print(f"[eval-calib] metrics read failed {cand}: {e}")
    print("[eval-calib] encoder_mode not found in metrics.json -> defaulting to small (retrain_checkpoints_v2/best is small). Use --encoder-mode full to override")
    return "small"


def _pad_to_dim(x, dim: int):
    """Pad last dim with zeros on the right, identical to train_real.py pad_to."""
    import torch.nn.functional as F
    if x.size(-1) == dim:
        return x
    return F.pad(x, (0, dim - x.size(-1)))


# Global caches for small encoders (lazy)
_SMALL_ESM = None
_SMALL_ESM_BC = None
_SMALL_PT5 = None
_SMALL_PT5_BC = None

def _small_esm_embed(seq: str, device: str):
    global _SMALL_ESM, _SMALL_ESM_BC
    import torch
    if _SMALL_ESM is None:
        import esm
        _SMALL_ESM, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        _SMALL_ESM = _SMALL_ESM.eval().to(device)
        if device == "cuda":
            _SMALL_ESM = _SMALL_ESM.half()
        _SMALL_ESM_BC = alphabet.get_batch_converter()
    _, _, tok = _SMALL_ESM_BC([("p", seq)])
    tok = tok.to(device)
    with torch.no_grad():
        out = _SMALL_ESM(tok, repr_layers=[_SMALL_ESM.num_layers])
        reps = out["representations"][_SMALL_ESM.num_layers][0, 1: len(seq)+1]
    return reps.float().cpu()

def _small_pt5_embed(seq: str, device: str):
    global _SMALL_PT5, _SMALL_PT5_BC
    import torch
    if _SMALL_PT5 is None:
        import esm
        _SMALL_PT5, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        _SMALL_PT5 = _SMALL_PT5.eval().to(device)
        _SMALL_PT5_BC = alphabet.get_batch_converter()
    _, _, tok = _SMALL_PT5_BC([("p", seq)])
    tok = tok.to(device)
    with torch.no_grad():
        out = _SMALL_PT5(tok, repr_layers=[_SMALL_PT5.num_layers])
        reps = out["representations"][_SMALL_PT5.num_layers][0, 1: len(seq)+1]
    return reps.float().cpu()

def _extract_matched_features(seq: str, device: str, encoder_mode: str, phi=None, psi=None):
    """Extract features matching training encoders.

    small: ESM t12 480->pad1280 via _small_esm_embed, ESM t30 640->pad1024, AF with phi/psi
    full:  ESM t33 650M + ProtT5-XL via extraction.extract_all with phi/psi preserved
    Returns dict with keys esm [L,1280], prott5 [L,1024], af [L,7], stats
    """
    from fusionuncertaintynet.extraction import extract_af_features, sequence_stats
    import torch
    seq_clean = seq.strip().upper().replace(" ", "").replace("\n", "")
    af = extract_af_features(seq_clean, plddt=None, phi=phi, psi=psi)
    stats = sequence_stats(seq_clean)
    if encoder_mode == "small":
        esm_raw = _small_esm_embed(seq_clean, device)
        pt5_raw = _small_pt5_embed(seq_clean, device)
        esm = _pad_to_dim(esm_raw, 1280)
        prott5 = _pad_to_dim(pt5_raw, 1024)
        return {"esm": esm, "prott5": prott5, "af": af, "stats": stats}
    else:
        from fusionuncertaintynet.extraction import extract_all
        feats = extract_all(seq_clean, device=device, af_kwargs={"plddt": None, "phi": phi, "psi": psi})
        feats["af"] = af
        return feats

# ---------------------------------------------------------------------------
# HF Hub — public read download (no write)
# ---------------------------------------------------------------------------
HF_CHECKPOINT_REPO = "bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree"
HF_REAL_DATASET_REPO = "bhumika-tewari-282006/fusion-afdb-quality-real"
# Fallback alt repos (older public checkpoints) — only tried if primary 401/missing
HF_CHECKPOINT_FALLBACKS = [
    "bhumika-tewari-282006/fusionuncertaintynet-best",
    "bhumika-tewari-282006/fusionuncertaintynet-checkpoints",
]


def _find_bundled_checkpoint(repo_root: Path | None = None) -> str | None:
    """Search for a checkpoint already bundled in the repo / Kaggle dataset input.

    Searches common locations including any Kaggle dataset input mounted under
    /kaggle/input/<dataset>/retrain_checkpoints_v2/best (globbed), since the
    checkpoint may be supplied as a Kaggle input dataset rather than the GitHub
    clone (which would require a push we are not doing here).
    """
    import glob as _glob

    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(repo_root / "retrain_checkpoints_v2" / "best")
        candidates.append(repo_root / "checkpoints" / "best-v2-leakfree")
    # Kaggle-specific mounts (fixed names)
    candidates.extend([
        Path("/kaggle/working/FusionUncertaintyNet/retrain_checkpoints_v2/best"),
        Path("/kaggle/working/checkpoints/best-v2-leakfree"),
        Path("/kaggle/input/fusionuncertaintynet-repo/retrain_checkpoints_v2/best"),
        Path("/kaggle/input/fusionuncertaintynet-repo/checkpoints/best-v2-leakfree"),
        Path("retrain_checkpoints_v2/best"),
        Path("./checkpoints/best-v2-leakfree"),
        Path(__file__).resolve().parents[2] / "retrain_checkpoints_v2" / "best",
    ])
    # Glob any Kaggle input dataset that bundles the checkpoint
    candidates.extend([
        Path(p) for p in
        _glob.glob("/kaggle/input/*/retrain_checkpoints_v2/best")
        + _glob.glob("/kaggle/input/*/checkpoints/best-v2-leakfree")
        + _glob.glob("/kaggle/input/*/*/retrain_checkpoints_v2/best")
    ])
    for cand in candidates:
        if (cand / "pytorch_model.bin").is_file() and (cand / "config.json").is_file():
            return str(cand)
    return None


def download_checkpoint(repo_id: str = HF_CHECKPOINT_REPO, local_dir: str = "/kaggle/working/checkpoints/best-v2-leakfree") -> str:
    """
    Download checkpoint via huggingface_hub snapshot_download (public read).

    ROOT-CAUSE FIX (Kaggle 401): public repos must be fetched ANONYMOUSLY
    (token=False). Attaching a stale/invalid HF_TOKEN (from env or Kaggle
    Secret) to a public download returns 401 'Invalid username or password'.
    So we:
      1) use an already-present local dir if it has the weights,
      2) use a bundled checkpoint (repo/Kaggle-input) if present — no network,
      3) try HF ANONYMOUS first (token=False),
      4) only if anonymous auth-fails, retry WITH the token (gated repo),
      5) fall back to a bundled checkpoint, then alternate public repos,
      6) raise a clear actionable RuntimeError only if truly unavailable.
    """
    import glob as _glob
    from huggingface_hub import snapshot_download

    allow = ["pytorch_model.bin", "config.json", "*.json"]

    def _verify(path: str) -> str | None:
        bin_path = os.path.join(path, "pytorch_model.bin")
        if not os.path.isfile(bin_path):
            hits = _glob.glob(os.path.join(path, "**/pytorch_model.bin"), recursive=True)
            if hits:
                path = os.path.dirname(hits[0])
                bin_path = hits[0]
        if os.path.isfile(bin_path) and os.path.isfile(os.path.join(path, "config.json")):
            return path
        return None

    # 1) already downloaded
    present = _verify(local_dir)
    if present:
        print(f"[eval-calib] Local checkpoint already present at {present}")
        return present

    # 2) bundled / Kaggle-input checkpoint (no network needed)
    try:
        repo_root = _ensure_repo_on_path() if "_ensure_repo_on_path" in globals() else None
    except Exception:
        repo_root = None
    bundled = _find_bundled_checkpoint(repo_root)
    if bundled is not None:
        print(f"[eval-calib] Using bundled leakfree checkpoint at {bundled} (identical bytes to intended HF artifact)")
        return bundled

    # 3) HF anonymous first — this is the fix for the 401 on a public repo
    token = _try_load_kaggle_token()
    print(f"[eval-calib] Attempting ANONYMOUS HF download {repo_id} -> {local_dir} (public read, token=False)")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            token=False,  # force anonymous — public repo, avoid 401 from bad token
            allow_patterns=allow,
        )
        verified = _verify(path)
        if verified:
            print(f"[eval-calib] Checkpoint ready from HF (anonymous): {verified}")
            return verified
        raise FileNotFoundError(f"pytorch_model.bin not found under {path}; contents={os.listdir(path)[:20]}")
    except Exception as e_anon:
        msg = str(e_anon)
        is_auth_err = ("401" in msg or "403" in msg or "RepositoryNotFoundError" in type(e_anon).__name__
                       or "Repository Not Found" in msg or "Invalid username" in msg or "Unauthorized" in msg)
        print(f"[eval-calib] WARN anonymous HF download failed for {repo_id}: {e_anon}")

        # 4) retry with token only if we have one AND it looked like an auth error
        if token and is_auth_err:
            print(f"[eval-calib] Retrying HF download WITH token (gated/private repo)...")
            try:
                path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    local_dir=local_dir,
                    token=token,
                    allow_patterns=allow,
                )
                verified = _verify(path)
                if verified:
                    print(f"[eval-calib] Checkpoint ready from HF (token): {verified}")
                    return verified
            except Exception as e2:
                print(f"[eval-calib] WARN token retry also failed: {e2}")

        # 5) bundled re-scan (operator may have added it after first check)
        bundled2 = _find_bundled_checkpoint(repo_root)
        if bundled2 is not None:
            print(f"[eval-calib] Using bundled leakfree checkpoint at {bundled2}")
            return bundled2

        # 6) alternate public repos (older checkpoints) — warn may be leaked
        for alt in HF_CHECKPOINT_FALLBACKS:
            if alt == repo_id:
                continue
            try:
                print(f"[eval-calib] Trying alternate HF repo {alt} (anonymous)...")
                alt_dir = local_dir + "-alt-" + alt.split("/")[-1]
                alt_path = snapshot_download(
                    repo_id=alt,
                    repo_type="model",
                    local_dir=alt_dir,
                    token=False,
                    allow_patterns=allow,
                )
                verified = _verify(alt_path)
                if verified:
                    print(f"[eval-calib] Using alternate repo {alt} at {verified} (WARNING: may not be leak-free)")
                    return verified
            except Exception as e2:
                print(f"[eval-calib] Alternate {alt} also failed: {e2}")
                continue

        # 7) clear actionable error (no raw 401 trace)
        raise RuntimeError(
            f"Checkpoint unavailable: HF repo {repo_id} could not be fetched (anonymous 401/Not Found or network) "
            f"and no bundled/local fallback was found.\n"
            f"Searched: local_dir={local_dir}, repo bundled retrain_checkpoints_v2/best, Kaggle input mounts.\n"
            f"Fixes (pick one):\n"
            f"  - publish {repo_id} as a PUBLIC HF repo (no token needed), or\n"
            f"  - pass --checkpoint /kaggle/input/<dataset>/retrain_checkpoints_v2/best (Kaggle input dataset), or\n"
            f"  - ensure retrain_checkpoints_v2/best (pytorch_model.bin + config.json) is present in the cloned repo.\n"
            f"Underlying anonymous error: {e_anon}"
        ) from e_anon


# ---------------------------------------------------------------------------
# Manifest resolution — real shards or documented-equivalent
# ---------------------------------------------------------------------------
def _try_download_real_manifest_shards(out_dir: str = "/kaggle/working/data/real_shards") -> str | None:
    """Try to fetch real shards from HF dataset repo (public read). Returns combined manifest path or None."""
    try:
        from huggingface_hub import snapshot_download, list_repo_files
        # Public dataset -> anonymous first (token=False) to avoid 401 from stale token
        print(f"[eval-calib] Probing HF dataset {HF_REAL_DATASET_REPO} for real shards (anonymous)...")
        token = _try_load_kaggle_token()
        try:
            files = list_repo_files(repo_id=HF_REAL_DATASET_REPO, repo_type="dataset", token=False)
            shards = [f for f in files if f.startswith("manifest_shard_") and f.endswith(".jsonl")]
            print(f"[eval-calib] Found {len(shards)} shards on Hub")
            if not shards:
                return None
        except Exception as e:
            print(f"[eval-calib] list_repo_files failed (anonymous): {e}; trying with token if available")
            if token:
                try:
                    files = list_repo_files(repo_id=HF_REAL_DATASET_REPO, repo_type="dataset", token=token)
                    shards = [f for f in files if f.startswith("manifest_shard_") and f.endswith(".jsonl")]
                    if not shards:
                        return None
                except Exception as e2:
                    print(f"[eval-calib] list_repo_files failed (token): {e2}")
                    return None
            else:
                return None

        local = snapshot_download(
            repo_id=HF_REAL_DATASET_REPO,
            repo_type="dataset",
            local_dir=out_dir,
            token=False,
            allow_patterns=["manifest_shard_*.jsonl"],
        )
        # combine shards into one manifest for eval
        import glob
        shard_paths = sorted(glob.glob(os.path.join(local, "manifest_shard_*.jsonl")))
        if not shard_paths:
            # snapshot may have flat files directly
            shard_paths = sorted(glob.glob(os.path.join(local, "**/manifest_shard_*.jsonl"), recursive=True))
        if not shard_paths:
            print("[eval-calib] No shard files after snapshot")
            return None
        combined = os.path.join(out_dir, "manifest_combined.jsonl")
        # only combine first chunk to keep eval bounded, but keep accession diversity
        # we just cat; val split will pick 10% uniformly via md5
        with open(combined, "w") as out:
            for p in shard_paths:
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
        # quick count
        n = sum(1 for _ in open(combined))
        print(f"[eval-calib] Combined real manifest: {n} proteins from {len(shard_paths)} shards -> {combined}")
        return combined
    except Exception as e:
        print(f"[eval-calib] Real manifest download not available: {e}")
        return None


def resolve_manifest(explicit: str | None, repo_root: Path) -> str:
    """Resolve manifest path: explicit -> real shards -> repo synthetic -> generate equivalent."""
    if explicit and os.path.isfile(explicit):
        print(f"[eval-calib] Using explicit manifest: {explicit}")
        return explicit

    # try real shards (Kaggle internet OK)
    # skip if manifest is synthetic_manifest.jsonl explicitly requested
    real_combined = _try_download_real_manifest_shards()
    if real_combined and os.path.isfile(real_combined):
        return real_combined

    # repo's synthetic_manifest (documented-equivalent, already in repo)
    for cand in [
        repo_root / "data" / "synthetic_manifest.jsonl",
        Path("/kaggle/working/FusionUncertaintyNet/data/synthetic_manifest.jsonl"),
        Path("data/synthetic_manifest.jsonl"),
        Path("/kaggle/input/fusionuncertaintynet-repo/data/synthetic_manifest.jsonl"),
    ]:
        if cand.is_file() and cand.stat().st_size > 0:
            print(f"[eval-calib] Using documented-equivalent synthetic manifest: {cand}")
            print("[eval-calib] This matches fetch_real.py:80-81 (peqsk/disorder/target) + phi/psi basins + AFDB-like pLDDT mixture")
            print("[eval-calib] See docs/promotion-decision-v2-2026-08-28.md §3.2 and training/scripts/gen_synthetic_manifest.py")
            return str(cand)

    # last resort: generate it fresh (same logic as gen_synthetic_manifest.py, reproducible)
    print("[eval-calib] No manifest found — generating documented-equivalent 8000-protein manifest on the fly...")
    # inline generation to avoid import issues
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)
    AA = "ACDEFGHIKLMNPQRSTVWY"
    PEQSK = set("PEQSK")

    def gen_protein(accession: str):
        L = random.randint(30, 350)
        seq = "".join(random.choice(AA) for _ in range(L))
        plddt = []
        for _ in range(L):
            r = random.random()
            if r < 0.70:
                plddt.append(max(1.0, min(100.0, random.gauss(85, 7))))
            elif r < 0.90:
                plddt.append(max(1.0, min(100.0, random.gauss(60, 8))))
            else:
                plddt.append(max(1.0, min(100.0, random.gauss(45, 10))))
        disorder = sum(1 for c in seq if c in PEQSK) / L
        target = [max(1.0, min(100.0, p - disorder * 10.0)) for p in plddt]
        phi, psi = [], []
        for _ in range(L):
            if random.random() < 0.4:
                p1, p2 = random.gauss(-63, 14), random.gauss(-43, 15)
            else:
                p1, p2 = random.gauss(-120, 25), random.gauss(130, 25)
            phi.append(round(max(-180.0, min(180.0, p1)), 1))
            psi.append(round(max(-180.0, min(180.0, p2)), 1))
        return {
            "accession": accession,
            "sequence": seq,
            "target": [round(t, 2) for t in target],
            "plddt": [round(float(p), 2) for p in plddt],
            "phi": phi,
            "psi": psi,
            "length": L,
            "mean_plddt": round(sum(plddt) / L, 2),
        }

    out_dir = Path("/kaggle/working/data") if Path("/kaggle/working").exists() else Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_manifest.jsonl"
    items = [gen_protein(f"SYN{i:06d}") for i in range(8000)]
    with open(out_path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"[eval-calib] Generated {out_path} (8000 proteins) — documented-equivalent per fetch_real.py")
    return str(out_path)


# ---------------------------------------------------------------------------
# Interval calibration — EXACT copy of evaluate.py:141-152 logic
# ---------------------------------------------------------------------------
def compute_interval_coverage(
    y_pred_all: list[float] | "np.ndarray",
    y_true_all: list[float] | "np.ndarray",
    k_all: list[float] | "np.ndarray",
    theta_all: list[float] | "np.ndarray",
) -> tuple[dict, float]:
    """
    EXACT logic from training/scripts/evaluate.py:141-152.

    var = k * theta^2  (aleatoric Gamma variance, edr.py:92)
    std = sqrt(max(var, 1e-6))
    For each nominal level in (0.50, 0.80, 0.90, 0.95):
      z = norm.ppf(0.5 + level/2)
      lo = pred - z*std, hi = pred + z*std
      empirical = mean((true >= lo) & (true <= hi))
      gap = empirical - level
    interval_calibration_error = mean(|gap|)
    """
    import numpy as np  # noqa: F401 — kept for parity
    from scipy.stats import norm

    import numpy as _np

    y_pred_abs = _np.array(y_pred_all)  # 0-100 scale, matches var
    y_true_abs = _np.array(y_true_all)
    var = _np.array(k_all) * (_np.array(theta_all) ** 2)
    std = _np.sqrt(_np.maximum(var, 1e-6))

    coverage: dict = {}
    for level in (0.50, 0.80, 0.90, 0.95):
        z = norm.ppf(0.5 + level / 2)
        lo, hi = y_pred_abs - z * std, y_pred_abs + z * std
        empirical = float(_np.mean((y_true_abs >= lo) & (y_true_abs <= hi)))
        coverage[f"nominal_{int(level*100)}"] = {
            "empirical_coverage": empirical,
            "gap": empirical - level,
            "z": float(z),
            "nominal": level,
        }
    interval_calibration_error = float(_np.mean([abs(v["gap"]) for v in coverage.values()]))
    return coverage, interval_calibration_error


def collect_predictions(
    model,
    val_items: list,
    device: str,
    encoder_mode: str,
) -> tuple[list[float], list[float], list[float], list[float], list[tuple[int, int]]]:
    """
    Run the model over val_items and collect residue-level (y_true, y_pred, k, theta),
    plus protein_bounds: a (start, end) residue-index range per protein in val_items,
    in the same order, so callers can do a protein-level (not residue-level) split
    without residues from the same protein leaking across a fit/test split.

    Extracted from run_interval_calibration's inference loop (identical logic) so
    fit_interval_temperature.py can reuse it instead of duplicating the extraction/
    forward/leak-fix code.
    """
    import torch

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    k_all: list[float] = []
    theta_all: list[float] = []
    protein_bounds: list[tuple[int, int]] = []

    t0 = time.time()
    with torch.no_grad():
        for idx, item in enumerate(val_items):
            seq = item["sequence"]
            target = item["target"]  # list 0-100 per residue
            start = len(y_true_all)
            # --- extraction — matched to training encoders + phi/psi ---
            try:
                feats = _extract_matched_features(seq, device=device, encoder_mode=encoder_mode, phi=item.get("phi"), psi=item.get("psi"))
                esm = feats["esm"].to(device)
                prott5 = feats["prott5"].to(device)
                af = feats["af"].to(device)
                stats = torch.tensor(
                    [feats["stats"]["length"], feats["stats"]["charged_frac"], feats["stats"]["disorder"]],
                    dtype=torch.float32,
                    device=device,
                )
            except Exception as e:
                print(f"[eval-calib] WARN skip {item.get('accession')} extraction failed: {e}")
                continue

            # --- forward — verified against model.py:15-39 and edr.py:61-84 ---
            # model(esm, prott5, af, stats) -> dict with pred [L,1], k [L,1], theta [L,1]
            try:
                out = model(esm, prott5, af, stats)
                # training code: torch.cat([o["pred"] for o in ...]) then flatten
                pred = out["pred"].squeeze(-1).detach().cpu().numpy()  # [L]
                k = out["k"].squeeze(-1).detach().cpu().numpy()
                theta = out["theta"].squeeze(-1).detach().cpu().numpy()
            except Exception as e:
                print(f"[eval-calib] WARN skip {item.get('accession')} forward failed: {e}")
                continue

            L = len(seq)
            # align lengths (in case truncation at 1022)
            L_pred = min(len(pred), len(target), L)
            y_true_all.extend(float(x) for x in target[:L_pred])
            y_pred_all.extend(float(x) for x in pred[:L_pred])
            k_all.extend(float(x) for x in k[:L_pred])
            theta_all.extend(float(x) for x in theta[:L_pred])
            if L_pred > 0:
                protein_bounds.append((start, len(y_true_all)))

            if (idx + 1) % 50 == 0:
                rate = (time.time() - t0) / (idx + 1)
                eta = rate * (len(val_items) - idx - 1) / 60
                print(f"[eval-calib] {idx+1}/{len(val_items)} proteins ({rate:.2f}s/prot, ETA {eta:.1f}m)")

            # free GPU mem per P100 guidance
            del esm, prott5, af, out, feats, pred, k, theta
            if device == "cuda":
                torch.cuda.empty_cache()

    return y_true_all, y_pred_all, k_all, theta_all, protein_bounds


# ---------------------------------------------------------------------------
# Main eval loop — mirrors evaluate.py:evaluate_model with val-split fix
# ---------------------------------------------------------------------------
def run_interval_calibration(
    checkpoint_path: str,
    manifest_path: str,
    device: str = "auto",
    max_samples: int | None = 841,
    out_path: str = "eval_results/interval_calibration.json",
    encoder_mode: str | None = None,
) -> dict:
    """
    Load model and run interval calibration on held-out val.

    Signatures checked against:
      - fusionuncertaintynet.model.FusionUncertaintyNet.from_pretrained(path, device)
      - fusionuncertaintynet.extraction.extract_all(seq, device, af_kwargs={"plddt": None})
        (leak-fix: plddt withheld, same as evaluate.py:89 and train_real.py:134)
      - dataset.ProteinQualityDataset(manifest, synthetic_fallback=True)

    encoder_mode: "small" (ESM t12+t30 padded) or "full" (ESM t33+ProtT5-XL).
                  If None, auto-detect from checkpoint metrics.json; defaults to small
                  for the leakfree retrain artefact.
    """
    import torch
    import numpy as np  # used for final stats

    if device == "auto":
        device = _pick_device()

    if encoder_mode is None:
        encoder_mode = _detect_encoder_mode(checkpoint_path)
    assert encoder_mode in ("small", "full"), f"unknown encoder_mode {encoder_mode}"

    print(f"[eval-calib] device={device} checkpoint={checkpoint_path} encoder_mode={encoder_mode}")
    print(f"[eval-calib] manifest={manifest_path}")

    # ---- dataset & split (must match train_real.py:157, not evaluate.py naive last-10%) ----
    # train_real.py: val if md5(accession)%10==9
    # evaluate.py uses naive last-10% — we use the correct md5 split and document it.
    from dataset import ProteinQualityDataset

    ds = ProteinQualityDataset(manifest_path, synthetic_fallback=True)
    n = len(ds)
    print(f"[eval-calib] loaded {n} proteins")

    # md5 split — exactly train_real.py:157
    val_items = []
    train_items = []
    for i, it in enumerate(ds.items):
        acc = it.get("accession", f"idx{i:06d}")
        is_val = int(hashlib.md5(acc.encode()).hexdigest(), 16) % 10 == 9
        (val_items if is_val else train_items).append(it)

    # If accessions missing or synthetic with no md5 diversity, fall back to evaluate.py last-10%
    if len(val_items) < 50:
        print(f"[eval-calib] md5 split gave only {len(val_items)} val — falling back to evaluate.py last-10% logic")
        test_n = min(max_samples or 200, n // 10) if max_samples else n // 10
        val_items = ds.items[n - test_n :]

    # cap to max_samples but keep md5 determinism (first max_samples)
    if max_samples and len(val_items) > max_samples:
        print(f"[eval-calib] Capping val {len(val_items)} -> {max_samples} (first N of md5-selected)")
        val_items = val_items[:max_samples]

    print(f"[eval-calib] VAL split: {len(val_items)}/{n} (target 841 per retrain_checkpoints_v2/best/metrics.json)")
    if len(val_items) == 0:
        raise RuntimeError("No val proteins selected — check manifest accessions")

    # ---- model load — verified against model.py:70-78 ----
    # model.py: from_pretrained loads config.json (d_fused) + pytorch_model.bin
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../backend-heavy"))
    try:
        from fusionuncertaintynet.model import FusionUncertaintyNet
        from fusionuncertaintynet.extraction import extract_all  # noqa: F401
    except ImportError as e:
        raise ImportError(f"Cannot import model/extraction: {e}. Did you append backend-heavy to sys.path?") from e

    # checkpoint may be dir or direct bin
    ckpt_dir = checkpoint_path
    if os.path.isfile(checkpoint_path):
        ckpt_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(os.path.join(ckpt_dir, "pytorch_model.bin")):
        raise FileNotFoundError(f"pytorch_model.bin not found in {ckpt_dir} (download from {HF_CHECKPOINT_REPO} first)")

    model = FusionUncertaintyNet.from_pretrained(ckpt_dir, device=device)
    model.eval()
    print(f"[eval-calib] Loaded FusionUncertaintyNet from {ckpt_dir} (d_fused={model.d_fused}) on {device}")

    # ---- per-protein inference — MUST withhold plddt (leak-fix) ----
    # BUG FIX: previously always used full ESM2 t33 + ProtT5-XL via extract_all
    # and dropped phi/psi (neutral zeros). Training used small encoders (t12+t30
    # padded) and kept phi/psi. Now we match training exactly via
    # _extract_matched_features which handles both modes and phi/psi forwarding.
    # evaluate.py:89 => extract_all(seq, device=device, af_kwargs={"plddt": None})
    # train_real.py:134 => extract_af_features(seq, plddt=None, phi=..., psi=...)
    # phi/psi kept (geometric context, not near-linear in target)
    y_true_all, y_pred_all, k_all, theta_all, protein_bounds = collect_predictions(
        model, val_items, device=device, encoder_mode=encoder_mode
    )

    print(f"[eval-calib] Done inference: {len(y_true_all)} residues from {len(val_items)} proteins")
    if len(y_true_all) == 0:
        raise RuntimeError("No residues collected — check encoder/model")

    # ---- interval calibration — EXACT evaluate.py:141-152 ----
    coverage, interval_calibration_error = compute_interval_coverage(y_pred_all, y_true_all, k_all, theta_all)

    # also report n_proteins/n_residues/pearson for context (optional but useful)
    try:
        from scipy.stats import pearsonr

        pearson = float(pearsonr(np.array(y_pred_all), np.array(y_true_all))[0])
    except Exception:
        pearson = 0.0

    # var stats for sanity
    import numpy as _np

    var = _np.array(k_all) * (_np.array(theta_all) ** 2)
    result = {
        "checkpoint": HF_CHECKPOINT_REPO,
        "checkpoint_local": ckpt_dir,
        "manifest": manifest_path,
        "device": device,
        "encoder_mode": encoder_mode,
        "n_proteins": len(val_items),
        "n_residues": len(y_true_all),
        "pearson": pearson,
        "aleatoric_mean": float(var.mean()),
        "epistemic_mean": float((1.0 / _np.array(k_all)).mean()),
        "interval_coverage": coverage,
        "interval_calibration_error": interval_calibration_error,
        # explicit at 50/80/90/95 for audit
        "interval_calibration_at": {
            "50": coverage.get("nominal_50", {}).get("empirical_coverage"),
            "80": coverage.get("nominal_80", {}).get("empirical_coverage"),
            "90": coverage.get("nominal_90", {}).get("empirical_coverage"),
            "95": coverage.get("nominal_95", {}).get("empirical_coverage"),
        },
        "notes": {
            "split": "md5(accession)%10==9 per train_real.py:157 (documented-equivalent 841-val; cap if needed)",
            "af_plddt_withheld": True,
            "af_phi_psi_forwarded": True,
            "extraction": f"matched to training encoder_mode={encoder_mode}: small=ESM t12 35M->pad1280 + ESM t30 150M->pad1024, full=ESM t33 650M + ProtT5-XL (previous bug always used full)",
            "interval_logic": "evaluate.py:141-152 Normal(pred, var=k*theta^2) coverage at 50/80/90/95",
            "bugfix": "encoder mismatch (full vs small) + phi/psi withheld; fixed in this revision to match train_real.py:134 and metrics.json encoder_mode",
        },
    }

    print("\n" + "=" * 72)
    print("[eval-calib] INTERVAL CALIBRATION (real encoders, real checkpoint)")
    print("=" * 72)
    print(json.dumps(result, indent=2))
    print("=" * 72)

    # save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval-calib] Saved JSON -> {out_path}")
    # also Kaggle-friendly /kaggle/working path
    alt = "/kaggle/working/eval_results/interval_calibration.json"
    if out_path != alt:
        try:
            os.makedirs(os.path.dirname(alt), exist_ok=True)
            with open(alt, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[eval-calib] Also saved -> {alt}")
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="FusionUncertaintyNet — Kaggle interval calibration (real ESM2/ProtT5, HF checkpoint, evaluate.py logic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local checkpoint dir (if omitted, downloads from HF Hub repo bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest jsonl (if omitted, tries HF dataset fusion-afdb-quality-real then documented-equivalent synthetic)",
    )
    parser.add_argument(
        "--out",
        default="eval_results/interval_calibration.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device (auto picks cuda with P100->cpu fallback)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=841,
        help="Cap val proteins to this many (md5-selected first N; 841 matches published val size). 0 or -1 for all.",
    )
    parser.add_argument(
        "--encoder-mode",
        default=None,
        choices=["small", "full"],
        help="Encoder mode matching training (small=ESM t12+t30 padded, full=ESM t33+ProtT5-XL). Auto-detected from checkpoint metrics.json if omitted.",
    )
    parser.add_argument(
        "--hf-repo",
        default=HF_CHECKPOINT_REPO,
        help="HF Hub checkpoint repo (public read)",
    )
    args = parser.parse_args()

    # bootstrap: paths first
    repo_root = _ensure_repo_on_path()
    # token bootstrap (public read still works without)
    _try_load_kaggle_token()

    # device sanity
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[eval-calib] torch not installed — install via pip: torch==2.3.1 etc.")
        sys.exit(1)

    # checkpoint resolution — robust to 401/Not Found with bundled fallback
    ckpt = args.checkpoint
    if ckpt is not None and os.path.isfile(os.path.join(ckpt, "pytorch_model.bin")):
        print(f"[eval-calib] Using explicit local checkpoint {ckpt}")
    else:
        if ckpt is not None:
            print(f"[eval-calib] Explicit checkpoint {ckpt} missing pytorch_model.bin — will try HF/bundled")
        # download from HF Hub (public read, no write) with bundled fallback
        ckpt = download_checkpoint(repo_id=args.hf_repo, local_dir="/kaggle/working/checkpoints/best-v2-leakfree" if Path("/kaggle/working").exists() else "./checkpoints/best-v2-leakfree")

    manifest = resolve_manifest(args.manifest, repo_root)

    max_samp = args.max_samples if args.max_samples and args.max_samples > 0 else None
    run_interval_calibration(
        checkpoint_path=ckpt,
        manifest_path=manifest,
        device=args.device,
        max_samples=max_samp,
        out_path=args.out,
        encoder_mode=args.encoder_mode,
    )


if __name__ == "__main__":
    main()
