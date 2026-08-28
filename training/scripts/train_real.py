"""
REAL training pipeline for FusionUncertaintyNet — no dummy vectors anywhere.

Dual-path auto-selection by hardware:
  GPU (sm>=70, e.g. T4): paper-faithful frozen encoders
      ESM2 t33 650M (1280d, fp16) + ProtT5-XL (1024d, fp16)
  CPU / P100 (sm_60 unsupported by torch>=2): still REAL pretrained PLMs, smaller
      ESM2 t12 35M (480d -> zero-pad 1280) + ESM2 t30 150M (640d -> zero-pad 1024)

Data: manifest from AlphaFold DB (REAL sequences + REAL per-residue pLDDT targets/features).
Memory-bounded supervision: per protein keep K=64 residue slices (fp16); full-length
extraction runs every pass, storage stays ~GBs so 10s of epochs are cheap.
Split: md5(accession)%10==9 -> val (no leakage).
Outputs: epoch checkpoints + best (val Pearson) -> HF Hub; metrics.json.
"""
import os, sys, json, time, hashlib, argparse, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "../../backend-heavy"))
sys.path.append(os.path.dirname(__file__))
from fusionuncertaintynet.model import FusionUncertaintyNet          # noqa: E402
from fusionuncertaintynet.losses import composite_loss, pearson_corr  # noqa: E402
from fusionuncertaintynet.extraction import sequence_stats, extract_af_features, AA_ORDER  # noqa: E402

# ---------------------------------------------------------------- hardware
def pick_device():
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 7:
                return "cuda", "full"
            print(f"[real-train] GPU sm_{cap[0]}{cap[1]} unsupported by torch -> CPU small-PLM path")
        except Exception:
            pass
    return "cpu", "small"

def pad_to(x, dim):
    if x.size(-1) == dim:
        return x
    return F.pad(x, (0, dim - x.size(-1)))

# ---------------------------------------------------------------- encoders
class RealEncoders:
    """Lazy real encoders for the chosen path."""
    def __init__(self, mode, device):
        self.mode, self.device = mode, device
        self.esm = None; self.esm_bc = None
        self.pt5 = None; self.pt5_tok = None

    def _load_esm(self):
        if self.esm is not None:
            return
        import esm
        if self.mode == "full":
            self.esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            self.esm, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.esm.eval().to(self.device)
        if self.device == "cuda":
            self.esm = self.esm.half()
        self.esm_bc = alphabet.get_batch_converter()

    @torch.no_grad()
    def esm_embed(self, seq):
        self._load_esm()
        _, _, tok = self.esm_bc([("p", seq)])
        tok = tok.to(self.device)
        out = self.esm(tok, repr_layers=[self.esm.num_layers])
        reps = out["representations"][self.esm.num_layers][0, 1:len(seq)+1]
        return reps.float().cpu()

    def _load_pt5(self):
        if self.pt5 is not None:
            return
        from transformers import T5Tokenizer, T5EncoderModel
        name = "Rostlab/prot_t5_xl_uniref50" if self.mode == "full" else None
        if name is None:   # CPU path uses real ESM2-t30 for second branch
            import esm
            self.pt5, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            self.pt5.eval().to(self.device)
            self.pt5_bc = alphabet.get_batch_converter()
            self.pt5_is_esm = True
            return
        self.pt5_tok = T5Tokenizer.from_pretrained(name, do_lower_case=False)
        self.pt5 = T5EncoderModel.from_pretrained(name).eval().to(self.device)
        if self.device == "cuda":
            self.pt5 = self.pt5.half()
        self.pt5_is_esm = False

    @torch.no_grad()
    def pt5_embed(self, seq):
        self._load_pt5()
        if getattr(self, "pt5_is_esm", False):
            _, _, tok = self.pt5_bc([("p", seq)])
            tok = tok.to(self.device)
            out = self.pt5(tok, repr_layers=[self.pt5.num_layers])
            return out["representations"][self.pt5.num_layers][0, 1:len(seq)+1].float().cpu()
        spaced = " ".join(list(seq))
        enc = self.pt5_tok(spaced, return_tensors="pt")
        out = self.pt5(input_ids=enc["input_ids"].to(self.device),
                       attention_mask=enc["attention_mask"].to(self.device))
        return out.last_hidden_state[0, :len(seq)].float().cpu()

# ---------------------------------------------------------------- feature build
K_RES = 64   # residue slices stored per protein

def build_split(items, encoders, device, max_len, limit, log_every=250):
    """Extract REAL features; store K residue slices per protein (fp16)."""
    train, val = [], []
    t0 = time.time()
    for n, it in enumerate(items[:limit]):
        seq = it["sequence"][:max_len]
        L = len(seq)
        if L < 30:
            continue
        try:
            esm = encoders.esm_embed(seq)                    # [L, D_esm]
            pt5 = encoders.pt5_embed(seq)                    # [L, D_pt5]
        except Exception as e:
            print(f"[skip] {it['accession']}: {e}", flush=True)
            continue
        # [LEAK-FIX] `target` in the manifest is derived as plddt - disorder*10
        # (see data-pipeline/fetch_real.py). Feeding the source pLDDT back in as
        # an AF input feature made the regression trivially solvable by the AF
        # branch alone (naive raw-pLDDT baseline measured r=0.9995 on held-out
        # AFDB proteins vs. this model's own reported val Pearson of 0.9987 --
        # the "fusion" contributed nothing and the model lost to the baseline
        # it was supposed to beat). Withhold pLDDT from the AF feature branch
        # during training so ESM2/ProtT5 must actually carry the prediction;
        # phi/psi geometric context is kept since it isn't a near-linear
        # transform of the target.
        af = extract_af_features(seq, plddt=None,
                                 phi=it.get("phi"), psi=it.get("psi"))  # [L,7]
        tgt = torch.tensor(it["target"][:L], dtype=torch.float32)
        phi = torch.tensor(it.get("phi", [0]*L)[:L], dtype=torch.float32)
        psi = torch.tensor(it.get("psi", [0]*L)[:L], dtype=torch.float32)

        g = torch.Generator().manual_seed(n)
        K = min(K_RES, L)
        idx = torch.randperm(L, generator=g)[:K]

        rec = {
            "accession": it["accession"],
            "esm": pad_to(esm[idx], 1280).half(),
            "pt5": pad_to(pt5[idx], 1024).half(),
            "af":  af[idx].half(),
            "y":   tgt[idx],
            "phi": phi[idx], "psi": psi[idx],
            "stats": torch.tensor([
                sequence_stats(seq)["length"],
                sequence_stats(seq)["charged_frac"],
                sequence_stats(seq)["disorder"],
            ], dtype=torch.float16),
        }
        bucket = val if int(hashlib.md5(it["accession"].encode()).hexdigest(), 16) % 10 == 9 else train
        bucket.append(rec)
        if (n + 1) % log_every == 0:
            r = (time.time() - t0) / (n + 1)
            eta = r * (limit - n - 1) / 60
            print(f"[extract] {n+1}/{limit} ({r:.2f}s/item, ETA {eta:.0f}m) "
                  f"train={len(train)} val={len(val)}", flush=True)
    return train, val

# ---------------------------------------------------------------- training
def run_epoch(model, data, device, opt=None, bs=512):
    torch.manual_seed(0)
    model.train(opt is not None)
    tot, nb, preds, tgts = 0.0, 0, [], []
    order = torch.randperm(len(data))
    for s in range(0, len(order), 8):                      # micro-batches of proteins
        batch = [data[i] for i in order[s:s+8]]
        # per-protein forwards keep gating semantics exact ([L,D] seq + [3] stats)
        outs = []
        for b in batch:
            esm = b["esm"].float().to(device)
            pt5 = b["pt5"].float().to(device)
            af  = b["af"].float().to(device)
            st  = b["stats"].float().to(device)
            with torch.set_grad_enabled(opt is not None):
                outs.append(model(esm, pt5, af, st))
        pred = torch.cat([o["pred"] for o in outs]).to(device)
        k    = torch.cat([o["k"] for o in outs]).to(device)
        th   = torch.cat([o["theta"] for o in outs]).to(device)
        y    = torch.cat([b["y"] for b in batch]).unsqueeze(-1).to(device)
        phi  = torch.cat([b["phi"] for b in batch]).to(device)
        psi  = torch.cat([b["psi"] for b in batch]).to(device)

        with torch.set_grad_enabled(opt is not None):
            loss, logs = composite_loss(pred, y, k, th, phi=phi, psi=psi)
            if opt is not None:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        tot += loss.item(); nb += 1
        preds.append(pred.detach().flatten().cpu())
        tgts.append(y.flatten().cpu())
    p = torch.cat(preds); t = torch.cat(tgts)
    return tot / max(nb, 1), pearson_corr(p, t).item(), p, t

def ece(conf, acc_, n_bins=10):
    conf, acc_ = conf.detach().numpy(), acc_.detach().numpy()
    e = 0.0
    for b in range(n_bins):
        lo, hi = b/n_bins, (b+1)/n_bins
        m = (conf >= lo) & ((conf < hi) if b < n_bins-1 else (conf <= hi))
        if m.sum():
            e += abs(conf[m].mean() - acc_[m].mean()) * m.sum()/len(conf)
    return float(e)

def main(**kw):
    if not kw:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--manifest", required=True)
        ap.add_argument("--n-samples", type=int, default=8000)
        ap.add_argument("--max-len", type=int, default=350)
        ap.add_argument("--epochs", type=int, default=10)
        ap.add_argument("--lr", type=float, default=1e-4)
        ap.add_argument("--out", default="./checkpoints")
        ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
        ap.add_argument("--hf-repo", default="bhumika-tewari-282006/fusionuncertaintynet-checkpoints")
        ap.add_argument("--hf-best-repo", default="bhumika-tewari-282006/fusionuncertaintynet-best")
        a = ap.parse_args()
        kw = vars(a)
    manifest      = kw["manifest"]
    n_samples     = kw.get("n_samples", kw.get("n-samples", 8000))
    max_len       = kw.get("max_len", 350) or kw.get("max-len", 350)
    epochs        = kw.get("epochs", 10)
    lr            = kw.get("lr", 1e-4)
    out           = kw.get("out", "./checkpoints")
    hf_token      = kw.get("hf_token") or os.getenv("HF_TOKEN")
    hf_repo       = kw.get("hf_repo", "bhumika-tewari-282006/fusionuncertaintynet-checkpoints")
    hf_best_repo  = kw.get("hf_best_repo", "bhumika-tewari-282006/fusionuncertaintynet-best")

    device, mode = pick_device()
    print(f"[real-train] device={device} encoder_mode={mode}", flush=True)

    items = []
    with open(manifest) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"[real-train] manifest rows={len(items)}", flush=True)

    encoders = RealEncoders(mode, device)
    train, val = build_split(items, encoders, device,
                             max_len=max_len, limit=n_samples)
    print(f"[real-train] proteins train={len(train)} val={len(val)} "
          f"(residues x{len(train)*len(train[0]['y'])})", flush=True)

    model = FusionUncertaintyNet(d_fused=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_corr, hist = -1.0, []
    for ep in range(1, epochs + 1):
        tr_loss, tr_r, _, _ = run_epoch(model, train, device, opt=opt)
        va_loss, va_r, vp, vt = run_epoch(model, val, device)
        sched.step()
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                     "val_pearson": va_r})
        print(f"[epoch {ep}] train {tr_loss:.3f} | val {va_loss:.3f} "
              f"| pearson {va_r:.4f}", flush=True)

        ck = f"{out}/epoch-{ep}"
        os.makedirs(ck, exist_ok=True)
        model.save_pretrained(ck)
        json.dump({"encoder_mode": mode, "history": hist}, open(f"{ck}/metrics.json", "w"), indent=2)
        _push(ck, hf_repo, hf_token)
        if va_r > best_corr:
            best_corr = va_r
            bk = f"{out}/best"
            os.makedirs(bk, exist_ok=True)
            model.save_pretrained(bk)
            json.dump({"best_val_pearson": best_corr, "encoder_mode": mode,
                       "n_train_proteins": len(train), "n_val_proteins": len(val),
                       "history": hist,
                       "ece": ece(vp.clamp(1,100)/100, vt.clamp(1,100)/100)},
                      open(f"{bk}/metrics.json", "w"), indent=2)
            _push(bk, hf_best_repo, hf_token)

    print(f"[real-train] DONE best_val_pearson={best_corr:.4f}", flush=True)

def _push(path, repo, token):
    if not token:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=path, repo_id=repo, repo_type="model")
        print(f"[push] {path} -> {repo}", flush=True)
    except Exception as e:
        print(f"[push-failed] {e}", flush=True)

if __name__ == "__main__":
    main()
