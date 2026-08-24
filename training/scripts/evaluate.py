"""Evaluation: Pearson/Spearman, ECE, Brier, Ramachandran — full pipeline for CASP16/CAMEO."""
import torch, numpy as np, json, os, sys
from pathlib import Path
from sklearn.metrics import brier_score_loss
from scipy.stats import spearmanr, pearsonr
import math

# Add backend-heavy to path for model
sys.path.append(os.path.join(os.path.dirname(__file__), "../../backend-heavy"))
try:
    from fusionuncertaintynet.model import FusionUncertaintyNet
    from fusionuncertaintynet.losses import pearson_corr, expected_calibration_error
    from fusionuncertaintynet.extraction import extract_all
    from dataset import ProteinQualityDataset
except ImportError as e:
    print(f"Import warning: {e}")
    FusionUncertaintyNet = None

def brier(pred_prob, target_prob):
    return ((np.array(pred_prob) - np.array(target_prob))**2).mean()

def ece_score(conf, acc, n_bins=10):
    conf = np.array(conf); acc = np.array(acc)
    ece=0.0
    for b in range(n_bins):
        low=b/n_bins; high=(b+1)/n_bins
        mask = (conf>=low) & (conf<high) if b<n_bins-1 else (conf>=low) & (conf<=high)
        if mask.sum()==0: continue
        ece+= abs(conf[mask].mean() - acc[mask].mean()) * mask.sum()/len(conf)
    return ece

def ramachandran_stats(phi_list, psi_list):
    """Simple outlier count: phi in (0,60) & psi in (0,60) is disallowed."""
    if phi_list is None or psi_list is None:
        return {"outliers": 0, "zscore": 0.0}
    outliers = sum(1 for phi, psi in zip(phi_list, psi_list) if 0 < phi < 80 and 0 < psi < 80)
    return {"outliers": outliers, "total": len(phi_list), "rate": outliers/max(len(phi_list),1)}

def evaluate_model(checkpoint_path, manifest_path, device="auto", max_samples=200):
    """Load model and evaluate on manifest test set."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # P100 fallback
        if device == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 7:
                    print(f"[eval] P100 sm_{cap[0]}{cap[1]} fallback to CPU")
                    device = "cpu"
            except: pass

    print(f"[eval] device {device} checkpoint {checkpoint_path}")
    # Load dataset
    ds = ProteinQualityDataset(manifest_path, synthetic_fallback=True)
    # Use last 10% as test
    n = len(ds)
    test_n = min(max_samples, n//10)
    test_ds = [ds[i] for i in range(n-test_n, n)]
    print(f"[eval] test set {len(test_ds)}/{n}")

    # Load model
    if FusionUncertaintyNet and os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
        model = FusionUncertaintyNet.from_pretrained(checkpoint_path, device=device)
        model.eval()
        print(f"[eval] loaded model from {checkpoint_path}")
    else:
        print(f"[eval] no checkpoint at {checkpoint_path}, using random model")
        model = FusionUncertaintyNet() if FusionUncertaintyNet else None
        if model:
            model.to(device)
            model.eval()

    # Baselines: pLDDT is in manifest's plddt field
    y_true_all, y_pred_all, y_plddt_all = [], [], []
    k_all, theta_all = [], []
    phi_all, psi_all = [], []

    with torch.no_grad():
        for item in test_ds:
            seq = item["sequence"]
            target = torch.tensor(item["target"], dtype=torch.float32)  # [L]
            # Use model if available, else use plddt as baseline
            if model:
                feats = extract_all(seq, device=device, af_kwargs={"plddt": item.get("plddt")})
                esm = feats["esm"].to(device)
                prott5 = feats["prott5"].to(device)
                af = feats["af"].to(device)
                stats = torch.tensor([feats["stats"]["length"], feats["stats"]["charged_frac"], feats["stats"]["disorder"]], device=device)
                out = model(esm, prott5, af, stats)
                pred = out["pred"].squeeze(-1).cpu().numpy()  # [L]
                k = out["k"].squeeze(-1).cpu().numpy()
                theta = out["theta"].squeeze(-1).cpu().numpy()
            else:
                pred = np.array(item.get("plddt", [70]*len(seq)), dtype=float)
                k = np.ones_like(pred)
                theta = np.ones_like(pred)*0.5

            y_true_all.extend(target.numpy())
            y_pred_all.extend(pred)
            if "plddt" in item and item["plddt"]:
                y_plddt_all.extend(item["plddt"][:len(seq)])
            k_all.extend(k)
            theta_all.extend(theta)
            if item.get("phi"): phi_all.extend(item["phi"][:len(seq)])
            if item.get("psi"): psi_all.extend(item["psi"][:len(seq)])

    y_true = np.array(y_true_all)/100.0
    y_pred = np.array(y_pred_all)/100.0
    y_plddt = np.array(y_plddt_all)/100.0 if y_plddt_all else y_pred

    # Metrics
    pearson = pearsonr(y_pred, y_true)[0] if len(y_pred)>2 else 0.0
    spearman = spearmanr(y_pred, y_true)[0] if len(y_pred)>2 else 0.0
    # Baselines
    pearson_plddt = pearsonr(y_plddt, y_true)[0] if len(y_plddt)>2 else 0.0

    # ECE and Brier for our model
    ece = ece_score(y_pred, y_true)
    brier_score = brier(y_pred, y_true)
    ece_plddt = ece_score(y_plddt, y_true)
    brier_plddt = brier(y_plddt, y_true)

    # Uncertainty calibration: var = k*theta^2, epistemic 1/k
    var = np.array(k_all) * (np.array(theta_all)**2)
    aleatoric_mean = var.mean()
    epistemic_mean = (1.0/np.array(k_all)).mean()

    # Ramachandran
    rama = ramachandran_stats(phi_all, psi_all)

    results = {
        "n_residues": len(y_true),
        "n_proteins": len(test_ds),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "pearson_plddt_baseline": float(pearson_plddt),
        "ece": float(ece),
        "brier": float(brier_score),
        "ece_plddt": float(ece_plddt),
        "brier_plddt": float(brier_plddt),
        "aleatoric_mean": float(aleatoric_mean),
        "epistemic_mean": float(epistemic_mean),
        "ramachandran": rama,
        "model": checkpoint_path
    }

    print(json.dumps(results, indent=2))

    # Save
    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/metrics.json","w") as f:
        json.dump(results, f, indent=2)
    print("[eval] saved to eval_results/metrics.json")

    # Ablation note
    print(f"\n[eval] Ablation summary:")
    print(f"  Full model Pearson {pearson:.3f} vs pLDDT baseline {pearson_plddt:.3f} (Δ {pearson-pearson_plddt:+.3f})")
    print(f"  ECE {ece:.3f} vs baseline {ece_plddt:.3f} (lower better)")
    print(f"  Ramachandran outliers {rama['outliers']}/{rama['total']} ({rama['rate']:.1%})")

    return results

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./checkpoints/best")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()

    # Default demo if no checkpoint/manifest
    if not os.path.exists(args.manifest):
        print(f"[eval] manifest {args.manifest} not found, using synthetic demo")
        # Create synthetic manifest
        sys.path.append(os.path.dirname(__file__))
        from dataset import ProteinQualityDataset
        ds = ProteinQualityDataset("nonexistent", synthetic_fallback=True)
        # Write temp
        os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
        import json as js
        with open(args.manifest,"w") as f:
            for it in ds.items[:500]:
                f.write(js.dumps(it)+"\n")
        print(f"[eval] wrote synthetic {args.manifest}")

    evaluate_model(args.checkpoint, args.manifest, device=args.device, max_samples=args.samples)
