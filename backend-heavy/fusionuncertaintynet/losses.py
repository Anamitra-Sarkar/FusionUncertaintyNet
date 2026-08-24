"""Composite loss: MSE + Evidential NLL_Gamma + Biophysical constraint."""
import torch
import torch.nn.functional as F
import math

def mse_loss(pred, target):
    """pred [B,L,1] or [L,1], target same shape 0-100"""
    return F.mse_loss(pred, target)

def gamma_nll_loss(y, k, theta, eps=1e-6):
    """
    NLL of Gamma(y|k,theta): -log p(y)
    p(y)= y^{k-1} exp(-y/theta) / (Gamma(k) theta^k)
    y in 0-100 scaled to 0-~100; we clamp y> eps
    Returns mean NLL over all residues.
    """
    y = y.clamp_min(eps)
    k = k.clamp_min(eps+1.0)  # >=1
    theta = theta.clamp_min(eps)
    # log Gamma(k) via lgamma
    # log p = (k-1)log y - y/theta - lgamma(k) - k*log theta
    logp = (k - 1) * torch.log(y) - y / theta - torch.lgamma(k) - k * torch.log(theta)
    nll = -logp
    # mask invalid? just mean
    return nll.mean()

def ramachandran_penalty(phi, psi, weight=0.1):
    """
    Simplified Ramachandran penalty: penalize phi/psi outside allowed beta/alpha regions.
    phi, psi in degrees [-180,180] or None. If None, no penalty (0).
    For demo, we use loose box: alpha [-80,-20]x[-60,10], beta [-150,-90]x[100,180]U[-180,-150]x etc.
    Penalty = distance to nearest allowed center if outside.
    """
    if phi is None or psi is None:
        return torch.tensor(0.0, device=phi.device if phi is not None else "cpu")
    # expect tensors [B,L] or [L]
    # simple: penalize if phi>0 and psi>0 simultaneously (disallowed)
    # this is a proxy for training stability, not full physics
    # penalize extreme outliers: phi in (0,60) and psi in (0,60) etc
    # We compute a soft penalty
    phi = phi.float()
    psi = psi.float()
    # define disallowed square
    # penalty if both in [0, 80] -> high
    mask = ((phi > 0) & (phi < 80) & (psi > 0) & (psi < 80)).float()
    penalty = (mask * ((phi.abs() + psi.abs()) / 180.0)).mean()
    return penalty * weight

def composite_loss(pred, target, k, theta, phi=None, psi=None, lambda_pred=1.0, lambda_ev=0.5, lambda_const=0.1):
    """
    L_total = λ_pred * MSE + λ_ev * NLL_Gamma + λ_const * Ramachandran
    """
    l_pred = mse_loss(pred, target)
    l_ev = gamma_nll_loss(target.clamp_min(1e-3), k, theta)
    l_const = ramachandran_penalty(phi, psi) if phi is not None else torch.tensor(0.0, device=pred.device)
    total = lambda_pred * l_pred + lambda_ev * l_ev + lambda_const * l_const
    return total, {"mse": l_pred.detach(), "ev_nll": l_ev.detach(), "const": l_const.detach() if isinstance(l_const, torch.Tensor) else l_const}

# ---- metrics for logging ----
def pearson_corr(pred, target):
    pred = pred.detach().flatten()
    target = target.detach().flatten()
    if pred.numel() < 2:
        return torch.tensor(0.0)
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = torch.sqrt((pred**2).sum() * (target**2).sum()) + 1e-8
    return (pred * target).sum() / denom

def expected_calibration_error(conf, acc, n_bins=10):
    """Simple ECE: bin conf (0-1), compare to acc (0-1)."""
    # conf, acc are [N] floats 0-1
    conf = conf.detach().cpu()
    acc = acc.detach().cpu()
    ece = 0.0
    for b in range(n_bins):
        low = b / n_bins
        high = (b+1)/ n_bins
        mask = (conf >= low) & (conf < high) if b < n_bins-1 else (conf >= low) & (conf <= high)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean().item()
        bin_acc = acc[mask].mean().item()
        ece += abs(bin_conf - bin_acc) * mask.sum().item() / conf.numel()
    return ece
