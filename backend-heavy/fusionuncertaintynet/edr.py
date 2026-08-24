"""Evidential Deep Regression — dual head Gamma."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = x + residual
        x = self.act(x)
        return x

class EvidentialHead(nn.Module):
    """
    Fused 512 -> 512 ->256 ->128 -> (pred + k,theta)
    Inputs: [B,L,512] or [L,512]
    Outputs: pred [B,L,1] in 0-100, k [B,L,1]>0, theta [B,L,1]>0
    """
    def __init__(self, d_in=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block1 = ResidualBlock(512, dropout)
        self.proj256 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block2 = ResidualBlock(256, dropout)
        self.proj128 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # dual heads
        self.head_pred = nn.Linear(128, 1)
        self.head_ev = nn.Linear(128, 2)  # k, theta raw

    def forward(self, fused):
        # fused: [B,L,512] or [L,512]
        single = fused.dim() == 2
        if single:
            fused = fused.unsqueeze(0)
        x = self.input_proj(fused)
        x = self.block1(x)
        x = self.proj256(x)
        x = self.block2(x)
        x = self.proj128(x)
        pred_raw = self.head_pred(x)  # [B,L,1]
        ev_raw = self.head_ev(x)      # [B,L,2]

        # pred to 0-100 via sigmoid
        pred = torch.sigmoid(pred_raw) * 100.0
        # k, theta via softplus + epsilon to avoid 0
        k = F.softplus(ev_raw[..., 0:1]) + 1e-3 + 1.0  # >=1 for stability
        theta = F.softplus(ev_raw[..., 1:2]) + 1e-3 + 0.1

        if single:
            pred = pred.squeeze(0)
            k = k.squeeze(0)
            theta = theta.squeeze(0)
        return pred, k, theta

    def predict_with_uncertainty(self, fused):
        """Returns dict with mean, aleatoric, epistemic."""
        pred, k, theta = self.forward(fused)
        # Gamma mean = k*theta, var = k*theta^2
        # Our pred is already calibrated mean; but ensure consistency: we return pred as mean
        # For uncertainty: aleatoric ~ var, epistemic ~ 1/k
        var = k * (theta ** 2)
        aleatoric = var  # [B,L,1]
        epistemic = 1.0 / k  # proxy
        total = aleatoric + epistemic * 50.0  # scaled sum for ranking
        return {
            "pred": pred,
            "k": k,
            "theta": theta,
            "mean": pred,  # same
            "var": var,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_unc": total
        }
