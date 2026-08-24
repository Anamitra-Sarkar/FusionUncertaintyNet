"""Adaptive Fusion Module — gating conditioned on sequence stats."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusion(nn.Module):
    """
    Inputs:
      esm:   [B, L, 1280] or [L,1280]  -> proj to 512
      prott5:[B, L, 1024] or [L,1024] -> proj to 512
      af:    [B, L, 7]    or [L,7]    -> proj to 512
      stats: [B, 3] or [3]  (length_norm, charged_frac, disorder)
    Outputs:
      fused: [B, L, 512] or [L,512]
      gates: [B, 3] or [3]  softmax weights
    """
    def __init__(self, d_esm=1280, d_prott5=1024, d_af=7, d_fused=512, stats_dim=3, hidden=32, dropout=0.1):
        super().__init__()
        self.proj_esm = nn.Sequential(
            nn.Linear(d_esm, d_fused),
            nn.LayerNorm(d_fused),
            nn.Dropout(dropout)
        )
        self.proj_prott5 = nn.Sequential(
            nn.Linear(d_prott5, d_fused),
            nn.LayerNorm(d_fused),
            nn.Dropout(dropout)
        )
        self.proj_af = nn.Sequential(
            nn.Linear(d_af, d_fused),
            nn.LayerNorm(d_fused),
            nn.Dropout(dropout)
        )
        # Gating MLP: stats -> 3 weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(stats_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, esm, prott5, af, stats):
        # handle unbatched [L, D] -> [1, L, D]
        single = esm.dim() == 2
        if single:
            esm = esm.unsqueeze(0)
            prott5 = prott5.unsqueeze(0)
            af = af.unsqueeze(0)
            stats = stats.unsqueeze(0) if isinstance(stats, torch.Tensor) else torch.tensor([stats]).unsqueeze(0)

        # ensure stats is tensor [B,3]
        if not isinstance(stats, torch.Tensor):
            stats = torch.tensor(stats, dtype=torch.float32, device=esm.device)
            if stats.dim() == 1:
                stats = stats.unsqueeze(0).expand(esm.size(0), -1)
        else:
            stats = stats.to(esm.device).float()
            if stats.dim() == 1:
                stats = stats.unsqueeze(0).expand(esm.size(0), -1)

        g_logits = self.gate_mlp(stats)  # [B,3]
        gates = F.softmax(g_logits, dim=-1)  # [B,3]

        # project
        e = self.proj_esm(esm)       # [B,L,512]
        p = self.proj_prott5(prott5) # [B,L,512]
        a = self.proj_af(af)         # [B,L,512]

        # weighted sum: gates[:,0] for esm etc — broadcast over L
        g0 = gates[:, 0].view(-1, 1, 1)
        g1 = gates[:, 1].view(-1, 1, 1)
        g2 = gates[:, 2].view(-1, 1, 1)

        fused = g0 * e + g1 * p + g2 * a
        fused = self.dropout(fused)

        if single:
            fused = fused.squeeze(0)
            gates = gates.squeeze(0)
        return fused, gates

    def get_gates(self, stats):
        """Utility for inference logging without full forward."""
        if not isinstance(stats, torch.Tensor):
            stats = torch.tensor(stats, dtype=torch.float32)
        if stats.dim() == 1:
            stats = stats.unsqueeze(0)
        logits = self.gate_mlp(stats)
        return F.softmax(logits, dim=-1)
