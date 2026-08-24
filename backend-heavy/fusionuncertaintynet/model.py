"""Full FusionUncertaintyNet — end-to-end."""
import torch
import torch.nn as nn
from .fusion import AdaptiveFusion
from .edr import EvidentialHead
from .extraction import sequence_stats

class FusionUncertaintyNet(nn.Module):
    def __init__(self, d_fused=512, dropout=0.1):
        super().__init__()
        self.fusion = AdaptiveFusion(d_fused=d_fused, dropout=dropout)
        self.edr = EvidentialHead(d_in=d_fused, dropout=dropout)
        self.d_fused = d_fused

    def forward(self, esm, prott5, af, stats=None, seq: str | None = None):
        """
        esm: [B,L,1280] or [L,1280]
        prott5: [B,L,1024] or [L,1024]
        af: [B,L,7] or [L,7]
        stats: [B,3] or [3] or None (computed from seq if provided)
        """
        if stats is None and seq is not None:
            s = sequence_stats(seq)
            stats = torch.tensor([s["length"], s["charged_frac"], s["disorder"]], dtype=torch.float32, device=esm.device)
            if esm.dim() == 3:
                stats = stats.unsqueeze(0).expand(esm.size(0), -1)
        elif stats is None:
            # neutral
            B = esm.size(0) if esm.dim()==3 else 1
            device = esm.device
            stats = torch.tensor([[0.5, 0.2, 0.5]]*B, dtype=torch.float32, device=device)
            if esm.dim()==2:
                stats = stats.squeeze(0)

        fused, gates = self.fusion(esm, prott5, af, stats)
        out = self.edr.predict_with_uncertainty(fused)
        out["fused"] = fused
        out["gates"] = gates
        return out

    def predict_sequence(self, seq: str, af_kwargs=None, device="auto"):
        """Convenience: extract features then predict. For inference server, extraction done outside for caching."""
        from .extraction import extract_all
        res = extract_all(seq, device=device, af_kwargs=af_kwargs)
        # to tensor device
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        esm = res["esm"].to(dev)
        prott5 = res["prott5"].to(dev)
        af = res["af"].to(dev)
        stats_list = [res["stats"]["length"], res["stats"]["charged_frac"], res["stats"]["disorder"]]
        stats = torch.tensor(stats_list, dtype=torch.float32, device=dev)
        self.to(dev)
        self.eval()
        with torch.no_grad():
            out = self.forward(esm, prott5, af, stats, seq=seq)
        # move to cpu
        for k in out:
            if isinstance(out[k], torch.Tensor):
                out[k] = out[k].cpu()
        return out

    def save_pretrained(self, path):
        import os, json
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        with open(f"{path}/config.json", "w") as f:
            json.dump({"d_fused": self.d_fused, "architecture": "FusionUncertaintyNet"}, f, indent=2)

    @classmethod
    def from_pretrained(cls, path, device="cpu"):
        import json, os, torch
        with open(f"{path}/config.json") as f:
            cfg = json.load(f)
        model = cls(d_fused=cfg.get("d_fused", 512))
        sd = torch.load(f"{path}/pytorch_model.bin", map_location=device)
        model.load_state_dict(sd)
        model.to(device)
        return model
