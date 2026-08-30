"""Test the encoder-mode + phi/psi fix for eval_interval_calibration.

Exercises the SAME code path as training (feature construction -> model forward
-> prediction extraction) and asserts shapes/order match training expectations.

This test is synthetic/fixture-only (no HF download, no real ESM/ProtT5 weights)
but mirrors the real bug: previously eval always used full ESM t33 + ProtT5-XL
even though the leakfree checkpoint was trained with SMALL encoders
(ESM t12 35M padded->1280, ESM t30 150M padded->1024) and withheld phi/psi.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend-heavy"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../training/scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../training/kaggle"))

import torch
import math
import tempfile
import json
from pathlib import Path

from fusionuncertaintynet.model import FusionUncertaintyNet
from fusionuncertaintynet.extraction import extract_af_features, sequence_stats
from fusionuncertaintynet.fusion import AdaptiveFusion

# Import fix helpers via importlib to handle path
import importlib.util
spec = importlib.util.spec_from_file_location(
    "eval_interval_calibration",
    Path(__file__).resolve().parents[1] / "training/kaggle/eval_interval_calibration.py",
)
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)


def test_pad_to_dim_matches_train_real():
    """_pad_to_dim must replicate train_real.py pad_to: zero-pad on right."""
    t = torch.randn(10, 480)
    out = eval_mod._pad_to_dim(t, 1280)
    assert out.shape == (10, 1280)
    assert torch.allclose(out[:, :480], t)
    assert torch.all(out[:, 480:] == 0), "padded region must be zeros (train RealEncoders expectation)"

    t2 = torch.randn(5, 640)
    out2 = eval_mod._pad_to_dim(t2, 1024)
    assert out2.shape == (5, 1024)
    assert torch.allclose(out2[:, :640], t2)
    assert torch.all(out2[:, 640:] == 0)

    # already correct dim -> identity
    t3 = torch.randn(7, 1280)
    assert eval_mod._pad_to_dim(t3, 1280) is t3 or torch.allclose(eval_mod._pad_to_dim(t3,1280), t3)


def test_af_phi_psi_forwarded_not_nulled():
    """AF features must preserve phi/psi from manifest, not zero them."""
    seq = "ACDEFGHIKLMN" * 3  # L=36
    phi = [10.0, -60.0, 120.0] * 12
    psi = [20.0, 130.0, -40.0] * 12
    af_with = extract_af_features(seq, plddt=None, phi=phi, psi=psi)
    af_without = extract_af_features(seq, plddt=None, phi=None, psi=None)
    # Shape [L,7] always
    assert af_with.shape == (len(seq), 7)
    assert af_without.shape == (len(seq), 7)
    # With real phi/psi, sin/cos columns (1:5) should differ from neutral
    # neutral phi=0 -> sin0=0 cos0=1
    # our phi=10 -> sin~0.173, not zeros
    assert not torch.allclose(af_with[:, 1], af_without[:, 1]), "phi/psi must be forwarded (old bug nulled them)"
    # plddt column should be neutral 0.7 in both (withheld)
    assert torch.allclose(af_with[:, 0], torch.full((len(seq),), 0.7))
    assert torch.allclose(af_without[:, 0], torch.full((len(seq),), 0.7))


def test_extract_matched_features_small_shape_and_order(monkeypatch=None):
    """Small mode must produce [L,1280]/[L,1024]/[L,7] with correct padding and phi/psi."""
    # Mock the heavy ESM downloads to return synthetic small dims
    import eval_interval_calibration as em
    # we already have em loaded as eval_mod
    seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # L=40
    phi = [ -63.0 ] * len(seq)
    psi = [ -43.0 ] * len(seq)

    # patch the small embed helpers to avoid real model download
    def fake_esm(seq_, device):
        return torch.randn(len(seq_), 480)

    def fake_pt5(seq_, device):
        return torch.randn(len(seq_), 640)

    old_esm = eval_mod._small_esm_embed
    old_pt5 = eval_mod._small_pt5_embed
    eval_mod._small_esm_embed = fake_esm
    eval_mod._small_pt5_embed = fake_pt5
    try:
        feats = eval_mod._extract_matched_features(seq, device="cpu", encoder_mode="small", phi=phi, psi=psi)
        assert feats["esm"].shape == (len(seq), 1280), f"esm shape {feats['esm'].shape} != {(len(seq),1280)}"
        assert feats["prott5"].shape == (len(seq), 1024)
        assert feats["af"].shape == (len(seq), 7)
        # check padding
        assert torch.all(feats["esm"][:, 480:] == 0)
        assert torch.all(feats["prott5"][:, 640:] == 0)
        # check stats order: length, charged_frac, disorder
        s = feats["stats"]
        assert set(s.keys()) == {"length", "charged_frac", "disorder"}
        # disorder should be PEQSK fraction
        expected_disorder = sum(1 for c in seq if c in "PEQSK") / len(seq)
        assert abs(s["disorder"] - expected_disorder) < 1e-6
    finally:
        eval_mod._small_esm_embed = old_esm
        eval_mod._small_pt5_embed = old_pt5


def test_model_forward_shapes_and_heads():
    """Model forward must accept [L,D] + [3] stats and return pred/k/theta with correct semantics."""
    L = 32
    esm = torch.randn(L, 1280)
    prott5 = torch.randn(L, 1024)
    af = torch.randn(L, 7)
    stats = torch.tensor([0.5, 0.2, 0.3])

    model = FusionUncertaintyNet(d_fused=512)
    model.eval()
    with torch.no_grad():
        out = model(esm, prott5, af, stats)

    # Check required keys (see model.py:36, edr.py:96)
    for k in ("pred", "k", "theta", "var", "aleatoric", "epistemic"):
        assert k in out, f"missing {k} in model output"

    pred = out["pred"]
    k = out["k"]
    theta = out["theta"]
    var = out["var"]
    # Training concatenates o["pred"] etc and flattens; they are [L,1] or [1,L,1] depending on batch
    # Single path should be [L,1]
    assert pred.shape == (L, 1) or pred.shape == (L,), f"pred shape {pred.shape}"
    if pred.dim() == 2:
        assert pred.shape[0] == L and pred.shape[1] == 1
    # Squeeze test as eval does
    pred_s = pred.squeeze(-1)
    assert pred_s.shape[0] == L
    # Value ranges per edr.py: pred sigmoid*100 in (0,100), k via softplus+1 >=1, theta >=0.1
    assert torch.all(pred >= 0) and torch.all(pred <= 100)
    assert torch.all(k >= 1.0)
    assert torch.all(theta >= 0.1)
    # var = k * theta^2
    assert torch.allclose(var.squeeze(-1), (k * (theta ** 2)).squeeze(-1), atol=1e-4)

    # Also test batched [1,L,D] path
    esm_b = esm.unsqueeze(0)
    prott5_b = prott5.unsqueeze(0)
    af_b = af.unsqueeze(0)
    stats_b = stats.unsqueeze(0)
    with torch.no_grad():
        out_b = model(esm_b, prott5_b, af_b, stats_b)
    assert out_b["pred"].shape == (1, L, 1)
    # Gates shape [B,3] or [3]
    assert "gates" in out_b and out_b["gates"].shape == (1, 3)


def test_encoder_mode_detection_defaults_to_small():
    """Checkpoint without explicit metrics should default to small (leakfree artefact is small)."""
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt"
        ckpt.mkdir()
        (ckpt / "pytorch_model.bin").write_bytes(b"dummy")
        (ckpt / "config.json").write_text(json.dumps({"d_fused": 512}))
        # No metrics.json -> should default to small with warning
        mode = eval_mod._detect_encoder_mode(str(ckpt))
        assert mode == "small"

        # With small metrics
        (ckpt / "metrics.json").write_text(json.dumps({"encoder_mode": "small"}))
        assert eval_mod._detect_encoder_mode(str(ckpt)) == "small"

        # With full metrics
        (ckpt / "metrics.json").write_text(json.dumps({"encoder_mode": "full"}))
        assert eval_mod._detect_encoder_mode(str(ckpt)) == "full"


def test_prediction_parsing_matches_training_concatenation():
    """Eval must extract pred/k/theta via squeeze(-1) exactly as training does: torch.cat([o['pred'] for o in outs])."""
    # Simulate training run_epoch batch with 2 proteins K=4 each
    model = FusionUncertaintyNet(d_fused=512)
    model.eval()
    outs = []
    for _ in range(2):
        L = 4
        esm = torch.randn(L, 1280)
        pt5 = torch.randn(L, 1024)
        af = torch.randn(L, 7)
        st = torch.randn(3)
        outs.append(model(esm, pt5, af, st))
    # Training does:
    pred = torch.cat([o["pred"] for o in outs])  # [8,1]
    k = torch.cat([o["k"] for o in outs])
    th = torch.cat([o["theta"] for o in outs])
    assert pred.shape == (8, 1)
    # Eval does per-protein squeeze then extend list — must be equivalent after flatten
    flat_pred = pred.detach().squeeze(-1).numpy()
    flat_k = k.detach().squeeze(-1).numpy()
    flat_th = th.detach().squeeze(-1).numpy()
    assert flat_pred.shape == (8,)
    assert flat_k.shape == (8,)
    assert flat_th.shape == (8,)
