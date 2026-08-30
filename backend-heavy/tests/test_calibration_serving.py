"""Tests for opt-in isotonic calibration wiring in backend-heavy/app/main.py.

Verifies:
  (a) with CALIBRATION_ARTIFACT_ENABLED unset (default), response is byte-identical to raw
  (b) with enabled=true and synthetic breakpoints, calibration is applied and
      changes theta/aleatoric/total in expected monotone direction (<1 factor for over-coverage)
  (c) apply_isotonic_calibration round-trips against hand-computed synthetic points
  (d) fail-closed: enabled but missing artifact falls back to raw (no 500)
"""
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient


def _load_heavy(monkeypatch, extra_env=None):
    # clear calibration env
    for k in [
        "CALIBRATION_ARTIFACT_ENABLED",
        "CALIBRATION_ARTIFACT_REPO",
        "CALIBRATION_ARTIFACT_PATH",
        "CALIBRATION_ARTIFACT_REVISION",
        "CALIBRATION_ARTIFACT_LOCAL_PATH",
        "MODEL_RELEASE_APPROVED",
        "MODEL_ARTIFACT_REVISION",
        "MODEL_PATH",
        "HEAVY_SHARED_SECRET",
    ]:
        monkeypatch.delenv(k, raising=False)
    if extra_env:
        for k, v in extra_env.items():
            monkeypatch.setenv(k, v)
    # ensure required auth for predict
    monkeypatch.setenv("HEAVY_SHARED_SECRET", "test-internal-credential")
    # import fresh
    import app.main as heavy

    heavy = importlib.reload(heavy)
    # reset calibration cache after reload
    try:
        heavy._reset_calibration_cache()
    except Exception:
        pass
    return heavy


class FakeModel:
    def __init__(self, k_val=2.0, theta_val=1.0, pred_val=50.0):
        self.k_val = k_val
        self.theta_val = theta_val
        self.pred_val = pred_val

    def eval(self):
        pass

    def __call__(self, esm, prott5, af, stats, seq=None):
        # seq length determines L
        L = esm.shape[0] if esm.dim() == 2 else esm.shape[1]
        # allow test to vary by seq length; produce constant outputs
        device = esm.device
        pred = torch.full((L, 1), self.pred_val, dtype=torch.float32, device=device)
        k = torch.full((L, 1), self.k_val, dtype=torch.float32, device=device)
        theta = torch.full((L, 1), self.theta_val, dtype=torch.float32, device=device)
        var = k * (theta ** 2)
        ale = var
        epi = 1.0 / k
        tot = ale + epi * 50.0
        gates = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
        return {
            "pred": pred,
            "k": k,
            "theta": theta,
            "aleatoric": ale,
            "epistemic": epi,
            "total_unc": tot,
            "gates": gates,
        }


def _mock_extraction(monkeypatch):
    """Mock extract_all to avoid needing real ESM weights."""
    import types

    fake_mod = types.ModuleType("fusionuncertaintynet.extraction")

    def extract_all(seq, device="cpu", af_kwargs=None):
        L = len(seq)
        # return minimal dict matching real extraction
        return {
            "esm": torch.zeros(L, 1280, dtype=torch.float32),
            "prott5": torch.zeros(L, 1024, dtype=torch.float32),
            "af": torch.zeros(L, 7, dtype=torch.float32),
            "stats": {"length": float(L), "charged_frac": 0.2, "disorder": 0.1},
        }

    def extract_af_features(seq, plddt=None, phi=None, psi=None):
        return torch.zeros(len(seq), 7)

    def sequence_stats(seq):
        return {"length": float(len(seq)), "charged_frac": 0.2, "disorder": 0.1}

    fake_mod.extract_all = extract_all
    fake_mod.extract_af_features = extract_af_features
    fake_mod.sequence_stats = sequence_stats
    monkeypatch.setitem(sys.modules, "fusionuncertaintynet.extraction", fake_mod)
    return fake_mod


def _make_breakpoints_overcoverage(n=200):
    """Synthetic steep map simulating over-coverage (factor <1)."""
    x = np.linspace(0, 1, n)
    # steep in central region: y = 0.5 + 1.8*(x-0.5) clipped -> empirical more extreme
    y = np.clip(0.5 + 1.8 * (x - 0.5), 0, 1)
    return {"x": x.tolist(), "y": y.tolist()}


def _make_breakpoints_identity(n=200):
    x = np.linspace(0, 1, n)
    y = x.copy()
    return {"x": x.tolist(), "y": y.tolist()}


def test_calibration_disabled_is_byte_identical(monkeypatch):
    """(a) Most important regression: default (env unset) returns raw values."""
    heavy = _load_heavy(monkeypatch)
    _mock_extraction(monkeypatch)
    fake = FakeModel(k_val=2.0, theta_val=1.0, pred_val=60.0)
    monkeypatch.setattr(heavy, "get_model", lambda: fake)

    client = TestClient(heavy.app)
    payload = {"sequence": "ACDEFG"}
    resp = client.post("/predict", json=payload, headers={"X-Render-Secret": "test-internal-credential"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # raw: var =2*1=2, ale=2, epi=0.5, tot=27
    r0 = data["residues"][0]
    assert abs(r0["k"] - 2.0) < 1e-6
    assert abs(r0["theta"] - 1.0) < 1e-6
    assert abs(r0["aleatoric"] - 2.0) < 1e-4
    assert abs(r0["epistemic"] - 0.5) < 1e-4
    assert abs(r0["total_unc"] - 27.0) < 1e-4

    # also verify calling again with explicitly disabled still same
    monkeypatch.setenv("CALIBRATION_ARTIFACT_ENABLED", "false")
    heavy._reset_calibration_cache()
    resp2 = client.post("/predict", json=payload, headers={"X-Render-Secret": "test-internal-credential"})
    assert resp2.status_code == 200
    assert resp2.json() == data, "disabled should be byte-identical"


def test_calibration_enabled_applies_and_shrinks(monkeypatch):
    """(b) With enabled=true and synthetic steep map, theta/ale/tot shrink."""
    bp = _make_breakpoints_overcoverage(n=500)
    artifact = {
        "version": "calibration-isotonic-v1",
        "fit_date": "2026-08-30T00:00:00Z",
        "checkpoint_repo": "bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree",
        "test_interval_calibration_error": 0.007,
        "breakpoints": bp,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(artifact, tf)
        tf_path = tf.name

    try:
        heavy = _load_heavy(
            monkeypatch,
            extra_env={
                "CALIBRATION_ARTIFACT_ENABLED": "true",
                "CALIBRATION_ARTIFACT_LOCAL_PATH": tf_path,
            },
        )
        _mock_extraction(monkeypatch)
        fake = FakeModel(k_val=2.0, theta_val=1.0, pred_val=50.0)
        monkeypatch.setattr(heavy, "get_model", lambda: fake)
        # ensure cache cleared
        heavy._reset_calibration_cache()

        client = TestClient(heavy.app)
        payload = {"sequence": "ACDEFGHIK"}
        resp = client.post("/predict", json=payload, headers={"X-Render-Secret": "test-internal-credential"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        r0 = data["residues"][0]
        # steep map should give factor <1 -> theta <1, ale <2, tot <27
        assert r0["theta"] < 1.0, f"expected calibrated theta <1, got {r0['theta']}"
        assert r0["aleatoric"] < 2.0, f"ale should shrink, got {r0['aleatoric']}"
        assert r0["total_unc"] < 27.0, f"tot should shrink, got {r0['total_unc']}"
        # k unchanged, pred unchanged, epistemic unchanged (since k unchanged)
        assert abs(r0["k"] - 2.0) < 1e-6
        assert abs(r0["epistemic"] - 0.5) < 1e-6
        assert 0.3 < r0["theta"] < 0.9, f"factor should be in reasonable shrink range, got {r0['theta']}"

        # verify monotone: identity map should not change
        bp_id = _make_breakpoints_identity()
        art2 = {**artifact, "breakpoints": bp_id}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf2:
            json.dump(art2, tf2)
            tf2_path = tf2.name
        monkeypatch.setenv("CALIBRATION_ARTIFACT_LOCAL_PATH", tf2_path)
        heavy._reset_calibration_cache()
        resp2 = client.post("/predict", json=payload, headers={"X-Render-Secret": "test-internal-credential"})
        assert resp2.status_code == 200
        r0_id = resp2.json()["residues"][0]
        # identity should give factor ~1 (within 0.02)
        assert abs(r0_id["theta"] - 1.0) < 0.02, f"identity should not rescale, got {r0_id['theta']}"
        os.unlink(tf2_path)
    finally:
        os.unlink(tf_path)
        # cleanup cache
        try:
            heavy._reset_calibration_cache()
        except Exception:
            pass


def test_calibration_fail_closed_when_missing(monkeypatch):
    """Fail-closed: enabled but artifact missing -> still returns raw 200."""
    heavy = _load_heavy(
        monkeypatch,
        extra_env={
            "CALIBRATION_ARTIFACT_ENABLED": "true",
            "CALIBRATION_ARTIFACT_LOCAL_PATH": "/tmp/nonexistent-calib-xyz.json",
            # also point HF to nonexistent to force download failure
            "CALIBRATION_ARTIFACT_REPO": "nonexistent/repo",
        },
    )
    _mock_extraction(monkeypatch)
    fake = FakeModel(k_val=2.0, theta_val=1.0)
    monkeypatch.setattr(heavy, "get_model", lambda: fake)
    heavy._reset_calibration_cache()

    client = TestClient(heavy.app)
    resp = client.post("/predict", json={"sequence": "ACDEFG"}, headers={"X-Render-Secret": "test-internal-credential"})
    # should not be 500 — fail-closed to raw
    assert resp.status_code == 200, resp.text
    r0 = resp.json()["residues"][0]
    assert abs(r0["theta"] - 1.0) < 1e-6, "fallback should be raw"
    assert abs(r0["aleatoric"] - 2.0) < 1e-4


def test_apply_isotonic_calibration_round_trip_hand_computed(monkeypatch):
    """(c) Pure-numpy helper round-trips against hand-computed synthetic points."""
    # import helper directly
    from fusionuncertaintynet.calibration import (
        apply_isotonic_calibration,
        compute_calibrated_interval_coverage_via_breakpoints,
    )

    # Simple identity breakpoints: should leave calibration factor =1
    x_id = np.linspace(0, 1, 10)
    y_id = x_id.copy()
    bp_id = {"x": x_id.tolist(), "y": y_id.tolist()}
    pred = np.array([50.0, 60.0])
    k = np.array([2.0, 3.0])
    theta = np.array([1.0, 0.5])
    k_cal, th_cal, ale_cal, epi_cal, tot_cal = apply_isotonic_calibration(pred, k, theta, bp_id)
    assert np.allclose(th_cal, theta, atol=0.02), f"identity should not change theta {th_cal} vs {theta}"
    # ale = k*theta^2, epi=1/k
    assert np.allclose(ale_cal, k * theta**2, atol=1e-6)
    assert np.allclose(epi_cal, 1.0 / k, atol=1e-9)

    # Steep map: hand compute expected factor via direct ppf ratio
    # Use small 5-point steep map for hand check
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.array([0.0, 0.15, 0.5, 0.85, 1.0])  # shallower? Actually compute factor manually via helper's _calibration_factor
    bp = {"x": x.tolist(), "y": y.tolist()}
    # Apply helper
    k2 = np.array([2.0])
    th2 = np.array([1.0])
    _, th_cal2, ale2, _, _ = apply_isotonic_calibration([50.0], k2, th2, bp)
    # Hand compute factor using same logic as helper (mean width ratio)
    from fusionuncertaintynet.calibration import _norm_ppf, _invert_breakpoints, _get_xy

    xb, yb = _get_xy(bp)
    ratios = []
    for lvl in (0.50, 0.80, 0.90, 0.95):
        lo = 0.5 - lvl / 2
        hi = 0.5 + lvl / 2
        rl = float(_invert_breakpoints(lo, xb, yb))
        rh = float(_invert_breakpoints(hi, xb, yb))
        num = float(_norm_ppf(rh)) - float(_norm_ppf(rl))
        den = float(_norm_ppf(hi)) - float(_norm_ppf(lo))
        ratios.append(num / den)
    expected_factor = float(np.mean(ratios))
    expected_factor = float(np.clip(expected_factor, 0.05, 5.0))
    assert abs(float(th_cal2[0]) - expected_factor) < 1e-6, f"hand factor {expected_factor} vs got {th_cal2[0]}"

    # Also check coverage via breakpoints matches sklearn-free path
    # Synthetic data where calibration should help: generate with over-coverage
    rng = np.random.default_rng(0)
    n = 2000
    y_pred = rng.uniform(40, 80, size=n)
    k_s = rng.uniform(1.5, 3.0, size=n)
    th_s = rng.uniform(0.5, 1.0, size=n)
    var = k_s * th_s**2
    std = np.sqrt(var)
    y_true = y_pred + rng.normal(0, 1, size=n) * std * 0.4  # over-coverage (true std smaller)
    # Fit isotonic on first half via sklearn to get "true" breakpoints, then check round-trip via pure-numpy
    # Here we simulate with our breakpoints directly
    # Just verify compute_calibrated_interval_coverage_via_breakpoints runs and returns plausible error
    cov, err = compute_calibrated_interval_coverage_via_breakpoints(y_true, y_pred, k_s, th_s, bp)
    assert 0 <= err <= 1
    assert set(cov.keys()) == {"nominal_50", "nominal_80", "nominal_90", "nominal_95"}
    for lvl in cov.values():
        assert 0 <= lvl["empirical_coverage"] <= 1
        assert lvl["raw_lo"] < lvl["raw_hi"]
