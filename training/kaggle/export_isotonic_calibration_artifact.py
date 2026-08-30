#!/usr/bin/env python3
"""
Export isotonic calibration artifact companion.

This is a small companion to training/kaggle/fit_interval_isotonic.py.
Its job is to let the coordinator re-serialize or re-upload the
calibration-isotonic-v1.json artifact WITHOUT re-running the full
ESM2/ProtT5 inference, e.g. if the fit was already done and the
breakpoints JSON exists locally.

When run as `python training/kaggle/export_isotonic_calibration_artifact.py
--artifact eval_results/calibration-isotonic-v1.json` it just re-validates
the artifact (checks breakpoints monotone, round-trip error via pure-numpy
helper) and attempts the same HfApi upload used by fit_interval_isotonic.py.

Normal workflow: `fit_interval_isotonic.py` already serializes and uploads
the artifact in one pass.  This script is for the "already fitted, just
re-export / re-upload" case and for local validation without GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from eval_interval_calibration import _try_load_kaggle_token  # type: ignore
except Exception:
    def _try_load_kaggle_token():  # type: ignore
        return os.getenv("HF_TOKEN")


def validate_artifact(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    assert "breakpoints" in data, "missing breakpoints"
    bp = data["breakpoints"]
    x = bp.get("x") or bp.get("nominal")
    y = bp.get("y") or bp.get("empirical")
    assert x is not None and y is not None, "breakpoints need x/y"
    assert len(x) == len(y) >= 2, "need >=2 breakpoints"
    import numpy as np

    x_a = np.array(x, dtype=float)
    y_a = np.array(y, dtype=float)
    assert np.all(np.diff(x_a) >= -1e-9), "x not increasing"
    assert np.all(np.diff(y_a) >= -1e-9), "y not monotone (isotonic violation)"
    assert 0 <= x_a[0] <= 1 and 0 <= y_a[0] <= 1
    # optional: check metadata
    assert data.get("version") == "calibration-isotonic-v1"
    assert "test_interval_calibration_error" in data
    print(f"[export] validated {path}: {len(x)} breakpoints, TEST error {data['test_interval_calibration_error']}")
    # try pure-numpy round-trip via calibration helper
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "../../backend-heavy"))
        from fusionuncertaintynet.calibration import apply_isotonic_calibration

        _k, _th, *_ = apply_isotonic_calibration([50.0], [2.0], [1.0], data["breakpoints"])
        print(f"[export] apply_isotonic_calibration smoke: theta 1.0 -> {float(_th[0]):.4f}")
    except Exception as e:
        print(f"[export] WARN calibration helper check skipped: {e}")
    return data


def upload_artifact(local_path: str, repo_id: str = "bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree") -> bool:
    token = None
    try:
        token = _try_load_kaggle_token()
    except Exception:
        pass
    if not token:
        token = os.getenv("HF_TOKEN", "").strip() or None
    if not token:
        print("[export] HF_TOKEN not set — skipping upload (local artifact still valid)")
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        try:
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception:
            pass
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo="calibration/calibration-isotonic-v1.json",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[export] Uploaded {local_path} -> {repo_id}:calibration/calibration-isotonic-v1.json")
        return True
    except Exception as e:
        print(f"[export] HF upload failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Validate and re-upload isotonic calibration artifact")
    ap.add_argument("--artifact", default="calibration-isotonic-v1.json", help="Path to calibration-isotonic-v1.json")
    ap.add_argument("--hf-repo", default="bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree")
    ap.add_argument("--no-upload", action="store_true", help="Only validate, do not upload")
    args = ap.parse_args()
    # resolve artifact path: try provided, then eval_results/, then cwd
    cand = args.artifact
    if not os.path.isfile(cand):
        for alt in ["eval_results/calibration-isotonic-v1.json", "calibration-isotonic-v1.json", os.path.join(os.path.dirname(__file__), "../../calibration-isotonic-v1.json")]:
            if os.path.isfile(alt):
                cand = alt
                break
    if not os.path.isfile(cand):
        print(f"[export] artifact not found: {args.artifact} (tried alternates)")
        sys.exit(1)
    validate_artifact(cand)
    if not args.no_upload:
        upload_artifact(cand, args.hf_repo)


if __name__ == "__main__":
    main()
