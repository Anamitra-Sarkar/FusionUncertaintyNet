"""Tests for isotonic recalibration — synthetic fixtures, no GPU/weights.

Verifies the real statistical logic that training/kaggle/fit_interval_isotonic.py
applies on Modal against the real checkpoint: PIT via Normal(pred, var=k*theta^2),
IsotonicRegression on FIT PIT -> empirical rank, inverse to calibrated intervals,
and the honest TEST improvement. Here we use synthetic (pred,k,theta,true) with
a KNOWN miscalibration pattern (true std = factor * predicted std) and confirm
the isotonic fit measurably reduces interval_calibration_error on held-out
synthetic data versus the raw miscalibrated baseline.

These are synthetic sanity checks, not the real Kaggle number — the real number
is measured on Modal against the real checkpoint and held-out AFDB proteins.
"""
import sys
import os
import importlib.util
from pathlib import Path

import numpy as np

# Load fit_interval_isotonic as module via spec (keeps repo conventions: no install needed)
_spec = importlib.util.spec_from_file_location(
    "fit_interval_isotonic",
    Path(__file__).resolve().parents[1] / "training/kaggle/fit_interval_isotonic.py",
)
iso_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iso_mod)

# Also load eval's compute_interval_coverage for baseline comparisons
_eval_spec = importlib.util.spec_from_file_location(
    "eval_interval_calibration",
    Path(__file__).resolve().parents[1] / "training/kaggle/eval_interval_calibration.py",
)
eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(eval_mod)


def _synthetic_data(n, factor: float, seed: int = 0):
    """
    Generate synthetic miscalibrated data where true std = factor * predicted std.
    factor > 1 => predicted intervals too narrow (under-cover raw).
    factor < 1 => predicted intervals too wide (over-cover raw) — the real model's pattern.
    """
    rng = np.random.default_rng(seed)
    y_pred = rng.uniform(40, 80, size=n)
    # k>=1, theta>=0.1 per edr.py constraints; pick moderate ranges
    k = rng.uniform(1.0, 5.0, size=n)
    theta = rng.uniform(0.5, 2.0, size=n)
    var = k * (theta ** 2)
    std_pred = np.sqrt(np.maximum(var, 1e-6))
    std_true = std_pred * factor
    y_true = y_pred + rng.normal(0, 1, size=n) * std_true
    return y_true, y_pred, k, theta


def test_compute_pit_at_pred_is_half():
    y_true = np.array([50.0, 60.0])
    y_pred = np.array([50.0, 60.0])
    k = np.array([2.0, 3.0])
    theta = np.array([1.0, 1.0])
    u = iso_mod.compute_pit(y_true, y_pred, k, theta)
    # When truth == pred, PIT = Phi(0) = 0.5
    assert np.allclose(u, 0.5, atol=1e-9), f"PIT at pred should be 0.5, got {u}"


def test_compute_pit_matches_eval_var_logic():
    """compute_pit must use var=k*theta^2 identical to compute_interval_coverage."""
    rng = np.random.default_rng(1)
    y_pred = rng.uniform(0, 100, size=10)
    y_true = rng.uniform(0, 100, size=10)
    k = rng.uniform(1, 3, size=10)
    theta = rng.uniform(0.2, 1.5, size=10)
    from scipy.stats import norm
    var = k * (theta ** 2)
    std = np.sqrt(np.maximum(var, 1e-6))
    expected = np.clip(norm.cdf((y_true - y_pred) / std), 1e-9, 1 - 1e-9)
    got = iso_mod.compute_pit(y_true, y_pred, k, theta)
    assert np.allclose(got, expected, atol=1e-12)


def test_fit_isotonic_monotone_and_clipped():
    u = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    ir, u_sorted, y_emp, y_hat = iso_mod.fit_isotonic_map(u)
    # y_hat must be monotone non-decreasing
    assert np.all(y_hat[1:] >= y_hat[:-1] - 1e-9), f"y_hat not monotone: {y_hat}"
    # Boundaries clipped to [0,1]
    assert y_hat.min() >= 0 and y_hat.max() <= 1
    # Predict out-of-bounds clips
    assert ir.predict([0.0])[0] >= 0
    assert ir.predict([1.0])[0] <= 1
    # Sorted input preserved
    assert np.all(u_sorted == np.sort(u))


def test_isotonic_reduces_error_on_synthetic_undercover():
    """
    Synthetic sanity check: true std is 3x predicted std => raw intervals under-cover
    (e.g. 50%-nominal captures far less than 50%). Isotonic fit on held-out data
    should measurably reduce interval_calibration_error.

    This is NOT the real Kaggle number — it proves the statistical method works
    on known miscalibration. The real number is measured on Modal/T4 against
    the real checkpoint (docs/ interval-calibration-result-2026-08-30.md).
    """
    n_fit, n_test = 5000, 5000
    y_true_fit, y_pred_fit, k_fit, theta_fit = _synthetic_data(n_fit, factor=3.0, seed=42)
    y_true_test, y_pred_test, k_test, theta_test = _synthetic_data(n_test, factor=3.0, seed=99)

    # Raw errors
    _, raw_err_fit = eval_mod.compute_interval_coverage(y_pred_fit, y_true_fit, k_fit, theta_fit)
    _, raw_err_test = eval_mod.compute_interval_coverage(y_pred_test, y_true_test, k_test, theta_test)
    # Fit isotonic on FIT only
    u_fit = iso_mod.compute_pit(y_true_fit, y_pred_fit, k_fit, theta_fit)
    ir, u_sorted, y_emp, y_hat = iso_mod.fit_isotonic_map(u_fit)
    # Calibrated coverage on held-out TEST (honest)
    _, iso_err_test = iso_mod.compute_calibrated_interval_coverage(
        y_true_test, y_pred_test, k_test, theta_test, ir, u_sorted, y_hat
    )
    print(f"[synthetic undercover] raw_fit_err={raw_err_fit:.4f} raw_test_err={raw_err_test:.4f} iso_test_err={iso_err_test:.4f}")
    # Isotonic should not make things worse; expect meaningful improvement
    assert iso_err_test < raw_err_test, f"isotonic did not improve: {iso_err_test:.4f} >= {raw_err_test:.4f}"
    # With 5k fit points and a pure scale miscalibration, isotonic should roughly halve error or better
    # (allow some tolerance for randomness)
    assert iso_err_test < raw_err_test * 0.7, f"expected at least 30% reduction, got raw {raw_err_test:.4f} -> iso {iso_err_test:.4f}"
    # Honest test error should be small (well below raw), ideally <0.05 for this simple synthetic pattern
    assert iso_err_test < 0.07, f"isotonic test error {iso_err_test:.4f} not <0.07 on synthetic"


def test_isotonic_reduces_error_on_synthetic_overcover():
    """
    Real model is over-conservative (intervals too wide, raw covers 91% at 50%-nominal).
    Test the same logic with factor=0.3 (predicted std 3x too large) => raw over-covers.
    Isotonic should also repair this.
    """
    n_fit, n_test = 8000, 8000
    y_true_fit, y_pred_fit, k_fit, theta_fit = _synthetic_data(n_fit, factor=0.33, seed=7)
    y_true_test, y_pred_test, k_test, theta_test = _synthetic_data(n_test, factor=0.33, seed=8)

    _, raw_err_test = eval_mod.compute_interval_coverage(y_pred_test, y_true_test, k_test, theta_test)
    u_fit = iso_mod.compute_pit(y_true_fit, y_pred_fit, k_fit, theta_fit)
    ir, u_sorted, y_emp, y_hat = iso_mod.fit_isotonic_map(u_fit)
    _, iso_err_test = iso_mod.compute_calibrated_interval_coverage(
        y_true_test, y_pred_test, k_test, theta_test, ir, u_sorted, y_hat
    )
    print(f"[synthetic overcover] raw_test_err={raw_err_test:.4f} iso_test_err={iso_err_test:.4f}")
    assert iso_err_test < raw_err_test
    assert iso_err_test < raw_err_test * 0.7
    assert iso_err_test < 0.07


def test_calibrated_intervals_are_coherent():
    """Calibrated lo < hi for all nominal levels, and wider nominal => wider interval."""
    rng = np.random.default_rng(123)
    n = 2000
    y_true, y_pred, k, theta = _synthetic_data(n, factor=2.0, seed=123)
    n_fit = 1000
    u_fit = iso_mod.compute_pit(y_true[:n_fit], y_pred[:n_fit], k[:n_fit], theta[:n_fit])
    ir, u_sorted, y_emp, y_hat = iso_mod.fit_isotonic_map(u_fit)
    # Check on remaining as test-like
    cov, err = iso_mod.compute_calibrated_interval_coverage(
        y_true[n_fit:], y_pred[n_fit:], k[n_fit:], theta[n_fit:], ir, u_sorted, y_hat
    )
    # Coverage must have all 4 nominal levels
    for lvl in ("nominal_50", "nominal_80", "nominal_90", "nominal_95"):
        assert lvl in cov
        assert 0 <= cov[lvl]["empirical_coverage"] <= 1
        assert cov[lvl]["raw_lo"] < cov[lvl]["raw_hi"], f"{lvl} raw_lo >= raw_hi"
    # Wider nominal should have higher empirical coverage (monotone)
    emps = [cov[f"nominal_{x}"]["empirical_coverage"] for x in (50, 80, 90, 95)]
    assert emps == sorted(emps), f"empirical coverages not monotone: {emps}"


def test_fit_temperature_also_improves_synthetic():
    """
    On the same synthetic scale miscalibration, scalar temperature should also help
    (but may not fully clear shape issues). This ensures our fit_temperature helper
    is consistent with isotonic baseline reporting.
    Use factor=2.0 so optimum T=2.0 lies within the [0.05,2.0] grid used by
    fit_interval_temperature.py's fit_temperature (which caps at 2.0).
    """
    n_fit, n_test = 5000, 5000
    y_true_fit, y_pred_fit, k_fit, theta_fit = _synthetic_data(n_fit, factor=2.0, seed=55)
    y_true_test, y_pred_test, k_test, theta_test = _synthetic_data(n_test, factor=2.0, seed=56)

    _, raw_err_test = eval_mod.compute_interval_coverage(y_pred_test, y_true_test, k_test, theta_test)
    best_t, _ = iso_mod.fit_temperature(y_true_fit, y_pred_fit, k_fit, theta_fit)
    _, temp_err_test = eval_mod.compute_interval_coverage(y_pred_test, y_true_test, k_test, theta_test * best_t)
    print(f"[synthetic temp] raw {raw_err_test:.4f} -> temp(T={best_t:.3f}) {temp_err_test:.4f}")
    assert 1.5 < best_t <= 2.1, f"with true std 2x predicted, best T should be ~2, got {best_t}"
    assert temp_err_test < raw_err_test
    assert temp_err_test < 0.08  # temperature alone should also roughly fix pure scale error


def test_fit_isotonic_empty_raises():
    try:
        iso_mod.fit_isotonic_map(np.array([]))
        assert False, "should have raised ValueError on empty"
    except ValueError:
        pass


def test_invert_isotonic_extremes_clipped():
    """Inverse map at 0 and 1 should give valid PIT values in (0,1) not inf."""
    u = np.linspace(0.01, 0.99, 500)
    ir, u_sorted, y_emp, y_hat = iso_mod.fit_isotonic_map(u)
    for q in (0.0, 0.025, 0.25, 0.5, 0.75, 0.975, 1.0):
        raw = iso_mod._invert_isotonic(ir, u_sorted, y_hat, q)
        assert 0 < float(raw) < 1, f"invert({q}) gave {raw} not in (0,1)"
