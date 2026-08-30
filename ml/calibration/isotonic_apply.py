"""Re-export wrapper for backend-heavy calibration helper.

This file exists so training code that expects `ml/calibration/isotonic_apply.py`
and serving code that expects `fusionuncertaintynet.calibration` both share the
same pure-numpy implementation without duplication.

Import via either:
  from fusionuncertaintynet.calibration import apply_isotonic_calibration
  from ml.calibration.isotonic_apply import apply_isotonic_calibration
"""
import sys, os
# ensure backend-heavy is on path
_here = os.path.dirname(__file__)
_repo = os.path.abspath(os.path.join(_here, "../.."))
_bh = os.path.join(_repo, "backend-heavy")
if _bh not in sys.path:
    sys.path.insert(0, _bh)

from fusionuncertaintynet.calibration import (  # noqa: F401
    apply_isotonic_calibration,
    compute_calibrated_interval_coverage_via_breakpoints,
    _calibration_factor,
    _get_xy,
    _invert_breakpoints,
    _norm_ppf,
)
