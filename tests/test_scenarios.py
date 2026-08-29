import importlib.util
import sys
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "scenarios", Path(__file__).resolve().parents[1] / "precompute" / "scenarios.py")
scenarios = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scenarios)


def test_scenario_keys():
    assert set(scenarios.SCENARIOS) == {"clear", "cloudy",
                                        "high_metallicity", "hazy"}
    for cfg in scenarios.SCENARIOS.values():
        assert {"cloud_model", "cloud_type", "cloud_params",
                "C_to_O", "log_Met"} <= set(cfg)


def test_wl_grid():
    """Assert the POSEIDON wl_grid_constant_R convention, not a bit-exact
    formula: N points = round(ln(wl_max/wl_min) * R) (POSEIDON.core uses
    N_wl = round(...), evenly spaced in log-wavelength, with both endpoints
    forced exactly to wl_min/wl_max). This matches POSEIDON's own
    construction (verified against POSEIDON.core.wl_grid_constant_R in
    precompute/smoke_poseidon.py, Task 8) rather than the point count or
    per-point values of the pre-Task-8 make_wl_grid()."""
    wl = scenarios.make_wl_grid()

    n_expected = round(np.log(scenarios.WL_MAX_UM / scenarios.WL_MIN_UM)
                       * scenarios.R_MODEL)
    assert len(wl) == n_expected

    # Endpoints are forced exactly, not merely close.
    assert wl[0] == scenarios.WL_MIN_UM
    assert wl[-1] == scenarios.WL_MAX_UM

    # Constant spacing in log-wavelength (POSEIDON builds this via
    # np.linspace in log-space, so the diffs are constant up to floating
    # point, but not necessarily bit-identical to 1/R_MODEL since the
    # endpoints get overwritten after the linspace is built).
    log_diffs = np.diff(np.log(wl))
    np.testing.assert_allclose(log_diffs, log_diffs.mean(), rtol=1e-9)
