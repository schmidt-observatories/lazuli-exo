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
    wl = scenarios.make_wl_grid()
    assert wl[0] == scenarios.WL_MIN_UM
    assert wl[-1] <= scenarios.WL_MAX_UM
    np.testing.assert_allclose(np.diff(np.log(wl)), 1.0 / scenarios.R_MODEL)
