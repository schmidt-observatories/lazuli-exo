"""Unit tests for precompute/freeze_targets.py entry building (no network)."""
import importlib.util
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "freeze_targets",
    Path(__file__).resolve().parents[1] / "precompute" / "freeze_targets.py")
freeze_targets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze_targets)

NAN = float("nan")


def _row(**overrides):
    row = {
        "pl_radj": 1.0, "pl_bmassj": 1.0, "pl_orbper": 3.0,
        "pl_trandur": 3.0, "pl_eqt": 1200.0, "pl_ratror": 0.1,
        "pl_ratdor": 9.0, "pl_orbincl": 88.0, "pl_orbeccen": 0.0,
        "pl_orblper": 90.0, "st_rad": 1.0, "st_teff": 5700.0,
        "st_logg": 4.4, "st_met": 0.0, "sy_jmag": 9.0,
    }
    row.update(overrides)
    return row


def test_omega_zero_is_not_replaced_by_default():
    entry = freeze_targets.build_entry(_row(pl_orblper=0.0), "Testar-1 b")
    assert entry["w_deg"] == 0.0


def test_missing_omega_defaults_to_90():
    entry = freeze_targets.build_entry(_row(pl_orblper=NAN), "Testar-1 b")
    assert entry["w_deg"] == 90.0


def test_fallback_duration_uses_eccentric_formula():
    """With pl_trandur missing, the computed t14 must carry the
    sqrt(1-e^2)/(1+e sin w) factor and the eccentric impact parameter."""
    e, w_deg = 0.3, 45.0
    row = _row(pl_trandur=NAN, pl_orbeccen=e, pl_orblper=w_deg)
    entry = freeze_targets.build_entry(row, "Testar-1 b")

    w = math.radians(w_deg)
    a, inc, k = 9.0, math.radians(88.0), 0.1
    fac = math.sqrt(1.0 - e**2) / (1.0 + e * math.sin(w))
    b = a * math.cos(inc) * (1.0 - e**2) / (1.0 + e * math.sin(w))
    arg = math.sqrt((1.0 + k) ** 2 - b**2) / (a * math.sin(inc))
    expected = 3.0 * 86400.0 / math.pi * math.asin(min(arg, 1.0)) * fac
    assert entry["t14_s"] == pytest.approx(expected)
