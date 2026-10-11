"""Unit tests for precompute/freeze_targets.py entry building (no network)."""
import importlib.util
import math
from pathlib import Path

import pytest

from lazuli_exo import targets
from lazuli_exo.atmosphere import R_J, R_SUN

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


def test_missing_omega_forces_circular_orbit():
    """NULL w with nonzero e would put the transit at periastron (#7)."""
    entry = freeze_targets.build_entry(
        _row(pl_orblper=NAN, pl_orbeccen=0.3), "Testar-1 b")
    assert entry["w_deg"] == 90.0
    assert entry["ecc"] == 0.0
    assert entry["ecc_note"] == "w unavailable; circular assumed"
    assert "ecc_note" not in freeze_targets.build_entry(_row(), "Testar-1 b")


def test_k_frozen_from_pl_ratror():
    entry = freeze_targets.build_entry(_row(), "Testar-1 b")
    assert entry["k_source"] == "pl_ratror"
    assert entry["R_p_RJ"] == pytest.approx(0.1 * R_SUN / R_J)
    assert entry["R_p_archive_RJ"] == 1.0
    assert entry["depth"] == pytest.approx(0.01)
    assert entry["T_eq_conv_K"] == pytest.approx(5700.0 * math.sqrt(0.5 / 9.0))
    assert entry["T_eq_K"] == 1200.0           # archive value kept


def test_k_falls_back_to_archive_radii():
    entry = freeze_targets.build_entry(_row(pl_ratror=NAN), "Testar-1 b")
    assert entry["k_source"] == "pl_radj/st_rad"
    assert entry["R_p_RJ"] == pytest.approx(1.0)
    assert entry["depth"] == pytest.approx((R_J / R_SUN) ** 2)


def _k(t):
    return t["R_p_RJ"] * R_J / (t["R_s_RSun"] * R_SUN)


@pytest.mark.parametrize("name", targets.PLANET_NAMES)
def test_registry_depth_is_k_squared(name):
    t = targets.get_target(name)
    assert t["depth"] == pytest.approx(_k(t) ** 2, rel=1e-12)
    assert t["k_source"] == "pl_ratror"
    assert t["T_eq_conv_K"] == pytest.approx(
        t["T_eff_K"] * math.sqrt(0.5 / t["a_over_rs"]))


def test_registry_wasp39b_uses_pl_ratror():
    t = targets.get_target("WASP-39 b")
    assert _k(t) == pytest.approx(0.1457, abs=1e-3)
    assert t["R_p_archive_RJ"] == 1.279
    assert t["R_p_RJ"] == pytest.approx(0.1457 * 0.939 * R_SUN / R_J)


def test_registry_lhs1140b_circular():
    t = targets.get_target("LHS 1140 b")
    assert t["ecc"] == 0.0 and t["w_deg"] == 90.0
    assert t["ecc_note"] == "w unavailable; circular assumed"


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
