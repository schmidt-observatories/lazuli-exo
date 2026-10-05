import math
import pytest

from lazuli_exo import targets

REQUIRED = ["R_p_RJ", "M_p_MJ", "R_s_RSun", "T_eff_K", "log_g_star",
            "Met_star", "T_eq_K", "period_d", "t14_s", "depth",
            "a_over_rs", "inc_deg", "ecc", "w_deg", "J_mag"]


def test_fourteen_planets():
    assert len(targets.PLANET_NAMES) == 14
    assert "WASP-39 b" in targets.PLANET_NAMES
    assert "TRAPPIST-1 e" in targets.PLANET_NAMES


def test_all_fields_finite():
    for name in targets.PLANET_NAMES:
        t = targets.get_target(name)
        for key in REQUIRED:
            assert math.isfinite(t[key]), f"{name}.{key} = {t[key]}"


def test_physical_sanity():
    for name in targets.PLANET_NAMES:
        t = targets.get_target(name)
        assert 0.001 < t["depth"] < 0.1, name        # 0.1%–10% transits
        assert 60 * 10 < t["t14_s"] < 60 * 60 * 12, name
        assert 1 < t["a_over_rs"] < 300, name
        assert 2000 < t["T_eff_K"] < 7000, name
        assert 4 < t["J_mag"] < 13, name


def test_get_target_unknown():
    with pytest.raises(KeyError, match="WASP-39 b"):
        targets.get_target("Kepler-999x b")
