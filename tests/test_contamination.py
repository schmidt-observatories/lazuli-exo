import numpy as np
import pytest

import lazuli_exo
from lazuli_exo import star
from lazuli_exo.simulate import simulate
from test_simulate import PARAMS, both_grids, fake_depth_error, sim_mod  # noqa: F401

WL = np.array([0.45, 0.8, 1.4])


def test_zero_fractions_are_exactly_one(phoenix_fixture):
    eps, info = star.contamination(WL, 3100.0, 5.0, 0.0, tables_dir=phoenix_fixture)
    assert (eps == 1.0).all() and info["eps_min"] == info["eps_max"] == 1.0
    assert info["T_spot_K"] == pytest.approx(0.86 * 3100.0)
    assert info["T_fac_K"] == 3200.0


def test_cool_spot_is_darker_and_bluer():
    """Shipped grid: spot on a 3100 K star raises depth, more so in the blue."""
    eps, info = star.contamination(WL, 3100.0, 5.0, 0.0, f_spot=0.03)
    assert (eps > 1).all() and eps[0] > eps[-1]
    assert info["clamped_teff"] == [] and info["eps_max"] > info["eps_min"]
    _, info_fgk = star.contamination(WL, 5000.0, 4.5, 0.0, f_spot=0.03)
    assert info_fgk["T_spot_K"] == pytest.approx(4000.0)


def test_fraction_sum_must_be_below_one(phoenix_fixture):
    with pytest.raises(ValueError, match="f_spot"):
        star.contamination(WL, 3100.0, 5.0, 0.0, f_spot=0.6, f_fac=0.4,
                           tables_dir=phoenix_fixture)
    with pytest.raises(ValueError, match="f_spot"):
        star.contamination(WL, 3100.0, 5.0, 0.0, f_spot=-0.1, tables_dir=phoenix_fixture)


def test_blackbody_floor_warns(phoenix_fixture):
    with pytest.warns(UserWarning, match="PHOENIX"):
        _, info = star.contamination(WL, 2100.0, 5.0, 0.0, f_spot=0.1,
                                     tables_dir=phoenix_fixture)
    assert info["clamped_teff"] == ["spot"]


def test_gj1214b_giant_spots_via_observe():
    """Review (2026-10-10, area G): 282 ppm ptp at R=100 for f_spot = 0.03."""
    o0 = lazuli_exo.observe("GJ 1214 b", "clear", R=100)
    o1 = lazuli_exo.observe("GJ 1214 b", "clear", R=100, f_spot=0.03)
    ptp_ppm = np.ptp(o1.depth - o0.depth) * 1e6
    assert 200 < ptp_ppm < 400
    assert o1.flat_depth == o0.flat_depth
    assert o1.meta["atmosphere"]["tlse"]["f_spot"] == 0.03
    assert o0.meta["atmosphere"]["tlse"]["eps_max"] == 1.0


def test_simulate_accepts_tlse_keys(both_grids, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sim_mod, "depth_error", fake_depth_error)
    base = simulate(PARAMS, R=50, tables_dir=both_grids)
    off = simulate({**PARAMS, "f_spot": 0.0, "f_fac": 0.0, "T_spot_K": None,
                    "T_fac_K": None}, R=50, tables_dir=both_grids)
    np.testing.assert_array_equal(off.depth, base.depth)
    assert off.meta["atmosphere"]["tlse"]["f_spot"] == 0.0
    on = simulate({**PARAMS, "f_spot": 0.1}, R=50, tables_dir=both_grids)
    assert not np.array_equal(on.depth, base.depth)
