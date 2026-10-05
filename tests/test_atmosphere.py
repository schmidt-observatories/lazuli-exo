import numpy as np
import pytest

from conftest import ZETA0, ZETA_AMP
from terminator import atmosphere as atm

K_B, M_H, G = 1.380649e-23, 1.6735575e-27, 6.674e-11
R_J, M_J, R_SUN = 7.1492e7, 1.898e27, 6.957e8
M_G10 = 10.0 * R_J ** 2 / G / M_J          # mass giving g = 10 m/s^2 at 1 R_J (fiducial node, shift 0)
HJ = dict(T_eq_K=1000.0, R_p_RJ=1.0, M_p_MJ=M_G10, R_s_RSun=1.0)


def radius_m(depth, R_s_RSun):
    return np.sqrt(depth) * R_s_RSun * R_SUN


def test_node_recovery_shape_and_median_radius(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    depth, info = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    g = 10.0
    H = K_B * 1000.0 / (2.3 * M_H * g)
    np.testing.assert_allclose(info["g_ms2"], g)
    assert abs(info["zeta_shift"]) < 1e-9
    np.testing.assert_allclose(info["H_m"], H)
    band = (wl >= 0.40) & (wl <= 1.70)
    # R_p is the band-median transit radius (fixture: float32 zeta -> ~1e-7 rel)
    np.testing.assert_allclose(np.median(radius_m(depth[band], 1.0)), R_J, rtol=1e-6)
    # exact hydrostatic mapping: 1/r linear in zeta, R = R_p / (1 - h (zeta - zeta_ref)),
    # evaluated on the stored (float16) node values
    from terminator import tables
    node = tables.load_zeta_grid(zeta_fixture)["zeta"][2, 1, 1, 0].astype(float)   # T=1000, met=0, co=0.55, x=0
    zeta_eff = node + info["zeta_shift"]
    expect = R_J / (1 - (H / R_J) * (zeta_eff - info["zeta_ref"]))
    np.testing.assert_allclose(radius_m(depth, 1.0), expect, rtol=1e-6)


def test_doubling_teq_doubles_amplitude(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    d1, _ = atm.transmission_depth(wl, **{**HJ, "T_eq_K": 600.0}, tables_dir=zeta_fixture)
    d2, _ = atm.transmission_depth(wl, **{**HJ, "T_eq_K": 1200.0}, tables_dir=zeta_fixture)
    a1 = np.ptp(radius_m(d1, 1.0)); a2 = np.ptp(radius_m(d2, 1.0))
    np.testing.assert_allclose(a2 / a1, 2.0, rtol=2e-3)   # exact mapping: O(h) correction, h ~ 2e-3


def test_mu_default_and_override(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    _, i0 = atm.transmission_depth(wl, **HJ, log_met=1.0, tables_dir=zeta_fixture)
    assert i0["mu_source"] == "chemistry" and abs(i0["mu_used"] - 2.8) < 1e-5
    _, i1 = atm.transmission_depth(wl, **HJ, log_met=1.0, mu=18.0, tables_dir=zeta_fixture)
    assert i1["mu_source"] == "user" and i1["mu_used"] == 18.0
    assert i1["H_m"] < i0["H_m"]


def test_cloud_deck_monotonic_and_deep_deck_is_clear(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    clear, _ = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    amps = []
    for logp in (1.0, -3.0, -4.0, -5.0, -8.0):
        d, _ = atm.transmission_depth(wl, **HJ, log_P_cloud_bar=logp, tables_dir=zeta_fixture)
        amps.append(np.ptp(d))
    np.testing.assert_allclose(amps[0], np.ptp(clear))     # deck below photosphere -> clear
    assert all(a >= b for a, b in zip(amps, amps[1:]))      # higher deck -> flatter
    assert amps[-1] < 1e-12                                 # deck above everything -> flat


def test_haze_steepens_blue_end(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    clear, _ = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    hazy, _ = atm.transmission_depth(wl, **HJ, log_a_haze=3.0, haze_slope=-4.0, tables_dir=zeta_fixture)
    blue = wl < 0.5; red = wl > 1.5
    assert (hazy[blue].mean() - hazy[red].mean()) > (clear[blue].mean() - clear[red].mean())


def test_cold_planet_clamps_chemistry_but_not_H(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    _, info = atm.transmission_depth(wl, T_eq_K=250.0, R_p_RJ=0.2, M_p_MJ=0.03, R_s_RSun=0.4,
                                     tables_dir=zeta_fixture)
    assert info["T_clamped"] is True and info["T_chem_K"] == 300.0
    g = G * 0.03 * M_J / (0.2 * R_J) ** 2
    np.testing.assert_allclose(info["H_m"], K_B * 250.0 / (2.3 * M_H * g))


def test_ranges_raise(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    with pytest.raises(ValueError, match=r"log_met"):
        atm.transmission_depth(wl, **HJ, log_met=4.0, tables_dir=zeta_fixture)
    with pytest.raises(ValueError, match=r"c_to_o"):
        atm.transmission_depth(wl, **HJ, c_to_o=2.0, tables_dir=zeta_fixture)
    with pytest.raises(ValueError, match=r"wavelength"):
        atm.transmission_depth(np.array([0.30, 1.0]), **HJ, tables_dir=zeta_fixture)
    with pytest.raises(ValueError, match=r"wavelength"):
        atm.transmission_depth(np.array([1.0, 1.86]), **HJ, tables_dir=zeta_fixture)
    # the grid's nominal edge (one bin beyond the last bin centre) is accepted
    d, _ = atm.transmission_depth(np.array([0.35, 1.8505]), **HJ, tables_dir=zeta_fixture)
    assert np.isfinite(d).all()
    with pytest.raises(ValueError, match=r"T_eq_K"):
        atm.transmission_depth(wl, **{**HJ, "T_eq_K": 0.0}, tables_dir=zeta_fixture)


def test_puffy_atmosphere_uses_exact_hydrostatic_mapping(zeta_fixture):
    """g ~ 1/r^2: the high-zeta side extends more than H*dzeta, the low side less."""
    from terminator import tables
    wl = atm.grid_wavelengths(zeta_fixture)
    puffy = dict(T_eq_K=1500.0, R_p_RJ=1.0, M_p_MJ=0.05, R_s_RSun=1.0)   # g ~ 1.2, H/R_p ~ 0.06
    d, info = atm.transmission_depth(wl, **puffy, tables_dir=zeta_fixture)
    assert info["H_over_Rp"] > 0.05
    node = tables.load_zeta_grid(zeta_fixture)["zeta"][3, 1, 1, 0].astype(float)   # T=2000 node, x=0
    dz = node + info["zeta_shift"] - info["zeta_ref"]
    h = info["H_over_Rp"]
    r = radius_m(d, 1.0)
    np.testing.assert_allclose(r, R_J / (1 - h * dz), rtol=1e-6)
    linear = R_J * (1 + h * dz)
    assert (r[dz > 0.5] > linear[dz > 0.5]).all() and (r[dz < -0.5] > linear[dz < -0.5]).all()
    assert (r.max() - R_J) / (R_J - r.min()) > (dz.max() / -dz.min()) * 1.05


def test_unbound_atmosphere_raises(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    with pytest.raises(ValueError, match="unbound"):
        atm.transmission_depth(wl, T_eq_K=3000.0, R_p_RJ=1.5, M_p_MJ=0.005, R_s_RSun=1.0,
                               tables_dir=zeta_fixture)


def test_column_axis_interpolation(zeta_fixture):
    """A small planet between the two fiducial nodes blends their predictions."""
    from conftest import ZETA_NODE1_OFFSET
    from terminator import tables
    wl = atm.grid_wavelengths(zeta_fixture)
    g = tables.load_zeta_grid(zeta_fixture)
    x1 = g["meta"]["fiducials"][1]["x_column"]
    # choose M_p so that g = 10 (same as the fiducials) at R_p = 0.3 R_J -> shift = 0.5 ln 0.3
    R_p = 0.3 * R_J
    M_p_MJ = 10.0 * R_p ** 2 / G / M_J
    d, info = atm.transmission_depth(wl, T_eq_K=1000.0, R_p_RJ=0.3, M_p_MJ=M_p_MJ, R_s_RSun=0.5,
                                     tables_dir=zeta_fixture)
    np.testing.assert_allclose(info["zeta_shift"], 0.5 * np.log(0.3), rtol=1e-6)
    t = info["zeta_shift"] / x1                     # fraction of the way to node 1
    node0 = g["zeta"][2, 1, 1, 0].astype(float)
    node1 = g["zeta"][2, 1, 1, 1].astype(float)
    np.testing.assert_allclose(node1 - node0, ZETA_NODE1_OFFSET, atol=0.02)   # float16
    expect_zeta = (1 - t) * (node0 + info["zeta_shift"]) + t * (node1 + info["zeta_shift"] - x1)
    zeta_ref = np.median(expect_zeta[(wl >= 0.4) & (wl <= 1.7)])
    expect_r = R_p / (1 - (info["H_m"] / R_p) * (expect_zeta - zeta_ref))
    np.testing.assert_allclose(radius_m(d, 0.5), expect_r, rtol=1e-6)
    # a 1 R_J planet at g=10 sits on node 0 exactly and ignores node 1
    d0, i0 = atm.transmission_depth(wl, T_eq_K=1000.0, R_p_RJ=1.0,
                                    M_p_MJ=10.0 * R_J ** 2 / G / M_J, R_s_RSun=1.0,
                                    tables_dir=zeta_fixture)
    assert abs(i0["zeta_shift"]) < 1e-9
