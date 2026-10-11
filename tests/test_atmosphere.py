import numpy as np
import pytest

from conftest import ZETA0, ZETA_AMP
from lazuli_exo import atmosphere as atm

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
    from lazuli_exo import tables
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
    with pytest.warns(UserWarning, match="secondary"):     # mu > 15, see #39
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
    np.testing.assert_allclose(amps[0], np.ptp(clear), rtol=1e-6)   # deck far below -> clear
    assert all(a >= b for a, b in zip(amps, amps[1:]))      # higher deck -> flatter
    assert amps[-1] < 1e-3 * np.ptp(clear)                  # deck far above -> flat (e^-8 tail)


def test_haze_steepens_blue_end(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    clear, _ = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    hazy, _ = atm.transmission_depth(wl, **HJ, log_a_haze=3.0, haze_slope=-4.0, tables_dir=zeta_fixture)
    blue = wl < 0.5; red = wl > 1.5
    assert (hazy[blue].mean() - hazy[red].mean()) > (clear[blue].mean() - clear[red].mean())


def test_cold_planet_clamps_chemistry_but_not_H(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    with pytest.warns(UserWarning, match="grid floor"):
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
    from lazuli_exo import tables
    wl = atm.grid_wavelengths(zeta_fixture)
    puffy = dict(T_eq_K=1500.0, R_p_RJ=1.0, M_p_MJ=0.05, R_s_RSun=1.0)   # g ~ 1.2, H/R_p ~ 0.06
    d, info = atm.transmission_depth(wl, **puffy, tables_dir=zeta_fixture)
    assert info["H_over_Rp"] > 0.05
    dz = expected_zeta(zeta_fixture, (3, 1, 1), info) - info["zeta_ref"]    # T=2000 node
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


def expected_zeta(tdir, idx, info):
    """zeta linear in x through the stored nodes at grid index idx = (iT, imet, ico)."""
    from lazuli_exo import tables
    g = tables.load_zeta_grid(tdir)
    x1 = g["meta"]["fiducials"][1]["x_column"]
    node0 = g["zeta"][idx + (0,)].astype(float)
    node1 = g["zeta"][idx + (1,)].astype(float)
    return node0 + (node1 - node0) * (info["zeta_shift"] / x1)


def test_column_axis_interpolation_and_extrapolation(zeta_fixture):
    """zeta is linear in the slant-column variable through the two nodes."""
    from conftest import ZETA_X_SLOPE
    wl = atm.grid_wavelengths(zeta_fixture)
    band = (wl >= 0.4) & (wl <= 1.7)
    for R_p_RJ in (0.3, 3.0):            # inside (x = -0.60) and beyond (x = +0.55) the nodes
        R_p = R_p_RJ * R_J
        M_p_MJ = 10.0 * R_p ** 2 / G / M_J           # g = 10 -> x = 0.5 ln R_p
        d, info = atm.transmission_depth(wl, T_eq_K=1000.0, R_p_RJ=R_p_RJ, M_p_MJ=M_p_MJ,
                                         R_s_RSun=1.0, tables_dir=zeta_fixture)
        np.testing.assert_allclose(info["zeta_shift"], 0.5 * np.log(R_p_RJ), rtol=1e-6)
        np.testing.assert_allclose(info["column_slope_median"], ZETA_X_SLOPE, atol=0.03)
        z = expected_zeta(zeta_fixture, (2, 1, 1), info)
        expect_r = R_p / (1 - (info["H_m"] / R_p) * (z - np.median(z[band])))
        np.testing.assert_allclose(radius_m(d, 1.0), expect_r, rtol=1e-6)
    # a 1 R_J planet at g=10 sits on node 0 exactly and ignores node 1
    d0, i0 = atm.transmission_depth(wl, T_eq_K=1000.0, R_p_RJ=1.0,
                                    M_p_MJ=10.0 * R_J ** 2 / G / M_J, R_s_RSun=1.0,
                                    tables_dir=zeta_fixture)
    assert abs(i0["zeta_shift"]) < 1e-9


def test_ein_limits_and_deck_transition(zeta_fixture):
    from lazuli_exo.atmosphere import EULER_GAMMA, _ein
    np.testing.assert_allclose(_ein(1e-9), 1e-9, rtol=1e-6)
    np.testing.assert_allclose(_ein(1.0), 0.7965995992970531, rtol=1e-4)        # Ein(1)
    np.testing.assert_allclose(_ein(50.0), EULER_GAMMA + np.log(50.0), rtol=1e-4)
    np.testing.assert_allclose(_ein(1e9), EULER_GAMMA + np.log(1e9), rtol=1e-9)
    # a deck exactly at the clear band-median pressure: amplitude is reduced but the
    # partially transparent gas above the deck keeps ~half of the high side
    wl = atm.grid_wavelengths(zeta_fixture)
    clear, info = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    logp = -info["zeta_ref"] / np.log(10.0)
    decked, info_d = atm.transmission_depth(wl, **HJ, log_P_cloud_bar=logp, tables_dir=zeta_fixture)
    r_c, r_d = radius_m(clear, 1.0), radius_m(decked, 1.0)
    # zeta_eff >= zeta_deck everywhere: the deck radius is the floor (R_p is the
    # band-median radius, so it sits *above* the deck by (zeta_ref - zeta_d) H)
    h = info_d["H_m"] / R_J
    r_deck = R_J / (1 - h * (-logp * np.log(10.0) - info_d["zeta_ref"]))
    assert np.all(r_d >= r_deck * (1 - 1e-12))
    assert 0.3 < np.ptp(r_d) / np.ptp(r_c) < 0.7            # ~0.5 H of the 2 H survives


# --- haze / deck effective radius (issue #23) -------------------------------

def zeta_abs(depth, info, R_p_m):
    """zeta_eff from the depth by inverting the exact hydrostatic mapping."""
    return info["zeta_ref"] + (R_p_m / info["H_m"]) * (1 - R_p_m / radius_m(depth, 1.0))


def ein_exact(a):
    """Ein(a) = int_0^a (1 - e^-t)/t dt, Simpson on 20001 points (reference, numpy only)."""
    a = np.atleast_1d(np.asarray(a, float))
    t = np.linspace(0.0, 1.0, 20001)[None, :] * a[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(t > 0, -np.expm1(-t) / t, 1.0)
    h = a / 20000
    return h / 3 * (f[:, 0] + f[:, -1] + 4 * f[:, 1:-1:2].sum(1) + 2 * f[:, 2:-1:2].sum(1))


def gas_levels(tdir, info):
    """Effective-radius gas zeta and slant-column exponent as transmission_depth sees them."""
    from lazuli_exo import tables
    g = tables.load_zeta_grid(tdir)
    x1 = g["meta"]["fiducials"][1]["x_column"]
    node0 = g["zeta"][2, 1, 1, 0].astype(float)
    node1 = g["zeta"][2, 1, 1, 1].astype(float)
    slope = np.clip((node1 - node0) / x1, *atm.S_RANGE)
    return node0 + slope * info["zeta_shift"], slope


def test_haze_integral_reduces_to_old_formula_for_clear_and_gas_deck(zeta_fixture):
    """log_a_haze=-40 routes through the integral with a negligible haze: it must
    equal zeta_p (clear) and zeta_d + s Ein(exp((zeta_p - zeta_d)/s - gamma_E))
    (gas + deck) to 1e-6 H; without the haze keyword the old paths are bit-for-bit."""
    wl = atm.grid_wavelengths(zeta_fixture)
    clear, info = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    zeta_p, s = gas_levels(zeta_fixture, info)
    np.testing.assert_allclose(zeta_abs(clear, info, R_J), zeta_p, atol=1e-6)
    via_int, info_i = atm.transmission_depth(wl, **HJ, log_a_haze=-40.0, tables_dir=zeta_fixture)
    np.testing.assert_allclose(zeta_abs(via_int, info_i, R_J), zeta_p, atol=1e-6)

    logp = -4.0                              # deck inside the band's zeta range
    zeta_d = -logp * np.log(10.0)
    old = zeta_d + s * ein_exact(np.exp((zeta_p - zeta_d) / s - atm.EULER_GAMMA))
    decked, info_d = atm.transmission_depth(wl, **HJ, log_P_cloud_bar=logp, tables_dir=zeta_fixture)
    np.testing.assert_allclose(zeta_abs(decked, info_d, R_J), old, atol=1e-5)   # _ein table: 6e-6
    old_table = zeta_d + s * atm._ein(np.exp((zeta_p - zeta_d) / s - atm.EULER_GAMMA))
    np.testing.assert_allclose(zeta_abs(decked, info_d, R_J), old_table, atol=1e-9)
    via_int, info_i = atm.transmission_depth(wl, **HJ, log_P_cloud_bar=logp, log_a_haze=-40.0,
                                             tables_dir=zeta_fixture)
    np.testing.assert_allclose(zeta_abs(via_int, info_i, R_J), old, atol=1e-6)


def test_haze_dominated_effective_radius_sits_gamma_E_above_tau_one(zeta_fixture):
    """Where the haze sets tau the effective radius is gamma_E H above its tau_chord = 1
    level (exponential profile, exponent 1); logaddexp of the levels put it at tau = 1."""
    wl = atm.grid_wavelengths(zeta_fixture)
    hazy, info = atm.transmission_depth(wl, **HJ, log_a_haze=8.0, haze_slope=-4.0,
                                        tables_dir=zeta_fixture)
    kappa = 1e8 * atm.SIGMA_H2_RAYLEIGH_350NM * (wl / atm.HAZE_LAMBDA0_UM) ** -4.0
    zeta_h1 = -np.log(K_B * 1000.0 / (kappa * np.sqrt(2 * np.pi * R_J * info["H_m"])) / 1e5)
    zeta_p, _ = gas_levels(zeta_fixture, info)
    dom = zeta_h1 - zeta_p > 10.0            # gas tau < e^-12 at the haze level
    assert dom.sum() > 50
    np.testing.assert_allclose(zeta_abs(hazy, info, R_J)[dom] - zeta_h1[dom],
                               atm.EULER_GAMMA, atol=1e-4)
    assert np.all(np.logaddexp(zeta_p, zeta_h1)[dom] - zeta_h1[dom] < 1e-4)   # the old formula


def bruteforce_zeta_eff(R_ref, g_ref, T, mu, P_ref_bar, lnsig0, a, kappa, logP_cloud=None,
                        dz=0.02, n_u=600, u_max=12.0):
    """Exact chord integration (isothermal, hydrostatic, g ~ 1/r^2) of a gas with
    sigma = sigma0 (P/bar)^a plus a grey haze kappa (both per molecule), R_eff^2 =
    R_deck^2 + 2 int (1 - e^-tau) b db; returns -ln P(R_eff) and the thin-atmosphere
    tau_chord = 1 levels (zeta_g1, zeta_h1). Minimal port of review A's bf.py."""
    H0 = K_B * T / (mu * M_H * g_ref)
    lnP_ref = np.log(P_ref_bar)
    lnP = lambda r: lnP_ref + (R_ref / H0) * (R_ref / r - 1.0)                 # noqa: E731
    r_of = lambda lnp: 1.0 / (1.0 / R_ref + (H0 / R_ref ** 2) * (lnp - lnP_ref))  # noqa: E731
    lnn = np.log(1e5 / (K_B * T))                                           # molecules m^-3 per bar
    col = 0.5 * np.log(2 * np.pi * R_ref * H0)
    s = 1.0 / (1.0 + a)
    zeta_g1 = s * (lnsig0 + lnn + col - 0.5 * np.log(1 + a))
    zeta_h1 = np.log(kappa) + lnn + col
    zeta_bot = max(zeta_g1 - 12 * s, zeta_h1 - 12) if logP_cloud is None else -logP_cloud * np.log(10.0)
    zeta_b = np.arange(zeta_bot, max(zeta_g1 + 14 * s, zeta_h1 + 14), dz)
    b = r_of(-zeta_b)[:, None]
    l = np.sqrt(b * H0 * (b / R_ref) ** 2) * np.linspace(0.0, u_max, n_u)[None, :]
    lnp = lnP(np.sqrt(b ** 2 + l ** 2))
    I_gas = 2 * np.trapezoid(np.exp((1 + a) * lnp), l, axis=1)
    I_haze = 2 * np.trapezoid(np.exp(lnp), l, axis=1)
    tau = np.exp(lnsig0 + lnn) * I_gas + kappa * np.exp(lnn) * I_haze
    assert tau[0] > 30 or logP_cloud is not None
    assert tau[-1] < 1e-5
    b = b[:, 0]
    R_eff2 = b[0] ** 2 + 2 * np.trapezoid(-np.expm1(-tau) * b, b)
    return -lnP(np.sqrt(R_eff2)), zeta_g1, zeta_h1


@pytest.mark.parametrize("logP_cloud", [None, -2.0])
def test_two_component_effective_radius_vs_bruteforce(logP_cloud):
    """Gas with s = 0.6 plus haze (exponent 1) at comparable tau = 1 levels near 10 mbar,
    with and without a 10 mbar deck: the model's zeta_eff matches exact chord
    integration to < 0.05 H, which the old Ein-with-gas-exponent formula did not."""
    T, mu, g_ref, R_ref, P_ref = 1000.0, 2.3, 25.0, R_J, 1e-2      # H/R ~ 0.002, R_ref near the photosphere
    a = 1.0 / 0.6 - 1.0
    H0 = K_B * T / (mu * M_H * g_ref)
    lnn, col = np.log(1e5 / (K_B * T)), 0.5 * np.log(2 * np.pi * R_ref * H0)
    lnsig0 = 4.3 / 0.6 - lnn - col + 0.5 * np.log(1 + a)           # zeta_g1 = 4.3 (14 mbar)
    kappa = np.exp(4.6 - lnn - col)                                 # zeta_h1 = 4.6 (10 mbar)
    bf, zeta_g1, zeta_h1 = bruteforce_zeta_eff(R_ref, g_ref, T, mu, P_ref, lnsig0, a, kappa, logP_cloud)
    zeta_d = None if logP_cloud is None else -logP_cloud * np.log(10.0)
    model = atm._zeta_eff(zeta_g1, 0.6, zeta_h1, zeta_d)[0]
    assert abs(model - bf) < 0.05
    zd = zeta_g1 - 20 if zeta_d is None else zeta_d
    old = zd + 0.6 * atm._ein(np.exp((np.logaddexp(zeta_g1 + 0.6 * atm.EULER_GAMMA, zeta_h1) - zd)
                                     / 0.6 - atm.EULER_GAMMA))
    assert abs(old - bf) > abs(model - bf)


# --- validity warnings (#39) ----------------------------------------------

def test_cold_planet_warns(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    with pytest.warns(UserWarning, match="below the chemistry grid floor"):
        atm.transmission_depth(wl, T_eq_K=226.0, R_p_RJ=0.2, M_p_MJ=0.03, R_s_RSun=0.4,
                               tables_dir=zeta_fixture)


def test_heavy_mu_override_warns(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    with pytest.warns(UserWarning, match="secondary"):
        atm.transmission_depth(wl, **HJ, mu=28.0, tables_dir=zeta_fixture)


def test_ultra_hot_warns(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    with pytest.warns(UserWarning, match="above 2000 K"):
        atm.transmission_depth(wl, T_eq_K=2400.0, R_p_RJ=1.0, M_p_MJ=M_G10, R_s_RSun=1.0,
                               tables_dir=zeta_fixture)


def test_in_range_does_not_warn(zeta_fixture):
    import warnings
    wl = atm.grid_wavelengths(zeta_fixture)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        atm.transmission_depth(wl, **HJ, mu=2.3, tables_dir=zeta_fixture)
        atm.transmission_depth(wl, T_eq_K=2000.0, R_p_RJ=1.0, M_p_MJ=M_G10, R_s_RSun=1.0,
                               tables_dir=zeta_fixture)
