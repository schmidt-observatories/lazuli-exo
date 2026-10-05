"""Parametric isothermal transmission spectrum from the shared zeta grid.

zeta(lambda) = -ln P_photo(lambda)/bar is tabulated on a (T, [M/H], C/O) grid
for two fiducial planets (1 R_J at g = 10 m/s^2; 0.3 R_J at g = 30 m/s^2),
i.e. at two nodes x = 0 and -1.15 of the slant-column axis x = 0.5 ln[(R_p H)/(R_p H)_0]
(precompute/make_zeta_grid.py). For any other planet, with
x_p = 0.5 ln[(R_p H)/(R_p,0 H_0)] - ln(T_eq/T_chem):

    zeta_p = lerp_x( zeta_fid_k + (x_p - x_k) )     (clamped to the node range)
    zeta   = max(logaddexp(zeta_p, zeta_haze), -ln P_cloud)
    R(lam) = R_p / (1 - (H / R_p) (zeta - median_band(zeta))),  depth = (R / R_s)^2

so R_p is the band-median transit radius and H = k T_eq / (mu m_H g(R_p)).
The radius mapping is the exact isothermal hydrostatic solution with
g ~ 1/r^2 (1/r is linear in ln P), which POSEIDON also integrates; its first-
order form R_p + H dzeta under-predicts puffy atmospheres (H/R_p ~ 0.03 gives
~30 % too little amplitude). Fidelity: isothermal, equilibrium chemistry,
gray deck, power-law haze; no pressure-broadening correction.
"""
import numpy as np

from . import tables

K_B = 1.380649e-23
M_H = 1.6735575e-27
G = 6.674e-11
R_J = 7.1492e7
M_J = 1.898e27
R_SUN = 6.957e8
SIGMA_H2_RAYLEIGH_350NM = 5.31e-31   # m^2, POSEIDON MacMad17 haze anchor
HAZE_LAMBDA0_UM = 0.35
BAND_UM = (0.40, 1.70)


def grid_wavelengths(tables_dir=None):
    return tables.load_zeta_grid(tables_dir)["wavelength_um"]


def _interp_weights(axis, x):
    """Lower index and fractional weight for linear interpolation on `axis`."""
    axis = np.asarray(axis, float)
    if len(axis) == 1:
        return 0, 0.0
    i = int(np.clip(np.searchsorted(axis, x, side="right") - 1, 0, len(axis) - 2))
    return i, float((x - axis[i]) / (axis[i + 1] - axis[i]))


def _trilinear(values, axes, point):
    """Trilinear interpolation of values[a, b, c, ...] at point = (x, y, z)."""
    (i0, t0), (i1, t1), (i2, t2) = [_interp_weights(a, p)
                                    for a, p in zip(axes, point)]
    out = 0.0
    for di, w0 in ((0, 1 - t0), (1, t0)):
        for dj, w1 in ((0, 1 - t1), (1, t1)):
            for dk, w2 in ((0, 1 - t2), (1, t2)):
                w = w0 * w1 * w2
                if w:
                    corner = values[min(i0 + di, values.shape[0] - 1),
                                    min(i1 + dj, values.shape[1] - 1),
                                    min(i2 + dk, values.shape[2] - 1)]
                    out = out + w * np.asarray(corner, dtype=float)
    return out


def _check_range(name, x, axis):
    lo, hi = float(np.min(axis)), float(np.max(axis))
    if not (lo <= x <= hi):
        raise ValueError(f"{name}={x} outside the grid range [{lo}, {hi}]")


def transmission_depth(wl_um, *, T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun,
                       log_met=0.0, c_to_o=0.55, mu=None,
                       log_P_cloud_bar=None, log_a_haze=None, haze_slope=-4.0,
                       tables_dir=None):
    """Transit depth on `wl_um` plus a dict of derived quantities.

    R_p_RJ is the band-median (0.40-1.70 um) transit radius. T_eq below the
    grid floor clamps the chemistry lookup only (reported as T_clamped); the
    scale height always uses the true T_eq. mu=None takes the equilibrium-
    chemistry mean molecular weight from the grid.
    """
    g = tables.load_zeta_grid(tables_dir)
    wl_grid = g["wavelength_um"]
    wl_um = np.atleast_1d(np.asarray(wl_um, float))
    for name, x in (("T_eq_K", T_eq_K), ("R_p_RJ", R_p_RJ),
                    ("M_p_MJ", M_p_MJ), ("R_s_RSun", R_s_RSun)):
        if not (np.isfinite(x) and x > 0):
            raise ValueError(f"{name} must be > 0, got {x}")
    if mu is not None and not (np.isfinite(mu) and mu > 0):
        raise ValueError(f"mu must be > 0, got {mu}")
    # Stored zeta are bin averages at bin centres and the last partial bin of
    # the nominal span is dropped by the binning, so accept requests up to
    # 1.5 bin widths beyond the outer centres (half a bin past the outer bin
    # edges, which covers the nominal 0.35-1.85 um span); np.interp holds the
    # edge value there.
    lo = wl_grid[0] - 1.5 * (wl_grid[1] - wl_grid[0])
    hi = wl_grid[-1] + 1.5 * (wl_grid[-1] - wl_grid[-2])
    if wl_um.min() < lo or wl_um.max() > hi:
        raise ValueError(
            f"wavelength range [{wl_um.min():.3f}, {wl_um.max():.3f}] um "
            f"outside the grid [{wl_grid.min():.3f}, {wl_grid.max():.3f}] um")
    _check_range("log_met", log_met, g["log_met"])
    _check_range("c_to_o", c_to_o, g["c_to_o"])

    T_chem = float(np.clip(T_eq_K, g["T_K"].min(), g["T_K"].max()))
    T_clamped = T_chem != float(T_eq_K)
    axes = (g["T_K"], g["log_met"], g["c_to_o"])
    point = (T_chem, log_met, c_to_o)
    mu_chem = float(_trilinear(g["mu"], axes, point))
    mu_used = mu_chem if mu is None else float(mu)

    fids = g["meta"]["fiducials"]
    fid = fids[0]
    R_p = R_p_RJ * R_J
    R_s = R_s_RSun * R_SUN
    g_p = G * M_p_MJ * M_J / R_p ** 2
    H = K_B * T_eq_K / (mu_used * M_H * g_p)
    H_fid = K_B * T_chem / (mu_chem * M_H * fid["g_ms2"])
    shift = (0.5 * np.log((R_p * H) / (fid["R_p_m"] * H_fid))
             - np.log(T_eq_K / T_chem))
    # Each fiducial node predicts zeta_k + (x_p - x_k); blend linearly in x
    # between the two nearest nodes, clamped to the node range outside it.
    x_nodes = np.array([f["x_column"] for f in fids], float)
    order = np.argsort(x_nodes)
    x_p = float(np.clip(shift, x_nodes.min(), x_nodes.max()))
    i0, t = _interp_weights(x_nodes[order], x_p)
    k0, k1 = order[i0], order[min(i0 + 1, len(order) - 1)]
    pred0 = _trilinear(g["zeta"][..., k0, :], axes, point) + (shift - x_nodes[k0])
    pred1 = _trilinear(g["zeta"][..., k1, :], axes, point) + (shift - x_nodes[k1])
    zeta = (1 - t) * pred0 + t * pred1

    if log_a_haze is not None:
        kappa = 10.0 ** log_a_haze * SIGMA_H2_RAYLEIGH_350NM * (
            wl_grid / HAZE_LAMBDA0_UM) ** haze_slope          # m^2 per molecule
        P_haze_bar = K_B * T_eq_K / (kappa * np.sqrt(2 * np.pi * R_p * H)) / 1e5
        zeta = np.logaddexp(zeta, -np.log(P_haze_bar))
    if log_P_cloud_bar is not None:
        zeta = np.maximum(zeta, -log_P_cloud_bar * np.log(10.0))

    band = (wl_grid >= BAND_UM[0]) & (wl_grid <= BAND_UM[1])
    zeta_ref = float(np.median(zeta[band]))
    # 1/r linear in ln P (isothermal, g ~ r^-2); the denominator only
    # approaches 0 for an unbound atmosphere (H/R_p * dzeta -> 1).
    denom = 1.0 - (H / R_p) * (zeta - zeta_ref)
    if denom.min() <= 0.1:
        raise ValueError(
            f"atmosphere is unbound for this (T_eq, mu, g): H/R_p = {H / R_p:.3f}")
    radius = R_p / denom
    depth_grid = (radius / R_s) ** 2
    depth = np.interp(wl_um, wl_grid, depth_grid)

    info = {"g_ms2": float(g_p), "H_m": float(H), "H_over_Rp": float(H / R_p),
            "mu_used": mu_used,
            "mu_source": "chemistry" if mu is None else "user",
            "T_chem_K": T_chem, "T_clamped": bool(T_clamped),
            "zeta_ref": zeta_ref, "zeta_shift": float(shift),
            "feature_amplitude_ppm": float(np.ptp(depth_grid[band]) * 1e6),
            "band_um": list(BAND_UM)}
    return depth, info
