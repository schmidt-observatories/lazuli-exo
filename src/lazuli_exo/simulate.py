"""Assemble a parametric simulation into the existing SimulatedSpectrum."""
from . import binning
from .atmosphere import R_J, R_SUN, grid_wavelengths, transmission_depth
from .observe import SimulatedSpectrum
from .slicer import depth_error
from .star import contamination, phoenix_sed

REQUIRED = ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "teff_K", "logg", "feh",
            "mag", "t14_s")
ATMO_DEFAULTS = dict(log_met=0.0, c_to_o=0.55, mu=None, log_P_cloud_bar=None,
                     log_a_haze=None, haze_slope=-4.0)
TLSE_DEFAULTS = dict(f_spot=0.0, T_spot_K=None, f_fac=0.0, T_fac_K=None)


def simulate(params, R=100, n_transits=1, tables_dir=None,
             instrument="lazuli_cbe", **setup):
    """Simulate any transiting planet from a parameter dict.

    Required keys: T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun, teff_K, logg, feh, mag,
    t14_s. Optional: name, scenario, band ("J"), mag_system ("vega"),
    log_met, c_to_o, mu, log_P_cloud_bar, log_a_haze, haze_slope, and the
    stellar-contamination keys f_spot, T_spot_K, f_fac, T_fac_K
    (star.contamination; default 0 = immaculate photosphere). Needs
    slicersim + sncosmo at call time (not at import).

    ``**setup`` is the optional instrument setup forwarded to
    slicer.depth_error: nmd, tframe, spectrograph_mode, guiding_sigma,
    zodi_ab_mag (see slicer.normalize_setup).
    """
    missing = [k for k in REQUIRED if params.get(k) is None]
    if missing:
        raise ValueError(f"simulate() params missing {missing}")
    if n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    p = {**ATMO_DEFAULTS, **TLSE_DEFAULTS, **params}

    wl_model = grid_wavelengths(tables_dir)
    depth_model, info_a = transmission_depth(
        wl_model, T_eq_K=p["T_eq_K"], R_p_RJ=p["R_p_RJ"], M_p_MJ=p["M_p_MJ"],
        R_s_RSun=p["R_s_RSun"], log_met=p["log_met"], c_to_o=p["c_to_o"],
        mu=p["mu"], log_P_cloud_bar=p["log_P_cloud_bar"],
        log_a_haze=p["log_a_haze"], haze_slope=p["haze_slope"],
        tables_dir=tables_dir)
    eps, info_a["tlse"] = contamination(
        wl_model, p["teff_K"], p["logg"], p["feh"], f_spot=p["f_spot"],
        T_spot_K=p["T_spot_K"], f_fac=p["f_fac"], T_fac_K=p["T_fac_K"],
        tables_dir=tables_dir)
    depth_model = depth_model * eps
    wl_A, flam, info_s = phoenix_sed(p["teff_K"], p["logg"], p["feh"],
                                     tables_dir=tables_dir)
    wl_n, err, info_n = depth_error(
        wl_A, flam, mag=p["mag"], band=p.get("band", "J"),
        mag_system=p.get("mag_system", "vega"), t14_s=p["t14_s"],
        n_transits=n_transits, instrument=instrument, **setup)

    edges = binning.constant_R_edges(wl_n.min(), wl_n.max(), R)
    depth = binning.bin_model(wl_model, depth_model, edges)
    sigma = binning.bin_sigma(wl_n, err, edges)
    keep = (wl_model >= edges[0]) & (wl_model <= edges[-1])
    return SimulatedSpectrum(
        planet=str(p.get("name", "custom")),
        scenario=str(p.get("scenario", "custom")),
        n_transits=int(n_transits), R=int(R),
        wavelength=binning.bin_centers(edges), depth=depth, depth_err=sigma,
        model_wavelength=wl_model[keep], model_depth=depth_model[keep],
        flat_depth=float((p["R_p_RJ"] * R_J / (p["R_s_RSun"] * R_SUN)) ** 2),
        meta={"atmosphere": info_a, "star": info_s, "noise": info_n,
              "params": dict(p),
              "tables_dir": str(tables_dir) if tables_dir else None})
