"""Stellar SED from the PHOENIX mini-grid (precompute/make_phoenix_grid.py)."""
import warnings

import numpy as np

from . import tables
from .atmosphere import _trilinear


def phoenix_sed(teff_K, logg, feh, tables_dir=None):
    """(wavelength [A], F_lambda [erg s-1 cm-2 A-1], info) for a star.

    Trilinear in (log T_eff, log g, [Fe/H]) on the solar-metallicity grid
    (2000-12000 K, log g 3.5-5.5, [Fe/H] = 0 only; issue #34). T_eff below
    the floor, log g outside its range and any non-zero [Fe/H] clamp to the
    nearest node and are reported in info["clamped"][name] = (requested,
    used); T_eff above the ceiling raises ValueError. info["fallback"] is
    always None (kept for report.py). Absolute flux scale is irrelevant
    downstream (slicersim renormalizes to the magnitude).
    """
    if not (np.isfinite(teff_K) and teff_K > 0):
        raise ValueError(f"T_eff must be > 0 K, got {teff_K}")
    g = tables.load_phoenix_grid(tables_dir)
    teff_axis = g["teff_K"]
    if teff_K > teff_axis.max():
        raise ValueError(f"T_eff={teff_K} K above the PHOENIX grid ceiling "
                         f"{teff_axis.max():.0f} K")
    info = {"fallback": None, "clamped": {}, "teff_K": float(teff_K),
            "teff_used": float(teff_K), "logg_used": float(logg),
            "feh_used": float(feh), "grid": "phoenix"}
    for name, x, axis in (("teff", teff_K, teff_axis), ("logg", logg, g["logg"]),
                          ("feh", feh, g["feh"])):
        xc = float(np.clip(x, axis.min(), axis.max()))
        if xc != float(x):
            info["clamped"][name] = (float(x), xc)
            info[f"{name}_used"] = xc
    axes = (np.log10(teff_axis), g["logg"], g["feh"])
    point = (np.log10(info["teff_used"]), info["logg_used"], info["feh_used"])
    log_flam = _trilinear(g["log_flam"], axes, point)
    return g["wavelength_um"] * 1e4, 10.0 ** log_flam, info


# Covering fractions from the literature, for use as `f_spot`/`f_fac` inputs.
# These are data, not defaults: contamination() and simulate()/observe()
# default to an immaculate photosphere (f_spot = f_fac = 0).
#   Rackham, Apai & Giampapa 2018, AJ 853, 122, Table 3 (M dwarfs): "giant
#   spots" f_spot ~ 0.005-0.03 (we quote the upper end), "solar-like" spot
#   cover 0.08-0.12 for early M rising to 0.39 for late M (TRAPPIST-1-like),
#   with faculae f_fac ~ 0.2-0.3 where they give it.
#   Rackham, Apai & Giampapa 2019, AJ 157, 96 (FGK dwarfs): solar-like spot
#   cover f_spot ~ 0.03-0.08 (0.03 giant / 0.08 active), f_fac ~ 0.2-0.3.
TLSE_PRESETS = {
    "M_giant_spots":     dict(f_spot=0.03, f_fac=0.0),
    "M_early_solar":     dict(f_spot=0.08, f_fac=0.0),   # M0-M3, range 0.08-0.12
    "M_mid_solar":       dict(f_spot=0.12, f_fac=0.0),   # M4-M5
    "M_late_solar":      dict(f_spot=0.39, f_fac=0.0),   # M6-M8, TRAPPIST-1-like
    "M_early_solar_fac": dict(f_spot=0.08, f_fac=0.2),
    "M_mid_solar_fac":   dict(f_spot=0.12, f_fac=0.3),
    "M_late_solar_fac":  dict(f_spot=0.39, f_fac=0.3),
    "FGK_giant_spots":   dict(f_spot=0.03, f_fac=0.0),
    "FGK_solar":         dict(f_spot=0.08, f_fac=0.0),   # range 0.03-0.08
    "FGK_solar_fac":     dict(f_spot=0.08, f_fac=0.2),
}


def contamination(wl_um, teff_K, logg, feh, f_spot=0.0, T_spot_K=None,
                  f_fac=0.0, T_fac_K=None, tables_dir=None):
    """Transit light source effect eps(lambda) for unocculted spots/faculae.

    Rackham et al. 2018 (AJ 853, 122) eq. 2:
    eps = 1 / (1 - f_spot (1 - F_spot/F_phot) - f_fac (1 - F_fac/F_phot)),
    with F_* the PHOENIX surface fluxes (phoenix_sed) at (T_phot, T_spot,
    T_fac) and the host log g / [Fe/H]. The observed depth is
    depth_true * eps. Defaults: T_spot = 0.86 T_eff for T_eff < 4000 K
    (Rackham 2018, M dwarfs) else 0.80 T_eff (Rackham 2019, FGK);
    T_fac = T_eff + 100 K. Returns (eps on `wl_um`, info). With
    f_spot = f_fac = 0 returns exactly ones without touching the grid.
    Warns (UserWarning) if a component falls below the PHOENIX floor, where
    phoenix_sed clamps T_eff to the 2000 K node (info["clamped_teff"]).
    See TLSE_PRESETS for literature covering fractions.
    """
    wl_um = np.asarray(wl_um, dtype=float)
    f_spot, f_fac = float(f_spot), float(f_fac)
    if f_spot < 0 or f_fac < 0 or f_spot + f_fac >= 1:
        raise ValueError(f"need 0 <= f_spot + f_fac < 1, got f_spot={f_spot}, "
                         f"f_fac={f_fac}")
    if T_spot_K is None:
        T_spot_K = (0.86 if teff_K < 4000.0 else 0.80) * teff_K
    if T_fac_K is None:
        T_fac_K = teff_K + 100.0
    info = {"f_spot": f_spot, "T_spot_K": float(T_spot_K), "f_fac": f_fac,
            "T_fac_K": float(T_fac_K), "teff_K": float(teff_K),
            "eps_min": 1.0, "eps_max": 1.0, "clamped_teff": []}
    if f_spot == 0.0 and f_fac == 0.0:
        return np.ones_like(wl_um), info

    def sed(name, T):
        wl_A, flam, i = phoenix_sed(T, logg, feh, tables_dir=tables_dir)
        if "teff" in i["clamped"]:
            info["clamped_teff"].append(name)
        return wl_A / 1e4, flam

    wl_g, F_phot = sed("phot", teff_K)
    term = np.zeros_like(F_phot)
    if f_spot > 0:
        term += f_spot * (1 - sed("spot", T_spot_K)[1] / F_phot)
    if f_fac > 0:
        term += f_fac * (1 - sed("fac", T_fac_K)[1] / F_phot)
    eps_g = 1.0 / (1.0 - term)
    if info["clamped_teff"]:
        warnings.warn(
            f"contamination: {info['clamped_teff']} below the PHOENIX grid "
            "floor; T_eff clamped to the coolest node",
            UserWarning, stacklevel=2)
    band = (wl_g >= 0.40) & (wl_g <= 1.70)
    info["eps_min"] = float(eps_g[band].min())
    info["eps_max"] = float(eps_g[band].max())
    return np.interp(wl_um, wl_g, eps_g), info
