"""Stellar SED from the PHOENIX mini-grid (precompute/make_phoenix_grid.py)."""
import numpy as np

from . import tables
from .atmosphere import _trilinear

H_PLANCK, C_LIGHT, K_B = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def _blackbody_flam(wl_um, T):
    """pi * B_lambda in erg s^-1 cm^-2 A^-1 (surface flux)."""
    lam = wl_um * 1e-6
    B = (2 * H_PLANCK * C_LIGHT ** 2 / lam ** 5
         / np.expm1(H_PLANCK * C_LIGHT / (lam * K_B * T)))
    return np.pi * B * 1e-7            # W m^-2 m^-1 -> erg s^-1 cm^-2 A^-1


def phoenix_sed(teff_K, logg, feh, tables_dir=None):
    """(wavelength [A], F_lambda [erg s-1 cm-2 A-1], info) for a star.

    Trilinear in (log T_eff, log g, [Fe/H]). log g and [Fe/H] clamp to the
    grid edges (reported in info["clamped"]); T_eff below the grid floor
    returns a blackbody (info["fallback"] == "blackbody"); above the ceiling
    raises ValueError. Absolute flux scale is irrelevant downstream
    (slicersim renormalizes to the magnitude).
    """
    g = tables.load_phoenix_grid(tables_dir)
    wl_um = g["wavelength_um"]
    teff_axis = g["teff_K"]
    info = {"fallback": None, "clamped": {}, "teff_K": float(teff_K),
            "logg_used": float(logg), "feh_used": float(feh), "grid": "phoenix"}
    if teff_K > teff_axis.max():
        raise ValueError(f"T_eff={teff_K} K above the PHOENIX grid ceiling "
                         f"{teff_axis.max():.0f} K")
    if teff_K < teff_axis.min():
        info["fallback"] = "blackbody"
        info["grid"] = "blackbody"
        return wl_um * 1e4, _blackbody_flam(wl_um, float(teff_K)), info
    for name, x, axis in (("logg", logg, g["logg"]), ("feh", feh, g["feh"])):
        xc = float(np.clip(x, axis.min(), axis.max()))
        if xc != float(x):
            info["clamped"][name] = (float(x), xc)
            info[f"{name}_used"] = xc
    axes = (np.log10(teff_axis), g["logg"], g["feh"])
    point = (np.log10(teff_K), info["logg_used"], info["feh_used"])
    log_flam = _trilinear(g["log_flam"].astype(float), axes, point)
    return wl_um * 1e4, 10.0 ** log_flam, info
