"""Stellar SED from the PHOENIX mini-grid (precompute/make_phoenix_grid.py)."""
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
