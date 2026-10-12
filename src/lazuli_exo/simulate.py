"""Assemble a parametric simulation into the existing SimulatedSpectrum."""
import numpy as np

from . import binning
from .atmosphere import R_J, R_SUN, grid_wavelengths, transmission_depth
from .observe import SimulatedSpectrum
from .slicer import depth_error
from .star import contamination, phoenix_sed

STAR_REQUIRED = ("teff_K", "logg", "feh", "mag", "t14_s")
ATMO_REQUIRED = ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun")
REQUIRED = ATMO_REQUIRED + STAR_REQUIRED
ATMO_DEFAULTS = dict(log_met=0.0, c_to_o=0.55, mu=None, log_P_cloud_bar=None,
                     log_a_haze=None, haze_slope=-4.0)
TLSE_DEFAULTS = dict(f_spot=0.0, T_spot_K=None, f_fac=0.0, T_fac_K=None)


def simulate(params, R=100, n_transits=1, tables_dir=None,
             instrument="lazuli_cbe", spectrum=None, **setup):
    """Simulate any transiting planet from a parameter dict.

    Required keys: T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun, teff_K, logg, feh, mag,
    t14_s. Optional: name, scenario, band ("J"), mag_system ("vega"),
    log_met, c_to_o, mu, log_P_cloud_bar, log_a_haze, haze_slope, and the
    stellar-contamination keys f_spot, T_spot_K, f_fac, T_fac_K
    (star.contamination; default 0 = immaculate photosphere). Needs
    slicersim + sncosmo at call time (not at import).

    ``spectrum=(wl_um, depth)`` replaces the parametric atmosphere with your
    own forward model (POSEIDON, petitRADTRANS, PICASO, ...): a 1D transit
    depth (R_p/R_s)^2 on an increasing wavelength grid that covers the
    instrument band (0.40-1.70 um, with a few pixels of margin: the LSF
    reaches past the band and the edge value is held there). The star,
    noise, contamination and
    instrument response are then applied exactly as for the parametric
    model, and only the star keys (teff_K, logg, feh, mag, t14_s) are
    required; R_p_RJ and R_s_RSun, when given, set ``flat_depth``, otherwise
    it is the median of the spectrum. meta["atmosphere"]["source"] == "user".

    ``**setup`` is the optional instrument setup forwarded to
    slicer.depth_error: nmd, tframe, spectrograph_mode, guiding_sigma,
    zodi_ab_mag (see slicer.normalize_setup).
    """
    required = STAR_REQUIRED if spectrum is not None else REQUIRED
    missing = [k for k in required if params.get(k) is None]
    if missing:
        raise ValueError(f"simulate() params missing {missing}")
    if n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    p = {**ATMO_DEFAULTS, **TLSE_DEFAULTS, **params}

    if spectrum is not None:
        wl_model, depth_model = _user_spectrum(spectrum)
        info_a = {"source": "user", "n_points": int(wl_model.size),
                  "wl_range_um": [float(wl_model[0]), float(wl_model[-1])]}
    else:
        wl_model = grid_wavelengths(tables_dir)
        depth_model, info_a = transmission_depth(
            wl_model, T_eq_K=p["T_eq_K"], R_p_RJ=p["R_p_RJ"], M_p_MJ=p["M_p_MJ"],
            R_s_RSun=p["R_s_RSun"], log_met=p["log_met"], c_to_o=p["c_to_o"],
            mu=p["mu"], log_P_cloud_bar=p["log_P_cloud_bar"],
            log_a_haze=p["log_a_haze"], haze_slope=p["haze_slope"],
            tables_dir=tables_dir)
        info_a["source"] = "parametric"
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

    # within-pixel photon weighting from the R=1000 PHOENIX SED (coarser than
    # the R=10,000 star tables observe() uses); F_lambda -> photons: x lambda
    wl_s = wl_A * 1e-4
    edges, depth, sigma, info_i = binning.bin_observation(
        wl_model, depth_model, wl_n, err, R, noise_meta=info_n,
        f_phot=lambda lam: np.interp(lam, wl_s, flam * wl_s))
    info_i["photon_weighting"] = "phoenix_sed R=1000"
    keep = (wl_model >= edges[0]) & (wl_model <= edges[-1])
    if spectrum is not None and (wl_model[0] > edges[1] or wl_model[-1] < edges[-2]):
        # must reach into the first and last bin; a sub-bin sliver at either
        # end is held at the edge value by the interpolation
        raise ValueError(f"spectrum covers {wl_model[0]:.3f}-{wl_model[-1]:.3f} um; "
                         f"the binned range is {edges[0]:.3f}-{edges[-1]:.3f} um")
    if p.get("R_p_RJ") is not None and p.get("R_s_RSun") is not None:
        flat = float((p["R_p_RJ"] * R_J / (p["R_s_RSun"] * R_SUN)) ** 2)
    else:
        flat = float(np.median(depth_model[keep]))
    return SimulatedSpectrum(
        planet=str(p.get("name", "custom")),
        scenario=str(p.get("scenario", "user" if spectrum is not None else "custom")),
        n_transits=int(n_transits), R=int(R),
        wavelength=binning.bin_centers(edges), depth=depth, depth_err=sigma,
        model_wavelength=wl_model[keep], model_depth=depth_model[keep],
        flat_depth=flat,
        meta={"atmosphere": info_a, "star": info_s, "noise": info_n,
              "instrument_response": info_i, "params": dict(p),
              "tables_dir": str(tables_dir) if tables_dir else None})


def _user_spectrum(spectrum):
    """Validate a (wl_um, depth) pair for ``simulate(spectrum=...)``."""
    try:
        wl, depth = spectrum
        wl, depth = np.asarray(wl, float), np.asarray(depth, float)
    except (TypeError, ValueError):
        raise ValueError("spectrum must be a (wavelength_um, depth) pair of 1D arrays") from None
    if wl.ndim != 1 or depth.shape != wl.shape or wl.size < 2:
        raise ValueError("spectrum arrays must be 1D and the same length")
    if not (np.isfinite(wl).all() and np.isfinite(depth).all()):
        raise ValueError("spectrum must be finite")
    if np.any(np.diff(wl) <= 0):
        raise ValueError("spectrum wavelength must be strictly increasing (um)")
    if np.any(depth < 0) or np.any(depth > 1):
        raise ValueError("spectrum depth must be (R_p/R_s)^2 in [0, 1], not ppm or mmag")
    return wl, depth
