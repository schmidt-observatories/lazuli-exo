"""Assemble a simulated Lazuli IFS transmission-spectroscopy observation."""
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import binning, presets, tables
from .atmosphere import R_J, R_SUN, grid_wavelengths, transmission_depth
from .star import contamination

# keys transmission_depth() takes from a showcase/simulate params dict
ATMO_KEYS = ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "log_met", "c_to_o",
             "mu", "log_P_cloud_bar", "log_a_haze", "haze_slope")


@dataclass
class SimulatedSpectrum:
    planet: str
    scenario: str
    n_transits: int
    R: int
    wavelength: np.ndarray       # bin centers [um]
    depth: np.ndarray            # binned model transit depth
    depth_err: np.ndarray        # 1-sigma per bin, n_transits included
    model_wavelength: np.ndarray  # native-resolution model [um]
    model_depth: np.ndarray
    flat_depth: float            # (Rp/Rs)^2 from the archive radii
    meta: dict = field(default_factory=dict)

    def simulate_points(self, seed=None):
        """One noisy realization of the binned spectrum."""
        rng = np.random.default_rng(seed)
        return self.depth + rng.normal(0.0, self.depth_err)

    def detect(self):
        from .detect import detect_features
        return detect_features(self.wavelength, self.depth, self.depth_err)

    def light_curves(self, n_groups=1, seed=None, **kwargs):
        from .lightcurve import light_curves
        return light_curves(self, n_groups=n_groups, seed=seed, **kwargs)

    def plot(self, ax=None, **kwargs):
        """Model + one noisy realization; see plotting.plot_spectrum."""
        from .plotting import plot_spectrum
        return plot_spectrum(self, ax=ax, **kwargs)

    def plot_noise(self, ax=None, **kwargs):
        """Per-bin 1-sigma error vs wavelength; see plotting.plot_noise."""
        from .plotting import plot_noise
        return plot_noise(self, ax=ax, **kwargs)

    def plot_orbit(self, t_d=0.0, **kwargs):
        """The planet on its registry orbit; see plotting.plot_orbit."""
        from .plotting import plot_orbit
        return plot_orbit(self, t_d=t_d, **kwargs)

    def animate_orbit(self, **kwargs):
        """FuncAnimation of one orbit; see plotting.animate_orbit."""
        from .plotting import animate_orbit
        return animate_orbit(self, **kwargs)


def positive_int(name, value):
    """Validate a count or resolving power: a positive integer (an integral
    float such as 3.0 is accepted); bool is rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or not np.isfinite(value) or value != int(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def observe(planet, scenario="clear", n_transits=1, R=100, tables_dir=None,
            f_spot=0.0, T_spot_K=None, f_fac=0.0, T_fac_K=None):
    """Simulate observing showcase `planet` under `scenario` for `n_transits`.

    The spectrum is the parametric model evaluated at the frozen registry
    values (`presets.showcase`), so observe() and simulate() share one model
    and one radius convention: the archive R_p is the band-median transit
    radius and H uses the true T_eq. The per-bin 1-sigma uncertainties come
    from the precomputed slicersim noise table, scaled by 1/sqrt(n_transits).
    `f_spot, T_spot_K, f_fac, T_fac_K` add stellar contamination
    (star.contamination, registry T_eff/log g/[Fe/H]); default off.
    `flat_depth` stays the geometric (R_p/R_s)^2.
    """
    n_transits, R = positive_int("n_transits", n_transits), positive_int("R", R)
    tdir = tables.resolve_tables_dir(tables_dir)
    p = presets.showcase(planet, scenario)
    wl_m = grid_wavelengths(tdir)
    d_m, info_a = transmission_depth(
        wl_m, **{k: p[k] for k in ATMO_KEYS}, tables_dir=tdir)
    eps, info_a["tlse"] = contamination(
        wl_m, p["teff_K"], p["logg"], p["feh"], f_spot=f_spot,
        T_spot_K=T_spot_K, f_fac=f_fac, T_fac_K=T_fac_K, tables_dir=tdir)
    d_m = d_m * eps
    wl_n, e_n, meta_n = tables.load_noise(tdir, planet)

    if meta_n.get("fully_desaturated") is False:
        warnings.warn(
            f"{planet}: noise table was generated with residual detector "
            f"saturation ({meta_n.get('nsaturated_detpx')} px); its "
            "uncertainties are unvalidated — do not use for quantitative "
            "forecasts", UserWarning, stacklevel=2)

    # within-pixel photon weighting from the R=10,000 star table (F_lambda ->
    # photons: x lambda; absolute scale cancels); uniform if the table is absent
    try:
        wl_s, f_s, _ = tables.load_star(tdir, p["name"])
        f_phot, weighting = (lambda lam: np.interp(lam, wl_s, f_s * wl_s),
                             "star table R=10000")
    except FileNotFoundError:
        f_phot, weighting = None, "uniform (no star table)"
    edges, depth, err, info_i = binning.bin_observation(
        wl_m, d_m, wl_n, e_n, R, f_phot=f_phot, noise_meta=meta_n)
    err = err / np.sqrt(n_transits)
    info_i["photon_weighting"] = weighting

    keep = (wl_m >= edges[0]) & (wl_m <= edges[-1])
    return SimulatedSpectrum(
        planet=p["name"], scenario=scenario, n_transits=int(n_transits),
        R=int(R), wavelength=binning.bin_centers(edges), depth=depth,
        depth_err=err, model_wavelength=wl_m[keep], model_depth=d_m[keep],
        flat_depth=float((p["R_p_RJ"] * R_J / (p["R_s_RSun"] * R_SUN)) ** 2),
        meta={"atmosphere": info_a, "noise": meta_n, "params": p,
              "instrument_response": info_i,
              "k_source": "archive R_p / R_s (registry R_p_RJ, R_s_RSun)",
              "tables_dir": str(tdir)})
