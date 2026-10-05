"""Assemble a simulated Lazuli IFS transmission-spectroscopy observation."""
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import binning, presets, tables
from .atmosphere import R_J, R_SUN, grid_wavelengths, transmission_depth

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
        from .plotting import plot_spectrum
        return plot_spectrum(self, ax=ax, **kwargs)


def observe(planet, scenario="clear", n_transits=1, R=100, tables_dir=None):
    """Simulate observing showcase `planet` under `scenario` for `n_transits`.

    The spectrum is the parametric model evaluated at the frozen registry
    values (`presets.showcase`), so observe() and simulate() share one model
    and one radius convention: the archive R_p is the band-median transit
    radius and H uses the true T_eq. The per-bin 1-sigma uncertainties come
    from the precomputed slicersim noise table, scaled by 1/sqrt(n_transits).
    """
    if n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    tdir = tables.resolve_tables_dir(tables_dir)
    p = presets.showcase(planet, scenario)
    wl_m = grid_wavelengths(tdir)
    d_m, info_a = transmission_depth(
        wl_m, **{k: p[k] for k in ATMO_KEYS}, tables_dir=tdir)
    wl_n, e_n, meta_n = tables.load_noise(tdir, planet)

    if meta_n.get("fully_desaturated") is False:
        warnings.warn(
            f"{planet}: noise table was generated with residual detector "
            f"saturation ({meta_n.get('nsaturated_detpx')} px); its "
            "uncertainties are unvalidated — do not use for quantitative "
            "forecasts", UserWarning, stacklevel=2)

    edges = binning.constant_R_edges(wl_n.min(), wl_n.max(), R)
    depth = binning.bin_model(wl_m, d_m, edges)
    err = binning.bin_sigma(wl_n, e_n, edges) / np.sqrt(n_transits)

    keep = (wl_m >= edges[0]) & (wl_m <= edges[-1])
    return SimulatedSpectrum(
        planet=p["name"], scenario=scenario, n_transits=int(n_transits),
        R=int(R), wavelength=binning.bin_centers(edges), depth=depth,
        depth_err=err, model_wavelength=wl_m[keep], model_depth=d_m[keep],
        flat_depth=float((p["R_p_RJ"] * R_J / (p["R_s_RSun"] * R_SUN)) ** 2),
        meta={"atmosphere": info_a, "noise": meta_n, "params": p,
              "k_source": "archive R_p / R_s (registry R_p_RJ, R_s_RSun)",
              "tables_dir": str(tdir)})
