"""Assemble a simulated Lazuli IFS transmission-spectroscopy observation."""
from dataclasses import dataclass, field

import numpy as np

from . import binning, tables


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
    flat_depth: float            # (Rp/Rs)^2 baseline from the 'flat' table
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
    """Simulate observing `planet` under `scenario` for `n_transits` transits.

    Returns a SimulatedSpectrum with the noiseless binned model in `.depth`
    and per-bin 1-sigma uncertainties in `.depth_err` (analytic single-transit
    depth precision from the noise tables, scaled by 1/sqrt(n_transits)).
    """
    if n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    tdir = tables.resolve_tables_dir(tables_dir)

    wl_m, d_m, meta_m = tables.load_model(tdir, planet, scenario)
    wl_n, e_n, meta_n = tables.load_noise(tdir, planet)
    _, d_flat, _ = tables.load_model(tdir, planet, "flat")

    edges = binning.constant_R_edges(wl_n.min(), wl_n.max(), R)
    depth = binning.bin_model(wl_m, d_m, edges)
    err = binning.bin_sigma(wl_n, e_n, edges) / np.sqrt(n_transits)

    keep = (wl_m >= edges[0]) & (wl_m <= edges[-1])
    return SimulatedSpectrum(
        planet=meta_m["planet"], scenario=scenario, n_transits=int(n_transits),
        R=int(R), wavelength=binning.bin_centers(edges), depth=depth,
        depth_err=err, model_wavelength=wl_m[keep], model_depth=d_m[keep],
        flat_depth=float(np.median(d_flat)),
        meta={"model": meta_m, "noise": meta_n, "tables_dir": str(tdir)})
