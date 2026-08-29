"""Single source of truth for the precompute grids and atmosphere scenarios."""
import numpy as np

WL_MIN_UM = 0.35    # margin around the 0.40-1.70 um IFS bandpass
WL_MAX_UM = 1.85
R_MODEL = 10_000

PARAM_SPECIES = ["H2O", "CO", "CO2", "CH4", "NH3", "Na", "K"]
P_GRID_ARGS = (-7.0, 2.0, 100)   # log10(P/bar): min, max, n
P_REF_BAR = 10.0

# POSEIDON MacMad17 cloud_params ordering:
#   deck: [log_P_cloud (log10 bar)]; haze: [log_a, gamma]
SCENARIOS = {
    "clear": dict(cloud_model="cloud-free", cloud_type=None, cloud_params=[],
                  C_to_O=0.55, log_Met=0.0),
    "cloudy": dict(cloud_model="MacMad17", cloud_type="deck",
                   cloud_params=[-3.0], C_to_O=0.55, log_Met=0.0),
    "high_metallicity": dict(cloud_model="cloud-free", cloud_type=None,
                             cloud_params=[], C_to_O=0.55, log_Met=2.0),
    "hazy": dict(cloud_model="MacMad17", cloud_type="haze",
                 cloud_params=[2.0, -6.0], C_to_O=0.55, log_Met=0.0),
}


def make_wl_grid():
    """Constant-R grid, byte-for-byte matching POSEIDON.core.wl_grid_constant_R.

    POSEIDON builds this as N_wl = round((ln(wl_max)-ln(wl_min)) * R) points
    linearly spaced in log-wavelength (not floor(...)+1 points as an earlier
    version of this function did) — verified against
    POSEIDON.core.wl_grid_constant_R(0.35, 1.85, 10_000) in
    precompute/smoke_poseidon.py (Task 8).
    """
    delta_log_wl = 1.0 / R_MODEL
    n_wl = np.log(WL_MAX_UM / WL_MIN_UM) / delta_log_wl
    n_wl = int(np.around(n_wl))
    log_wl = np.linspace(np.log(WL_MIN_UM), np.log(WL_MAX_UM), n_wl)
    wl = np.exp(log_wl)
    wl[0] = WL_MIN_UM
    wl[-1] = WL_MAX_UM
    return wl
