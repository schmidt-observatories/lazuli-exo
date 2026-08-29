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
    """Constant-R grid matching POSEIDON's wl_grid_constant_R convention."""
    n = int(np.floor(np.log(WL_MAX_UM / WL_MIN_UM) * R_MODEL))
    return WL_MIN_UM * np.exp(np.arange(n + 1) / R_MODEL)
