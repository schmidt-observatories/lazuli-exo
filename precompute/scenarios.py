"""Single source of truth for the precompute grids and atmosphere scenarios."""
import numpy as np

WL_MIN_UM = 0.35    # margin around the 0.40-1.70 um IFS bandpass
WL_MAX_UM = 1.85
R_MODEL = 10_000

PARAM_SPECIES = ["H2O", "CO", "CO2", "CH4", "NH3", "Na", "K"]
P_GRID_ARGS = (-7.0, 2.0, 100)   # log10(P/bar): min, max, n
P_REF_BAR = 10.0

# POSEIDON MacMad17 cloud_params ordering:
#   deck: [log_P_cloud (log10 bar)]; haze: [log_a, gamma];
#   deck_haze: [log_a, gamma, log_P_cloud] (haze params first, then deck --
#   matches the append order in POSEIDON.parameters.determine_model_parameters).
#
# NOTE on "hazy": cloud_type='haze' ALONE (no deck) is a POSEIDON trap -- for
# a 1D model without a deck, POSEIDON.parameters.unpack_cloud_params sets
# f_cloud=0.0 (its comment claims this is an unused "dummy value", but it is
# NOT: f_cloud feeds transmission.extend_rad_transfer_grids' cloudy-sector
# calculation, which with f_cloud=0 flags zero sectors as cloudy, so the haze
# opacity (kappa_cloud) is computed but never mixed into the actual
# transmission spectrum -- a silent no-op, confirmed by an empirical A/B
# check: cloud_type='haze' gave a spectrum bit-identical (~1e-12) to 'clear'.
# POSEIDON's own tutorial (transmission_clouds.ipynb) confirms the supported
# recipe is cloud_type='deck_haze' (its "beginner" combined deck+haze
# prescription), which sets enable_deck=1 -> f_cloud=1.0, correctly enabling
# the haze mixing. We pin log_P_cloud to the pressure grid's own deepest
# level (P_GRID_ARGS hi=2.0 -> 100 bar), which is the same as P[0] here, so
# the deck's own opaque-cloud contribution (kappa_cloud_0 for P > P_cloud)
# never triggers (no layer has P > 100.0 when P_cloud == 100.0) -- the deck
# exists purely to unlock f_cloud=1 for the haze term, contributing no
# opacity of its own. Verified via a standalone debug diagnostic: with this
# fix, hazy vs clear differ by >100 ppm at the blue end, not ~1e-12.
SCENARIOS = {
    "clear": dict(cloud_model="cloud-free", cloud_type=None, cloud_params=[],
                  C_to_O=0.55, log_Met=0.0),
    "cloudy": dict(cloud_model="MacMad17", cloud_type="deck",
                   cloud_params=[-3.0], C_to_O=0.55, log_Met=0.0),
    "high_metallicity": dict(cloud_model="cloud-free", cloud_type=None,
                             cloud_params=[], C_to_O=0.55, log_Met=2.0),
    "hazy": dict(cloud_model="MacMad17", cloud_type="deck_haze",
                 cloud_params=[2.0, -6.0, 2.0], C_to_O=0.55, log_Met=0.0),
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

# Parametric zeta grid (precompute/make_zeta_grid.py). Shared with the
# validation script so both read one definition.
ZETA_T_K = [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800,
            2000, 2300, 2600, 3000]
ZETA_LOG_MET = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ZETA_C_TO_O = [0.3, 0.55, 0.8, 1.0]
# Fiducial planets spanning the slant-column axis x = 0.5 ln[(R_p H)/(R_p H)_0]
# (same g and T -> same H, so x = 0.5 ln(R_p/R_p0) = 0 and -1.151). Two nodes
# capture the first-order pressure dependence of the line-core opacity that a
# single fiducial misses for small planets (~0.15 H at band peaks).
FIDUCIALS = [dict(R_p_RJ=1.0, g_ms2=10.0, R_s_RSun=1.0),
             dict(R_p_RJ=0.1, g_ms2=10.0, R_s_RSun=1.0)]
FIDUCIAL = FIDUCIALS[0]
