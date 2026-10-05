"""Showcase planets x scenarios expressed as simulate() parameters."""
from .targets import PLANET_NAMES, get_target  # noqa: F401  (re-exported)

# Mirrors precompute/scenarios.py: cloudy = gray deck at 1 mbar; hazy =
# MacMad17 haze with log_a = 2 (100x H2 Rayleigh at 0.35 um), slope -6;
# flat = deck far above the gas photosphere; high_metallicity = 100x solar.
SCENARIO_PARAMS = {
    "clear": dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=None,
                  log_a_haze=None, haze_slope=-4.0),
    "cloudy": dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=-3.0,
                   log_a_haze=None, haze_slope=-4.0),
    "high_metallicity": dict(log_met=2.0, c_to_o=0.55, log_P_cloud_bar=None,
                             log_a_haze=None, haze_slope=-4.0),
    "hazy": dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=None,
                 log_a_haze=2.0, haze_slope=-6.0),
    "flat": dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=-6.0,
                 log_a_haze=None, haze_slope=-4.0),
}
SCENARIOS = sorted(SCENARIO_PARAMS)


def showcase(planet, scenario="clear"):
    """Full simulate() parameter dict for a frozen showcase target."""
    t = get_target(planet)
    if scenario not in SCENARIO_PARAMS:
        raise KeyError(f"Unknown scenario {scenario!r}. Available: {SCENARIOS}")
    return dict(name=t["name"], scenario=scenario,
                T_eq_K=t["T_eq_K"], R_p_RJ=t["R_p_RJ"], M_p_MJ=t["M_p_MJ"],
                R_s_RSun=t["R_s_RSun"], teff_K=t["T_eff_K"],
                logg=t["log_g_star"], feh=t["Met_star"], mag=t["J_mag"],
                band="J", mag_system="vega", t14_s=t["t14_s"], mu=None,
                **SCENARIO_PARAMS[scenario])
