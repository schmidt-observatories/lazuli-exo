"""Orbital geometry helpers shared by the precompute scripts and the runtime."""
import math

import numpy as np


def transit_duration_s(period_d, a_over_rs, rp_rs, inc_deg=90.0, ecc=0.0,
                       w_deg=90.0):
    """Total transit duration T14 in seconds (Winn 2010, eqs. 14 and 16).

    Impact parameter and duration both carry the eccentric conjunction-
    distance / velocity factors. Returns 0.0 for a non-transiting geometry.
    """
    w = math.radians(w_deg)
    i = math.radians(inc_deg)
    fac = math.sqrt(1.0 - ecc ** 2) / (1.0 + ecc * math.sin(w))
    b = a_over_rs * math.cos(i) * (1.0 - ecc ** 2) / (1.0 + ecc * math.sin(w))
    arg = math.sqrt(max((1.0 + rp_rs) ** 2 - b ** 2, 0.0)) / (
        a_over_rs * math.sin(i))
    return period_d * 86400.0 / math.pi * math.asin(min(arg, 1.0)) * fac


def projected_separation(t_s, period_d, a_over_rs, inc_deg=90.0, ecc=0.0,
                         w_deg=90.0):
    """Sky-projected star-planet separation z(t) in stellar radii for a
    Keplerian orbit (Winn 2010, eqs. 2-5), `t_s` seconds from mid-transit.

    Mid-transit is inferior conjunction (true anomaly f = pi/2 - w, planet
    in front of the star, batman/jaxoplanet convention), so z(0) is the
    eccentric impact parameter. Kepler's equation is solved by vectorised
    Newton iteration; for ecc = 0 this reduces to the circular arc.
    """
    w, i = math.radians(w_deg), math.radians(inc_deg)
    f_c = math.pi / 2.0 - w
    E_c = 2.0 * math.atan2(math.sqrt(1.0 - ecc) * math.sin(f_c / 2.0),
                           math.sqrt(1.0 + ecc) * math.cos(f_c / 2.0))
    M = (E_c - ecc * math.sin(E_c)
         + 2.0 * np.pi * np.asarray(t_s, float) / (period_d * 86400.0))
    E = M + ecc * np.sin(M)
    for _ in range(12):
        E = E - (E - ecc * np.sin(E) - M) / (1.0 - ecc * np.cos(E))
    f = 2.0 * np.arctan2(np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
                         np.sqrt(1.0 - ecc) * np.cos(E / 2.0))
    r = a_over_rs * (1.0 - ecc**2) / (1.0 + ecc * np.cos(f))
    return r * np.hypot(np.cos(w + f), np.sin(w + f) * math.cos(i))
