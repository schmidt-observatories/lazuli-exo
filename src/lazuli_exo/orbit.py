"""Orbital geometry helpers shared by the precompute scripts and the runtime."""
import math


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
