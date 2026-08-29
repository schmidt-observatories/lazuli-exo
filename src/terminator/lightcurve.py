"""Time-domain transit light curves derived from a SimulatedSpectrum.

The per-sample flux noise follows from the same depth uncertainties the
spectrum carries. Under the noise tables' own assumption of an equal
out-of-transit baseline, a band's depth precision sigma_band relates to the
per-sample flux noise at cadence c by

    sigma_point = sigma_band * sqrt(t14 / (2 c)),

so recombining the light curve recovers exactly the depth error bar shown
on the spectrum (the native integration time cancels).

Transit shape is a trapezoid for a CENTRAL transit (b = 0) with no limb
darkening: ingress duration t14 * k / (1 + k), k = sqrt(depth). The tables
carry no orbital geometry or limb-darkening coefficients, so anything more
detailed would be invented parameters. Band depths vary; the geometry is
shared across bands (set by the white-light k).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LightCurveSet:
    time_hr: np.ndarray      # sample times, hours from mid-transit
    flux_model: np.ndarray   # (n_groups, n_time) noiseless normalized flux
    flux: np.ndarray         # (n_groups, n_time) one noisy realization
    sigma_point: np.ndarray  # (n_groups,) per-sample 1-sigma flux noise
    depth: np.ndarray        # (n_groups,) band-combined transit depth
    wl_lo: np.ndarray        # (n_groups,) band low edge [um, bin centers]
    wl_hi: np.ndarray        # (n_groups,) band high edge [um, bin centers]
    labels: list
    cadence_s: float
    t14_s: float
    meta: dict = field(default_factory=dict)


def _band_combine(depth, depth_err, n_groups):
    """Inverse-variance combine contiguous, equal-count groups of bins."""
    groups = np.array_split(np.arange(depth.size), n_groups)
    w = 1.0 / depth_err**2
    d = np.array([np.sum(w[g] * depth[g]) / np.sum(w[g]) for g in groups])
    s = np.array([1.0 / np.sqrt(np.sum(w[g])) for g in groups])
    return groups, d, s


def light_curves(obs, n_groups=1, seed=None, max_points=2000):
    """Simulated transit light curves in `n_groups` wavelength bands.

    n_groups=1 is the white-light curve. The observation window is
    +/- t14 around mid-transit (the tables' equal out-of-transit
    baseline), sampled at the native integration time, coarsened only
    if that would exceed `max_points` samples.
    """
    if not 1 <= n_groups <= obs.wavelength.size:
        raise ValueError(f"n_groups must be in [1, {obs.wavelength.size}]")
    t14 = float(obs.meta["noise"]["t14_s"])
    t_obs = float(obs.meta["noise"]["t_obs_s"])

    cadence = max(t_obs, 2.0 * t14 / max_points)
    n_time = int(np.floor(2.0 * t14 / cadence))
    time_s = (np.arange(n_time) - (n_time - 1) / 2.0) * cadence

    groups, depth, sigma_band = _band_combine(obs.depth, obs.depth_err,
                                              n_groups)
    sigma_point = sigma_band * np.sqrt(t14 / (2.0 * cadence))

    # shared trapezoid geometry from the white-light depth, b=0, no LD
    k = float(np.sqrt(max(np.median(obs.depth), 0.0)))
    t_ingress = t14 * k / (1.0 + k)
    if t_ingress > 0:
        ramp = np.clip((t14 / 2.0 - np.abs(time_s)) / t_ingress, 0.0, 1.0)
    else:
        ramp = (np.abs(time_s) <= t14 / 2.0).astype(float)

    flux_model = 1.0 - depth[:, None] * ramp[None, :]
    rng = np.random.default_rng(seed)
    flux = flux_model + rng.normal(0.0, 1.0, flux_model.shape) \
        * sigma_point[:, None]

    wl_lo = np.array([obs.wavelength[g[0]] for g in groups])
    wl_hi = np.array([obs.wavelength[g[-1]] for g in groups])
    labels = ([f"{lo:.2f}-{hi:.2f} um" for lo, hi in zip(wl_lo, wl_hi)]
              if n_groups > 1 else ["white light"])

    return LightCurveSet(
        time_hr=time_s / 3600.0, flux_model=flux_model, flux=flux,
        sigma_point=sigma_point, depth=depth, wl_lo=wl_lo, wl_hi=wl_hi,
        labels=labels, cadence_s=float(cadence), t14_s=t14,
        meta={"planet": obs.planet, "scenario": obs.scenario,
              "n_transits": obs.n_transits})
