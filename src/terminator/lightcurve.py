"""Time-domain transit light curves derived from a SimulatedSpectrum.

The per-sample flux noise follows from the same depth uncertainties the
spectrum carries. Under the noise tables' own assumption of an equal
out-of-transit baseline, a band's depth precision sigma_band relates to the
per-sample flux noise at cadence c by

    sigma_point = sigma_band * sqrt(t14 / (2 c)),

so recombining the light curve recovers exactly the depth error bar shown
on the spectrum (the native integration time cancels).

Transit shape: when the planet is in the frozen target registry
(terminator.targets) and has a limb-darkening table
(precompute/make_ld_tables.py), the model is a quadratically limb-darkened
transit with the real geometry — b from a/R*, cos i and the eccentricity
factor, z(t) along the circular-orbit arc traversed at the conjunction
speed f_e = (1 + e sin w) / sqrt(1 - e^2). Per-band (u1, u2) interpolate
the LD table at the band's bin centers. Otherwise (synthetic planets,
missing LD table) it falls back to a trapezoid for a central transit
(b = 0, no limb darkening) with ingress t14 * k / (1 + k), k = sqrt(depth).
meta["shape"] records which model was used. Band depths vary; the geometry
is shared across bands.
"""
from dataclasses import dataclass, field

import numpy as np

from . import tables


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
    u1: np.ndarray = None    # (n_groups,) quadratic LD; None on fallback
    u2: np.ndarray = None
    meta: dict = field(default_factory=dict)


def transit_flux(z, k, u1, u2, n_r=401):
    """Normalized flux of a quadratically limb-darkened star occulted by
    an opaque disk of radius ratio `k` at projected separations `z`
    (stellar radii).

    Numpy-only numerical disk integration: per separation, quadrature over
    the occulted radial band [|z|-k, |z|+k] of the analytic overlap-arc
    weight times the intensity profile. The Chebyshev-type substitution
    r = c - h*cos(psi) absorbs the square-root behavior at both band edges
    (planet edge and stellar limb), so midpoint quadrature converges to
    ~1e-7 absolute at the default n_r.
    """
    z = np.atleast_1d(np.asarray(z, float))
    zc = np.maximum(np.abs(z), 1e-12)

    r_lo = np.clip(zc - k, 0.0, 1.0)
    r_hi = np.clip(zc + k, 0.0, 1.0)
    c = 0.5 * (r_hi + r_lo)[:, None]
    h = 0.5 * (r_hi - r_lo)[:, None]          # 0 when fully off-disk

    psi = (np.arange(n_r) + 0.5) * (np.pi / n_r)
    r = c - h * np.cos(psi)[None, :]
    mu = np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))
    inten = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2

    # arc of the annulus at radius r covered by the planet disk
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = (zc[:, None] ** 2 + r**2 - k**2) / (2.0 * zc[:, None] * r)
    alpha = 2.0 * np.arccos(np.clip(arg, -1.0, 1.0))

    dr = h * np.sin(psi)[None, :] * (np.pi / n_r)
    occulted = np.sum(inten * alpha * r * dr, axis=1)
    total = np.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    return 1.0 - occulted / total


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

    meta = {"planet": obs.planet, "scenario": obs.scenario,
            "n_transits": obs.n_transits}
    geo = _resolve_geometry(obs)
    if geo is not None:
        target, wl_ld, u1_ld, u2_ld = geo
        e, w = target["ecc"] or 0.0, np.radians(target["w_deg"] or 90.0)
        a_rs, inc = target["a_over_rs"], np.radians(target["inc_deg"])
        f_e = (1.0 + e * np.sin(w)) / np.sqrt(1.0 - e**2)
        b = a_rs * np.cos(inc) * (1.0 - e**2) / (1.0 + e * np.sin(w))
        theta = 2.0 * np.pi * time_s * f_e / (target["period_d"] * 86400.0)
        z = a_rs * np.hypot(np.sin(theta), np.cos(inc) * np.cos(theta))

        u1 = np.array([np.interp(obs.wavelength[g], wl_ld, u1_ld).mean()
                       for g in groups])
        u2 = np.array([np.interp(obs.wavelength[g], wl_ld, u2_ld).mean()
                       for g in groups])
        flux_model = np.stack([
            transit_flux(z, np.sqrt(max(d, 0.0)), u1[i], u2[i])
            for i, d in enumerate(depth)])
        meta.update(shape="quadratic-ld", b=float(b),
                    a_over_rs=float(a_rs))
    else:
        # trapezoid fallback: central transit (b=0), no limb darkening
        u1 = u2 = None
        k = float(np.sqrt(max(np.median(obs.depth), 0.0)))
        t_ingress = t14 * k / (1.0 + k)
        if t_ingress > 0:
            ramp = np.clip((t14 / 2.0 - np.abs(time_s)) / t_ingress,
                           0.0, 1.0)
        else:
            ramp = (np.abs(time_s) <= t14 / 2.0).astype(float)
        flux_model = 1.0 - depth[:, None] * ramp[None, :]
        meta.update(shape="trapezoid")

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
        u1=u1, u2=u2, meta=meta)


def _resolve_geometry(obs):
    """Registry params + LD table for obs.planet, or None to fall back."""
    from .targets import get_target
    try:
        target = get_target(obs.planet)
    except KeyError:
        return None
    needed = ("a_over_rs", "inc_deg", "period_d")
    if any(target.get(f) is None for f in needed):
        return None
    try:
        tdir = obs.meta.get("tables_dir")
        wl_ld, u1_ld, u2_ld, _ = tables.load_ld(
            tables.resolve_tables_dir(tdir), obs.planet)
    except (FileNotFoundError, KeyError):
        return None
    return target, wl_ld, u1_ld, u2_ld
