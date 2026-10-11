"""Time-domain transit light curves derived from a SimulatedSpectrum.

The per-sample flux noise follows from the same depth uncertainties the
spectrum carries. Under the noise tables' own assumption of an equal
out-of-transit baseline, a band's depth precision sigma_band relates to the
per-sample flux noise at cadence c by

    sigma_point = sigma_band * sqrt(t14 / (2 c)),

so a depth + baseline fit to the light curve recovers the depth error bar
shown on the spectrum to ~10 % depending on the template (0.83-1.07x
measured over the registry; the native integration time cancels). With
the baseline held fixed the fit is 1/sqrt(2) "too good" by construction,
since the sqrt(2) in the tables is the out-of-transit baseline.

Transit shape: when the planet is in the frozen target registry
(lazuli_exo.targets) and has a limb-darkening table
(precompute/make_ld_tables.py), the model is a quadratically limb-darkened
transit with the real geometry: z(t) is the exact sky-projected separation
of a Keplerian orbit (orbit.projected_separation; Kepler's equation solved
by Newton iteration, mid-transit at inferior conjunction so z(0) = b with
the eccentricity factor). The registry t14_s is the measured archive
duration and sets the noise and the window, but the geometric duration
t14_geom from (P, a/R*, k, i, e, w) can differ (HAT-P-11 b: 7184 s vs
8480 s), so the time fed to the orbit is scaled by t14_geom / t14_s and
the model's first-to-fourth-contact duration equals t14_s
(meta["t14_geom_s"], meta["time_stretch"]). Per-band (u1, u2) interpolate
the LD table at the band's bin centers. Otherwise (synthetic planets,
missing LD table) it falls back to a trapezoid for a central transit
(b = 0, no limb darkening) with ingress t14 * k / (1 + k), k = sqrt(depth).
meta["shape"] records which model was used. Band depths vary; the geometry
is shared across bands.
"""
from dataclasses import dataclass, field

import numpy as np

from . import orbit, tables


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


def transit_flux(z, k, u1, u2, n_r=1601):
    """Normalized flux of a quadratically limb-darkened star occulted by
    an opaque disk of radius ratio `k` at projected separations `z`
    (stellar radii).

    Numpy-only numerical disk integration: per separation, quadrature over
    the occulted radial band ``[|z|-k, |z|+k]`` of the analytic overlap-arc
    weight times the intensity profile. The Chebyshev-type substitution
    r = c - h*cos(psi) absorbs the square-root behavior at both band edges
    (planet edge and stellar limb). Measured against the analytic
    uniform-source solution, the maximum absolute flux error at the default
    n_r is <= 1e-6 for k <= 0.2 (1.7e-6 at k = 0.3); it grows roughly as
    k^2 and shrinks as n_r^-1.5. Flux is evaluated at exposure centers by the
    caller — there is no exposure-time integration.
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
        e = 0.0 if target["ecc"] is None else target["ecc"]
        w_deg = 90.0 if target["w_deg"] is None else target["w_deg"]
        a_rs, inc_deg, period = (target["a_over_rs"], target["inc_deg"],
                                 target["period_d"])
        b = (a_rs * np.cos(np.radians(inc_deg)) * (1.0 - e**2)
             / (1.0 + e * np.sin(np.radians(w_deg))))
        # t14_s is the measured archive duration (noise + window); stretch
        # the orbit's clock so the model transit lasts t14_s as well
        k_white = np.sqrt(max(_band_combine(obs.depth, obs.depth_err, 1)[1][0],
                              0.0))
        t14_geom = orbit.transit_duration_s(period, a_rs, k_white, inc_deg,
                                            e, w_deg)
        stretch = t14_geom / t14 if t14_geom > 0 else 1.0
        z = orbit.projected_separation(time_s * stretch, period, a_rs,
                                       inc_deg, e, w_deg)

        u1 = np.array([np.interp(obs.wavelength[g], wl_ld, u1_ld).mean()
                       for g in groups])
        u2 = np.array([np.interp(obs.wavelength[g], wl_ld, u2_ld).mean()
                       for g in groups])
        flux_model = np.stack([
            transit_flux(z, np.sqrt(max(d, 0.0)), u1[i], u2[i])
            for i, d in enumerate(depth)])
        meta.update(shape="quadratic-ld", b=float(b),
                    a_over_rs=float(a_rs), t14_geom_s=float(t14_geom),
                    time_stretch=float(stretch))
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
