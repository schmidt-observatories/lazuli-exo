"""Instrument-agnostic transit / flux models.

`FluxModel` maps time to normalized relative flux; `TransitModel` wraps the
`jaxoplanet` package. These models carry no dependency on any specific
instrument: a WCC or IFS simulator consumes a FluxModel through its
`relative_flux(time)` interface.
"""

import numpy as np

_JAXOPLANET_HINT = (
    "TransitModel requires the 'jaxoplanet' package. "
    "Install it with: pip install lazuli-exo[transit]"
)

# jaxoplanet implements polynomial limb darkening (Agol, Luger &
# Foreman-Mackey 2020); law name -> required coefficient count.
_POLY_LIMB_DARK = {"uniform": 0, "linear": 1, "quadratic": 2}

_PANDORA_HINT = (
    "MoonTransitModel requires the 'pandoramoon' package. "
    "Install it with: pip install lazuli-exo[pandora]"
)


class FluxModel:
    """Base class: map time -> relative flux, normalized to 1.0 out of event."""

    def relative_flux(self, time):
        raise NotImplementedError

    def plot(self, time, ax=None, **kw):
        """Plot ``relative_flux(time)``. Delegates to plotting.plot_flux_model."""
        from .plotting import plot_flux_model

        return plot_flux_model(self, time, ax=ax, **kw)


class TransitModel(FluxModel):
    """Exoplanet transit light-curve model backed by `jaxoplanet`.

    Parameters follow the batman.TransitParams convention: t0 (center),
    per (period), rp (Rp/R*), a (a/R*), inc (deg), ecc, w (deg), limb_dark,
    u (coeffs). Only polynomial limb-darkening laws are supported
    ("uniform", "linear", "quadratic").
    """

    def __init__(
        self,
        t0=0.0,
        per=1.0,
        rp=0.1,
        a=15.0,
        inc=87.0,
        ecc=0.0,
        w=90.0,
        limb_dark="quadratic",
        u=(0.1, 0.3),
    ):
        self.t0, self.per, self.rp, self.a = t0, per, rp, a
        self.inc, self.ecc, self.w = inc, ecc, w
        self.limb_dark, self.u = limb_dark, list(u)
        self.check_params()

    @classmethod
    def from_planet(
        cls, name, df=None, require_transit=True, limb_dark="quadratic", u=(0.1, 0.3)
    ):
        """Build a TransitModel from a NASA Exoplanet Archive (PSCompPars) row.

        Looks up ``name`` in ``df`` (or the local cache via
        load_exoplanet_archive if None) and maps archive columns to
        transit-model parameters. ``pl_orbper`` is required; other fields
        fall back to circular-orbit / direct-ratio defaults when missing.
        Limb-darkening is not in the archive, so it stays a user argument.

        If ``require_transit`` is True (default), a planet flagged as
        non-transiting in the archive (``tran_flag == 0``) raises ValueError —
        a transit model is not meaningful for it. Pass ``require_transit=False``
        to build one anyway. When ``tran_flag`` is absent/unknown, the check is
        skipped.

        Note: ``t0`` is taken from ``pl_tranmid`` (absolute BJD). For a
        relative-time light curve, pass a ``time`` array spanning that epoch or
        set ``t0=0`` after construction.
        """
        if df is None:
            from .archive import load_exoplanet_archive

            df = load_exoplanet_archive()

        def _norm(s):
            return str(s).lower().replace(" ", "")

        matches = df[df["pl_name"].map(_norm) == _norm(name)]
        if len(matches) == 0:
            raise ValueError(f"Planet {name!r} not found in archive table")
        row = matches.iloc[0]

        if require_transit:
            tran = row.get("tran_flag")
            known = tran is not None and not (
                isinstance(tran, float) and np.isnan(tran)
            )
            if known and int(tran) == 0:
                raise ValueError(
                    f"Planet {name!r} is not flagged as transiting "
                    "(tran_flag=0); pass require_transit=False to build a "
                    "model anyway"
                )

        def _val(col, default):
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        per = row.get("pl_orbper")
        if per is None or (isinstance(per, float) and np.isnan(per)):
            raise ValueError(
                f"Planet {name!r} has no pl_orbper; cannot build a transit model"
            )

        import astropy.constants as const
        import astropy.units as units

        rjup_rsun = float((const.R_jup / const.R_sun).decompose().value)
        au_rsun = float((1 * units.au).to(units.R_sun).value)

        rp = _val("pl_ratror", np.nan)
        if np.isnan(rp):
            rp = _val("pl_radj", np.nan) * rjup_rsun / _val("st_rad", np.nan)

        a = _val("pl_ratdor", np.nan)
        if np.isnan(a):
            a = _val("pl_orbsmax", np.nan) * au_rsun / _val("st_rad", np.nan)

        return cls(
            t0=_val("pl_tranmid", 0.0),
            per=float(per),
            rp=rp,
            a=a,
            inc=_val("pl_orbincl", 90.0),
            ecc=_val("pl_orbeccen", 0.0),
            w=_val("pl_orblper", 90.0),
            limb_dark=limb_dark,
            u=u,
        )

    def build_system(self):
        """Return ``(system, light_curve_fn)`` for the jaxoplanet backend.

        Note this enables jax's float64 mode globally: without it jax runs in
        float32 and the light curve is only accurate to ~1e-4.
        """
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            from jaxoplanet.light_curves import limb_dark_light_curve
            from jaxoplanet.orbits.keplerian import Central, System
        except ImportError as exc:  # pragma: no cover - exercised via hint
            raise ImportError(_JAXOPLANET_HINT) from exc

        # Fix the star at R* = 1 so a/R* and Rp/R* come out as given.
        central = Central.from_orbital_properties(
            period=self.per, semimajor=self.a, radius=1.0
        )
        body = dict(
            period=self.per,
            time_transit=self.t0,
            inclination=np.deg2rad(self.inc),
            radius=self.rp,
        )
        if self.ecc != 0.0:
            body.update(eccentricity=self.ecc, omega_peri=np.deg2rad(self.w))
        return System(central).add_body(**body), limb_dark_light_curve

    def check_params(self):
        """Reject non-finite or unphysical orbital parameters.

        Requires finite t0, per, rp, a, inc, ecc, w with per, rp, a > 0 and
        0 <= ecc < 1. Checked on construction and again before evaluation, so an
        archive row with missing values (NaN rp or a) fails loudly.
        """
        vals = dict(
            t0=self.t0,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
        )
        bad = [k for k, v in vals.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"non-finite transit parameter(s): {bad}")
        if not (self.per > 0 and self.rp > 0 and self.a > 0 and 0 <= self.ecc < 1):
            raise ValueError(
                "unphysical transit parameters: need per, rp, a > 0 and "
                f"0 <= ecc < 1; got per={self.per}, rp={self.rp}, a={self.a}, "
                f"ecc={self.ecc}"
            )

    def check_limb_dark(self):
        """Validate ``limb_dark`` and ``u`` against the supported polynomial laws."""
        n_coeff = _POLY_LIMB_DARK.get(self.limb_dark)
        if n_coeff is None:
            raise ValueError(
                f"limb_dark={self.limb_dark!r} is not supported: the "
                "jaxoplanet backend implements polynomial laws only "
                f"({', '.join(_POLY_LIMB_DARK)})"
            )
        if len(self.u) != n_coeff:
            raise ValueError(
                f"limb_dark={self.limb_dark!r} needs {n_coeff} "
                f"coefficient(s), got {len(self.u)}"
            )

    def relative_flux(self, time):
        time = np.asarray(time, dtype=float)
        self.check_params()
        self.check_limb_dark()
        system, limb_dark_light_curve = self.build_system()
        flux = limb_dark_light_curve(system, *self.u)(time)
        return 1.0 + np.asarray(flux).reshape(time.shape)


class MoonTransitModel(FluxModel):
    """Exomoon transit light-curve model backed by ``pandoramoon``.

    Parameters use physical units and are converted internally to the
    pandora-native units (meters / kg / stellar radii).

    Parameters
    ----------
    R_star : float
        Stellar radius in **solar radii**.
    u : tuple(float, float)
        Quadratic limb-darkening coefficients ``(u1, u2)``.
    t0 : float
        Transit center, days.
    per : float
        Planet (barycenter) orbital period, days.
    Rp : float
        Planet radius in **Earth radii**.
    a : float
        Planet semi-major axis in **AU**.
    b : float
        Impact parameter (dimensionless).
    ecc, w : float
        Eccentricity and argument of periastron (deg) of the planet orbit.
    M_planet : float
        Planet mass in **Earth masses**.
    Rm : float
        Moon radius in **Earth radii**.
    per_moon : float
        Moon orbital period around the planet, days.
    M_moon : float
        Moon mass in **Earth masses**.
    tau : float
        Moon orbital phase at ``t0`` (0-1).
    Omega, i_moon : float
        Moon orbit longitude of ascending node and inclination (deg).
    ecc_moon, w_moon : float
        Moon orbit eccentricity and argument of periastron (deg).
    supersampling_factor : int
        Integration supersampling (1 = none).

    Notes
    -----
    The moon's orbital semi-major axis is derived by pandora from the masses
    and ``per_moon`` via Kepler's third law, so masses are required physical
    inputs. ``relative_flux`` returns the **total** (planet+moon) curve;
    ``components`` returns ``(total, planet, moon)``.
    """

    def __init__(self, R_star=1.0, u=(0.4, 0.3), t0=0.0, per=10.0, Rp=11.2,
                 a=0.1, b=0.3, ecc=0.0, w=0.0, M_planet=318.0, Rm=1.0,
                 per_moon=2.5, M_moon=1.0, tau=0.5, Omega=0.0, i_moon=90.0,
                 ecc_moon=0.0, w_moon=0.0, supersampling_factor=1):
        self.R_star, self.u = R_star, tuple(u)
        self.t0, self.per, self.Rp, self.a, self.b = t0, per, Rp, a, b
        self.ecc, self.w, self.M_planet = ecc, w, M_planet
        self.Rm, self.per_moon, self.M_moon = Rm, per_moon, M_moon
        self.tau, self.Omega, self.i_moon = tau, Omega, i_moon
        self.ecc_moon, self.w_moon = ecc_moon, w_moon
        self.supersampling_factor = supersampling_factor

    def _pandora_params(self, time):
        try:
            import pandoramoon as pandora
        except ImportError as exc:  # pragma: no cover - exercised via hint
            raise ImportError(_PANDORA_HINT) from exc
        import astropy.constants as const
        import astropy.units as units

        R_sun_m = const.R_sun.to(units.m).value
        R_earth_m = const.R_earth.to(units.m).value
        M_earth_kg = const.M_earth.to(units.kg).value
        au_m = (1 * units.au).to(units.m).value
        R_star_m = self.R_star * R_sun_m

        p = pandora.model_params()
        p.R_star = R_star_m
        p.u1, p.u2 = self.u
        p.per_bary = self.per
        p.a_bary = self.a * au_m / R_star_m
        p.r_planet = self.Rp * R_earth_m / R_star_m
        p.b_bary = self.b
        p.w_bary = self.w
        p.ecc_bary = self.ecc
        p.t0_bary = self.t0
        p.t0_bary_offset = 0.0
        p.M_planet = self.M_planet * M_earth_kg
        p.r_moon = self.Rm * R_earth_m / R_star_m
        p.per_moon = self.per_moon
        p.tau_moon = self.tau
        p.Omega_moon = self.Omega
        p.i_moon = self.i_moon
        p.ecc_moon = self.ecc_moon
        p.w_moon = self.w_moon
        p.M_moon = self.M_moon * M_earth_kg
        p.epoch_distance = self.per
        p.supersampling_factor = self.supersampling_factor
        p.time = time
        return pandora, p

    def components(self, time):
        """Return ``(total, planet, moon)`` normalized light curves."""
        time = np.asarray(time, dtype=float)
        pandora, params = self._pandora_params(time)
        total, planet, moon = pandora.moon_model(params).light_curve(time)
        return (np.asarray(total), np.asarray(planet), np.asarray(moon))

    def relative_flux(self, time):
        return self.components(time)[0]

    def default_time(self, n=600, span=1.5):
        """Time grid of ``n`` points centered on ``t0``, spanning +/- ``span``
        transit durations (T14 estimated from the planet orbit)."""
        import astropy.constants as const
        import astropy.units as units
        R_sun_m = const.R_sun.to(units.m).value
        au_m = (1 * units.au).to(units.m).value
        a_over_rstar = self.a * au_m / (self.R_star * R_sun_m)
        t14 = self.per / np.pi * np.arcsin(1.0 / a_over_rstar)
        half = span * t14
        return np.linspace(self.t0 - half, self.t0 + half, int(n))

    def plot(self, time=None, show_components=True, ax=None, **kw):
        """Plot the moon-transit light curve (auto time window if ``time`` is
        None). Delegates to plotting.plot_moon_transit."""
        from .plotting import plot_moon_transit
        return plot_moon_transit(self, time=time, show_components=show_components,
                                 ax=ax, **kw)
