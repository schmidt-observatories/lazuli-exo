Transit light curves
====================

:meth:`~lazuli_exo.SimulatedSpectrum.light_curves` turns a simulated spectrum
into the time domain — one white-light curve, or several split by wavelength
band.

.. code-block:: python

   obs = lazuli_exo.observe("WASP-39 b", scenario="clear", n_transits=1)

   lc = obs.light_curves(n_groups=1, seed=0)    # white light
   lcb = obs.light_curves(n_groups=5, seed=0)   # five wavelength bands

``n_groups`` splits the binned spectrum into contiguous, equal-count groups,
inverse-variance combined. The geometry is shared across bands; the depth and
limb darkening are not.

The observation window
----------------------

Sampling spans :math:`\pm t_{14}` around mid-transit — the equal
out-of-transit baseline the noise tables assume (see :doc:`noise`) — at the
instrument's native integration time ``t_obs_s``. If that would exceed
``max_points`` samples (default 2000), the cadence is coarsened; it is never
made finer than the real integration time.

The occultation model
---------------------

:func:`~lazuli_exo.lightcurve.transit_flux` computes the normalized flux of a
quadratically limb-darkened star occulted by an opaque disk of radius ratio
:math:`k` at projected separations :math:`z` in stellar radii.

It is a numpy-only numerical disk integration: per separation, quadrature over
the occulted radial band :math:`[|z|-k,\, |z|+k]` of the analytic overlap-arc
weight times the intensity profile. A Chebyshev-type substitution
:math:`r = c - h\cos\psi` absorbs the square-root behaviour at both band edges
— the planet edge and the stellar limb.

**It is validated, not asserted.** Against the analytic uniform-source
solution (:math:`u_1 = u_2 = 0`), the maximum absolute flux error at the
default resolution is :math:`\sim 3 \times 10^{-7}` for :math:`k = 0.15` —
about 13 ppm of the transit depth, and the fractional error is independent of
:math:`k`. Notebook :doc:`../notebooks/04_light_curves` runs that comparison
in full.

Geometry
--------

For planets in the frozen registry (see :doc:`tables`), the geometry is real
rather than assumed:

- :math:`b` from :math:`a/R_\star` and :math:`\cos i`, with an eccentricity
  factor :math:`f_e = (1 + e\sin\omega)/\sqrt{1-e^2}`.
- :math:`z(t)` is the exact sky-projected separation of the Keplerian
  orbit (Kepler's equation solved by Newton iteration), with mid-transit at
  inferior conjunction so :math:`z(0) = b` exactly.
- The registry :math:`t_{14}` is the measured archive duration and sets the
  noise and the window; the geometric duration from
  :math:`(P, a/R_\star, k, i, e, \omega)` can differ (HAT-P-11 b: 7184 s vs
  8480 s), so the orbit clock is scaled by their ratio and the model's
  first-to-fourth-contact duration equals :math:`t_{14}`
  (``meta["t14_geom_s"]``, ``meta["time_stretch"]``).

Limb-darkening coefficients come from per-planet tables computed with
exotic-ld (PHOENIX grid, trilinear interpolation, 48 wavelength chunks),
interpolated to each band's bin centres. They vary strongly across the band —
for WASP-39 b, :math:`u_1` runs from 0.78 in the blue to 0.18 in the near-IR —
which is why banded curves change *shape*, not just depth.

The trapezoid fallback
----------------------

A planet with no registry entry, missing orbital parameters, or no
limb-darkening table falls back to a :math:`b = 0` trapezoid with no limb
darkening and ingress :math:`t_{14}\,k/(1+k)`, where
:math:`k = \sqrt{\mathrm{depth}}`.

``meta["shape"]`` always records which model was used, and ``u1``/``u2`` are
``None`` on the fallback path:

.. code-block:: python

   lc.meta["shape"]     # 'quadratic-ld'  or  'trapezoid'
   lc.meta["b"]         # present only for 'quadratic-ld'
   lc.u1                # None on the fallback

Per-point noise
---------------

The per-sample scatter is **derived from the band's depth precision**, not
computed independently:

.. math::

   \sigma_\mathrm{pt} = \sigma_\mathrm{band}
   \sqrt{ t_{14} / 2c }

for cadence :math:`c`. This keeps the light curves consistent with the same
equal out-of-transit baseline the spectra assume — the two views of an
observation cannot silently disagree about how much signal there was.

It also means the light curves inherit every caveat from :doc:`noise`,
including the unbounded :math:`1/\sqrt{N}` scaling.

Fields on ``LightCurveSet``
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``time_hr``
     - Sample times, hours from mid-transit.
   * - ``flux_model``
     - ``(n_groups, n_time)`` noiseless normalized flux.
   * - ``flux``
     - ``(n_groups, n_time)`` one noisy realization for the given seed.
   * - ``sigma_point``
     - ``(n_groups,)`` per-sample 1-sigma flux noise.
   * - ``depth``
     - ``(n_groups,)`` band-combined transit depth.
   * - ``wl_lo`` / ``wl_hi``
     - ``(n_groups,)`` band edges, in µm, at bin centres.
   * - ``labels``
     - ``['white light']``, or ``'0.40-0.53 um'``-style band labels.
   * - ``cadence_s`` / ``t14_s``
     - Sampling cadence and transit duration, seconds.
   * - ``u1`` / ``u2``
     - ``(n_groups,)`` quadratic limb darkening; ``None`` on the fallback.
   * - ``meta``
     - Planet, scenario, ``n_transits``, ``shape``, and geometry.
