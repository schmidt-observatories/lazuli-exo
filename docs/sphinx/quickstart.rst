Quick start
===========

This page walks through the core workflow. Everything below uses only the
public API exported at the top level of ``lazuli_exo``.

The mental model
----------------

.. code-block:: text

   precomputed tables                      runtime (numpy only)
   ------------------                      --------------------
   atmo/    zeta grid (opacities)  ─┐
   noise/   1-transit sigma(lambda) ─┼──►  observe()  ──►  SimulatedSpectrum
   ld/      limb darkening          ─┤                       ├─ .detect()
   targets  frozen NEA registry     ─┘                       ├─ .light_curves() ─ .plot()
                                                             ├─ .simulate_points()
                                                             ├─ .plot()
                                                             └─ .plot_noise()

``observe()`` evaluates the parametric model at the registry values of the
showcase planet (the same model ``simulate()`` runs for any planet), bins it
and the precomputed noise table onto a common constant-resolving-power grid,
and scales the uncertainty by :math:`1/\sqrt{N_\mathrm{transits}}`.

1. See what is available
------------------------

.. code-block:: python

   import lazuli_exo

   lazuli_exo.list_planets()
   # ['GJ 1214 b', 'GJ 3470 b', 'HAT-P-11 b', 'HD 189733 b', 'HD 209458 b',
   #  'K2-18 b', 'LHS 1140 b', 'TOI-270 d', 'TRAPPIST-1 e', 'WASP-107 b',
   #  'WASP-12 b', 'WASP-121 b', 'WASP-39 b', 'WASP-96 b']

   lazuli_exo.list_scenarios("WASP-39 b")
   # ['clear', 'cloudy', 'flat', 'hazy', 'high_metallicity']

2. Simulate an observation
--------------------------

.. code-block:: python

   obs = lazuli_exo.observe("WASP-39 b", scenario="cloudy",
                            n_transits=5, R=100)

   obs.wavelength        # bin centers [um]
   obs.depth             # binned model transit depth (noiseless)
   obs.depth_err         # per-bin 1-sigma, already scaled by 1/sqrt(N)
   obs.model_wavelength  # native-resolution model, for overplotting
   obs.model_depth
   obs.flat_depth        # geometric (Rp/Rs)^2 from the registry radii

``R`` has two limits, both set near 1.0 µm where the pixels are coarsest in
resolving power. Above the instrument resolving power (about 100)
``observe()`` warns that neighbouring bins are correlated by the line-spread
function; above the native channel sampling (about 230) it raises a
``ValueError``. See :doc:`user_guide/noise`.

3. Draw a noisy realization
---------------------------

``depth`` and ``depth_err`` are deterministic; noise is drawn on demand.

.. code-block:: python

   points = obs.simulate_points(seed=0)   # reproducible for a given seed

4. Detection diagnostics
------------------------

.. code-block:: python

   result = obs.detect()
   result["template_snr_diagonal"]   # overall template amplitude S/N
   result["band_snr"]                # {'Na': ..., 'K': ..., 'H2O': ..., ...}

.. warning::

   ``template_snr_diagonal`` is a **matched-template amplitude S/N against the
   best-fit flat line**, not a calibrated detection significance. The per-band
   values are the same statistic restricted to fixed wavelength masks: they
   mix every species in the mask plus haze, cloud, and continuum structure,
   and the masks overlap. They are **not molecule detections**. See
   :doc:`user_guide/detect`.

5. Transit light curves
-----------------------

.. code-block:: python

   lc = obs.light_curves(n_groups=1, seed=0)     # white light
   lc.time_hr, lc.flux_model[0], lc.flux[0]

   lcb = obs.light_curves(n_groups=5, seed=0)    # split into 5 bands
   lcb.labels        # ['0.40-0.53 um', '0.54-0.71 um', ...]
   lcb.u1, lcb.u2    # per-band quadratic limb darkening
   lcb.meta["shape"] # 'quadratic-ld' or 'trapezoid' (fallback)

See :doc:`user_guide/lightcurves`.

6. Plot
-------

Requires the ``[plot]`` extra. Every plot method takes an optional ``ax``
and returns the matplotlib Axes, so plots overlay and tile.

.. code-block:: python

   obs.plot(seed=0)                 # noiseless model + one simulated realization
   obs.plot_noise()                 # per-bin 1-sigma error against wavelength
   lcb.plot()                       # the banded light curves, stacked

   # overlay scenarios on one Axes, model lines only, in ppm
   fig, ax = plt.subplots()
   for s in ["clear", "cloudy", "hazy"]:
       lazuli_exo.observe("WASP-39 b", s).plot(ax=ax, show_points=False, units="ppm")

See :mod:`lazuli_exo.plotting` for the keywords (``units``, ``label``,
``show_model``, ``show_points``, ``offset``).

7. Check the provenance
-----------------------

Every observation carries the settings of the tables it was built from. This
is the first place to look when a number surprises you.

.. code-block:: python

   obs.meta["params"]       # the parameter dict the model was evaluated at
   obs.meta["atmosphere"]   # derived: g_ms2, H_m, H_over_Rp, mu_used, T_chem_K,
                            # T_clamped, feature_amplitude_ppm, tlse
   obs.meta["noise"]        # slicersim settings: instrument, nmd_used, t14_s,
                            # fully_desaturated, nsaturated_detpx, git_commit
   obs.meta["instrument_response"]   # LSF sigma, R_instr range, photon weighting
   obs.meta["tables_dir"]

Where to go next
----------------

- :doc:`user_guide/index`: what each subsystem does, and what it does not.
- :doc:`user_guide/plotting`: every plot method with its figure.
- :doc:`tutorials`: nine runnable notebooks.
- :doc:`api/index`: the full auto-generated API reference.


Any planet: the parametric model
--------------------------------

The showcase tables cover 14 planets. For anything else, describe the system
and let :func:`lazuli_exo.simulate` build the spectrum (needs ``slicersim``
and ``sncosmo`` installed):

.. code-block:: python

   import lazuli_exo

   p = lazuli_exo.presets.showcase("WASP-39 b", "clear")   # a dict you can edit, or build by hand
   p.update(T_eq_K=900.0, mu=None, log_P_cloud_bar=-2.0)    # cooler, chemistry mu, deck at 10 mbar
   obs = lazuli_exo.simulate(p, R=100, n_transits=5)

Bring your own spectrum
-----------------------

If you already have a forward model you trust (POSEIDON, petitRADTRANS,
PICASO, ...), skip the surrogate and pass its transit depth directly. The
star, the live slicersim noise, stellar contamination and the instrument
response are applied exactly as for the parametric model; only the star
keys are required.

.. code-block:: python

   # wl_um increasing, depth = (R_p/R_s)^2 (not ppm), covering 0.40-1.70 um
   # with a little margin (the LSF reaches a few pixels past the band)
   star = dict(teff_K=5485., logg=4.45, feh=0.0, mag=10.66, t14_s=10092.)
   obs = lazuli_exo.simulate(star, spectrum=(wl_um, depth), R=100, n_transits=5)
   obs.meta["atmosphere"]["source"]     # "user"

   obs.detect()["template_snr_diagonal"]
   obs.meta["atmosphere"]["H_m"], obs.meta["noise"]["mag_ab"]

``obs`` is the same :class:`~lazuli_exo.observe.SimulatedSpectrum` as above,
so ``detect()``, ``plot()`` and ``simulate_points()`` work unchanged. The
parameters and the physics are described in :doc:`user_guide/parametric`;
notebooks 6–9 of the :doc:`tutorials` work through them.
