Quick start
===========

This page walks through the core workflow. Everything below uses only the
public API exported at the top level of ``lazuli_exo``.

The mental model
----------------

.. code-block:: text

   precomputed tables                      runtime (numpy only)
   ------------------                      --------------------
   models/  transmission spectrum  ─┐
   noise/   1-transit sigma(lambda) ─┼──►  observe()  ──►  SimulatedSpectrum
   ld/      limb darkening          ─┤                       ├─ .detect()
   star/    stellar F_lambda        ─┘                       ├─ .light_curves()
                                                             ├─ .simulate_points()
                                                             └─ .plot()

There is no model evaluation at runtime. ``observe()`` reads the precomputed
model and noise tables, bins both onto a common constant-resolving-power grid,
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
   obs.flat_depth        # (Rp/Rs)^2 baseline from the 'flat' table

``R`` is capped by the native sampling of the noise tables — roughly 230 at
the blue end. Asking for more raises a ``ValueError`` naming the empty bin.
See :doc:`user_guide/noise`.

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
   values are the same statistic restricted to fixed wavelength masks — they
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

Requires the ``[plot]`` extra.

.. code-block:: python

   obs.plot(seed=0)   # noiseless model + one simulated realization

7. Check the provenance
-----------------------

Every observation carries the settings of the tables it was built from. This
is the first place to look when a number surprises you.

.. code-block:: python

   obs.meta["model"]   # POSEIDON settings: T_iso_K, T_clamped, cloud_*, log_Met
   obs.meta["noise"]   # slicersim settings: instrument, nmd_used, t14_s,
                       # fully_desaturated, nsaturated_detpx
   obs.meta["tables_dir"]

Where to go next
----------------

- :doc:`user_guide/index` — what each subsystem does and, more importantly,
  what it does not.
- :doc:`tutorials` — five runnable notebooks.
- :doc:`api/index` — the full auto-generated API reference.


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

   obs.detect()["template_snr_diagonal"]
   obs.meta["atmosphere"]["H_m"], obs.meta["noise"]["mag_ab"]

``obs`` is the same :class:`~lazuli_exo.observe.SimulatedSpectrum` as above,
so ``detect()``, ``plot()`` and ``simulate_points()`` work unchanged. The
parameters and the physics are described in :doc:`user_guide/parametric`;
notebooks 6–9 of the :doc:`tutorials` work through them.
