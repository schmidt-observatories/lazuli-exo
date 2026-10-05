lazuli-exo: exoplanet transmission spectroscopy with the Lazuli IFS
====================================================================

.. image:: _static/logo_schmidt_observatory_system.png
   :alt: Schmidt Observatory System
   :width: 320px

**lazuli-exo** simulates exoplanet transmission spectroscopy with the
**Integral Field Spectrograph (IFS)** on *Lazuli*, the flagship instrument of
the Schmidt Observatory System. It returns the model transmission spectrum of
a transiting planet binned to the instrument, with photon/read-noise error
bars for a chosen number of transits, a spectral-template S/N, and (for the
showcase planets) simulated transit light curves.

Two ways in:

- **Showcase tables** — 14 well-known planets × 5 atmosphere scenarios,
  precomputed with POSEIDON: :func:`lazuli_exo.observe`.
- **Parametric model** — any transiting planet from its star, radius, mass,
  :math:`T_\mathrm{eq}`, transit duration and atmosphere knobs ([M/H], C/O,
  :math:`\mu`, cloud deck, haze), with slicersim computing the noise live:
  :func:`lazuli_exo.simulate`.

.. warning::

   These are **isothermal, equilibrium-chemistry toy models with statistical
   noise only** — comparison material, *not* calibrated detection
   significances or observing-time forecasts. Read :doc:`user_guide/concepts`
   before quoting any number. The limits are specific and knowable, and they
   are all written down.

The runtime is **numpy-only with its lookup tables shipped as package data**.
POSEIDON, slicersim, jax and astropy are never imported by
``import lazuli_exo``; slicersim and sncosmo are needed at call time only by
the parametric model's live noise.

.. note::

   New here? Read :doc:`installation`, then :doc:`quickstart`. The
   :doc:`tutorials` are runnable Jupyter notebooks that cover every feature
   end to end; notebooks 6–9 cover the parametric model.

Highlights
----------

- **Atmosphere scenarios** — ``clear``, ``cloudy``, ``hazy``,
  ``high_metallicity`` and a featureless ``flat`` baseline for 14 showcase
  targets.
- **Parametric atmospheres** — a POSEIDON-derived grid of photospheric
  pressure :math:`\zeta(\lambda)` on (T, [M/H], C/O) plus analytic scale
  height, slant column, cloud deck and haze; validated against the showcase
  tables (:doc:`user_guide/parametric`).
- **Instrument noise** — per-channel single-transit depth uncertainty from
  slicersim, Vega→AB handled, readout desaturated per target, scaled by
  :math:`1/\sqrt{N_\mathrm{transits}}` and binned in inverse variance.
- **Spectral-template S/N** — a matched-template amplitude statistic against
  the best-fit flat line, overall and per molecular band mask.
- **Transit light curves** — quadratically limb-darkened, numpy-only, for the
  showcase planets.
- **Web ready** — a form/report layer (:mod:`lazuli_exo.forms`,
  :mod:`lazuli_exo.report`) that the local Flask applet and the Lazuli
  simulators site share (:doc:`user_guide/webapp`).

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
