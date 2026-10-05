The template statistic
======================

:meth:`~lazuli_exo.SimulatedSpectrum.detect` is the most easily misread part
of this package, so this page states precisely what it computes before saying
anything about how to use it.

What it computes
----------------

Given the binned model depth :math:`d_i` and uncertainties
:math:`\sigma_i`, with weights :math:`w_i = \sigma_i^{-2}` and the
weighted-mean depth :math:`\bar{d} = \sum w_i d_i / \sum w_i`:

.. math::

   \mathrm{S/N} = \sqrt{ \sum_i w_i \left( d_i - \bar{d} \right)^2 }

This is the expected S/N of fitting the **noiseless** model spectrum as a
template against the best-fit flat line, assuming independent Gaussian errors
and no nuisance parameters beyond the mean. It is returned as
``template_snr_diagonal``.

Note what it is measured on: ``depth``, the noiseless model — not a noisy
realization. Two calls with the same arguments always give the same number.

What it is not
--------------

.. warning::

   ``template_snr_diagonal`` is **not** a calibrated Gaussian-equivalent
   detection significance. There is:

   - no defined null/alternative likelihood ratio,
   - no nuisance model beyond a single free mean,
   - no covariance — errors are assumed diagonal,
   - no false-positive calibration.

   A value of 5 does not mean "5σ".

Per-band values
---------------

``band_snr`` evaluates the same sum restricted to a molecule's wavelength
mask, relative to the same *global* :math:`\bar{d}`:

.. code-block:: python

   from lazuli_exo.detect import BAND_MASKS

   BAND_MASKS
   # {'Na':  [(0.575, 0.605)],
   #  'K':   [(0.755, 0.785)],
   #  'H2O': [(0.90, 1.00), (1.10, 1.20), (1.30, 1.55)],
   #  'CH4': [(0.86, 0.91), (1.10, 1.20), (1.60, 1.70)],
   #  'CO2': [(1.42, 1.46), (1.56, 1.62)]}

.. warning::

   These are **band-restricted template norms, not molecule detections.**
   Each includes absorption from every other species in the mask, plus
   haze, cloud, and continuum structure, plus the offset of the masked region
   from the global mean. The masks *overlap* — H2O and CH4 both cover
   1.10–1.20 µm — so their values are not independent.

   CO and NH3 are present in the models but have no band masks in the
   0.4–1.7 µm range, so they are never reported. **Their absence from the
   output is not a non-detection.**

A worked warning: why ``hazy`` can beat ``clear``
-------------------------------------------------

The statistic rewards *any* departure from a flat line, including one that
carries no molecular information at all.

For WASP-39 b at one transit, ``hazy`` scores **132.7** against ``clear``'s
**123.0** — even though every individual band norm is *lower* in ``hazy``
(H2O drops from 82.8 to 58.5 as the molecular bands are partly veiled). The
extra amplitude is entirely outside the band masks, in the blue continuum
below 0.575 µm where the haze's Rayleigh-like slope is steepest:

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 20 18

   * - Scenario
     - Total S/N
     - In-mask
     - Out-of-mask
     - Out fraction
   * - ``clear``
     - 123.0
     - 102.2
     - 68.5
     - 0.31
   * - ``hazy``
     - 132.7
     - 83.9
     - 102.8
     - 0.60

Read ``template_snr_diagonal`` as **"how much structure is there"**, never as
**"how much atmosphere is detectable"**. Notebook
:doc:`../notebooks/02_scenarios` reproduces this table from the tables
themselves.

How to use it responsibly
-------------------------

The statistic is genuinely useful for *relative* questions under a fixed set
of assumptions:

- Ranking targets against one another for the same scenario.
- Comparing scenarios for the same target.
- Tracking how a quantity moves with ``n_transits`` or ``R``.

It is not useful for absolute statements — "detectable", "5σ", "N transits
required" — and the package deliberately gives you no function that produces
one.

The upgrade path
----------------

The rigorous replacement is a **likelihood ratio against per-molecule
leave-one-out model tables**: for each species, a model computed with that
species removed, and a ratio test against the full model. This slots into the
existing table schema without any API change — the schema already allows
arbitrary scenario names per planet.
