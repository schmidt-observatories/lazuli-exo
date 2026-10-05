The noise model
===============

Every error bar ``lazuli_exo`` reports traces back to one array per planet:
``depth_err_1transit(λ)``, the single-transit transit-depth uncertainty
computed offline with slicersim for the Lazuli IFS.

What ``observe()`` does with it
-------------------------------

Two operations, both simple and both worth knowing exactly:

1. **Bin** the native noise channels onto the requested constant-R grid, in
   inverse variance:

   .. math::

      \sigma_\mathrm{bin} = \left( \sum_{i \in \mathrm{bin}}
      \sigma_i^{-2} \right)^{-1/2}

2. **Scale** by the number of transits:

   .. math::

      \sigma_\mathrm{depth} = \sigma_\mathrm{bin} /
      \sqrt{N_\mathrm{transits}}

That is the entire noise model at runtime. The binned model ``depth`` is
untouched by ``n_transits``.

.. warning::

   The :math:`1/\sqrt{N}` scaling has **no floor**. There is no per-transit
   systematics term and no red noise, so the predicted precision improves
   without limit as transits accumulate. Any "number of transits needed"
   computed from these outputs is a lower bound, optimistic by an amount this
   model cannot tell you.

The resolution ceiling
----------------------

``R`` is limited by the *coarsest* part of the native noise sampling, not its
median. For the shipped tables the native sampling runs from about R = 231 at
the blue end to R = 848 at the red end, so requests much above R ≈ 230 will
empty a bin at the blue end and raise:

.. code-block:: text

   ValueError: Bin 162 ([0.5994, 0.6009] um) contains no noise samples;
   requested R exceeds the native noise sampling

The error names the offending bin and its wavelength range rather than
silently producing a ``NaN``. You can check the ceiling yourself:

.. code-block:: python

   from lazuli_exo import tables

   tdir = tables.resolve_tables_dir(None)
   wl, err, meta = tables.load_noise(tdir, "WASP-39 b")
   native_R = wl[:-1] / np.diff(wl)
   native_R.min()      # ~231 — the binding constraint

What the tables assume
----------------------

The assumptions are recorded in ``meta["noise"]`` on every observation:

``instrument``
    The instrument configuration used, e.g. ``lazuli_cbe``.

``norm_band`` / ``J_mag``
    The band the source was normalized in, and the host magnitude used.

``default_nmd`` / ``nmd_used``
    The MACC readout configuration — the default, and what was actually used
    after tuning to remove point-source saturation.

``t14_s`` / ``t_obs_s``
    Transit duration and the native integration time, both in seconds. These
    also set the light-curve window and cadence.

``assumption``
    A plain-text statement of the depth-uncertainty model: *equal
    out-of-transit baseline; SNR ~ sqrt(t); readout tuned per-target to remove
    point-source saturation before reading SNR*.

``fully_desaturated`` / ``nsaturated_detpx``
    Whether the detector ramp could be fully de-saturated, and how many pixels
    remained saturated if not.

Equal out-of-transit baseline
-----------------------------

The depth uncertainty combines an in-transit and an out-of-transit
measurement,

.. math::

   \sigma_\mathrm{depth} = \sqrt{ \mathrm{SNR}_\mathrm{in}^{-2} +
   \mathrm{SNR}_\mathrm{out}^{-2} }

with the two exposures assumed **equal**. This is neither optimized nor
constrained by a telescope-time budget. It is also the assumption that lets
the light-curve module derive its per-point scatter from the band depth
precision — see :doc:`lightcurves`.

Bright hosts: unvalidated uncertainties
---------------------------------------

.. danger::

   For **HAT-P-11 b**, **HD 189733 b**, and **HD 209458 b**, slicersim's
   detector ramp could not be fully de-saturated even at the minimal MACC
   configuration (2 groups × 1 frame). Saturation can bias extraction weights,
   nonlinearity, and wavelength dependence in *either* direction, so the
   uncertainties for these three targets are **unvalidated**. Treat their
   forecasts as indicative only until an unsaturated readout strategy or an
   injection/recovery validation exists.

Rather than silently ignoring this, ``observe()`` raises a ``UserWarning``:

.. code-block:: python

   obs = lazuli_exo.observe("HD 189733 b")
   # UserWarning: HD 189733 b: noise table was generated with residual
   # detector saturation (7452 px); its uncertainties are unvalidated —
   # do not use for quantitative forecasts

To find the affected targets programmatically, check the flag rather than
hard-coding the three names:

.. code-block:: python

   flagged = [p for p in lazuli_exo.list_planets()
              if tables.load_noise(tdir, p)[2].get("fully_desaturated") is False]
