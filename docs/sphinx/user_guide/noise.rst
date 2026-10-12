The noise model
===============

Every error bar ``lazuli_exo`` reports traces back to one array per planet:
``depth_err_1transit(λ)``, the single-transit transit-depth uncertainty
computed offline with slicersim for the Lazuli IFS.

What ``observe()`` does with it
-------------------------------

Two operations, both simple and both worth knowing exactly:

1. **Bin** the native noise channels onto the requested constant-R grid
   (edges snapped to whole detector pixels, see :doc:`spectra`), in
   inverse variance:

   .. math::

      \sigma_\mathrm{bin} = \left( \sum_{i \in \mathrm{bin}}
      \sigma_i^{-2} \right)^{-1/2}

2. **Scale** by the number of transits:

   .. math::

      \sigma_\mathrm{depth} = \sigma_\mathrm{bin} /
      \sqrt{N_\mathrm{transits}}

That is the entire noise model at runtime. The binned model ``depth`` is
untouched by ``n_transits``. ``obs.plot_noise()`` draws the result against
wavelength.

.. warning::

   The :math:`1/\sqrt{N}` scaling has **no floor**. There is no per-transit
   systematics term and no red noise, so the predicted precision improves
   without limit as transits accumulate. Any "number of transits needed"
   computed from these outputs is a lower bound, optimistic by an amount this
   model cannot tell you.

The two resolution limits
-------------------------

Both are set by the part of the band where the dispersion makes the pixels
coarsest in resolving power, which for the shipped tables is near 1.0 µm,
not by the median.

**The instrument resolving power** is 2.3 pixels per resolution element
(:data:`~lazuli_exo.binning.RESOLUTION_PX`): R ≈ 369 at 0.40 µm, a minimum
of R ≈ 101 near 1.02 µm, and R ≈ 165 at 1.69 µm. Above the minimum,
``observe()`` and ``simulate()`` still run but warn: bins narrower than a
resolution element are correlated by the line-spread function, so the error
bars are no longer independent.

.. code-block:: text

   UserWarning: requested R=150 exceeds the instrument resolving power
   (R_instr = 101-369 over the binned range); bins narrower than a
   resolution element are correlated by the LSF

**The native channel sampling** is the hard ceiling: one pixel per bin,
R ≈ 231 at its minimum (848 at 0.40 µm). Above it the request raises:

.. code-block:: text

   ValueError: requested R exceeds the native noise sampling (bins narrower
   than one channel; R_px >= 231)

Both numbers are on every observation and in the tables:

.. code-block:: python

   obs.meta["instrument_response"]["R_instr_min"]     # ~101
   from lazuli_exo import tables, binning
   wl, err, meta = tables.load_noise(tables.resolve_tables_dir(None), "WASP-39 b")
   binning.R_instr(wl).min()                            # ~101, the warning threshold
   (wl[:-1] / np.diff(wl)).min()                        # ~231, the hard ceiling

What the tables assume
----------------------

The assumptions are recorded in ``meta["noise"]`` on every observation:

``instrument``
    The instrument configuration used, e.g. ``lazuli_cbe``.

``norm_band`` / ``J_mag`` / ``mag_ab`` / ``vega_ab_offset``
    The band the source was normalized in, the archive (Vega) magnitude
    and its AB equivalent, which is what slicersim actually normalizes to.

``default_nmd`` / ``nmd_used``
    The MACC readout configuration: the default, and what was actually used
    after tuning to remove point-source saturation.

``t14_s`` / ``t_obs_s`` / ``tframe_s`` / ``n_reset_frames``
    Transit duration and the native integration time (including the reset
    frame overhead), in seconds. These also set the light-curve window and
    cadence.

``package_version`` / ``git_commit`` / ``slicersim_version`` / ``*_sha256``
    Provenance: the code and the input files the table was generated from.

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
precision. See :doc:`lightcurves`.

Bright hosts: unvalidated uncertainties
---------------------------------------

.. danger::

   For **HD 189733 b**, slicersim's detector ramp could not be fully
   de-saturated even at the minimal MACC configuration (2 groups × 1 frame;
   324 pixels remain saturated at the last read). Saturation can bias
   extraction weights, nonlinearity, and wavelength dependence in *either*
   direction, so its uncertainties are **unvalidated**. Treat its forecasts
   as indicative only until an unsaturated readout strategy or an
   injection/recovery validation exists. HAT-P-11 b and HD 209458 b were on
   this list before the 2026-10-11 table regeneration and now desaturate.
   HD 189733 b stays in the tables on purpose: it is the brightest host in
   the sample and the warning travels with every result.

Rather than silently ignoring this, ``observe()`` raises a ``UserWarning``:

.. code-block:: python

   obs = lazuli_exo.observe("HD 189733 b")
   # UserWarning: HD 189733 b: noise table was generated with residual
   # detector saturation (324 px); its uncertainties are unvalidated —
   # do not use for quantitative forecasts

To find the affected targets programmatically, check the flag rather than
hard-coding the name:

.. code-block:: python

   flagged = [p for p in lazuli_exo.list_planets()
              if tables.load_noise(tdir, p)[2].get("fully_desaturated") is False]
