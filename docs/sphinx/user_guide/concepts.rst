Concepts and known approximations
=================================

``lazuli_exo`` is a **comparison tool**, not a forecasting tool. It exists to
let you ask "how do these atmospheres, targets, or transit counts compare
under one consistent set of assumptions?", and it is explicit about which
assumptions those are.

.. warning::

   Every scenario is a toy H2/He atmosphere with idealized noise. No number
   produced by this package is a calibrated detection significance or an
   observing-time forecast. Treat outputs as *relative* and *qualitative*.

The measurement in one page
---------------------------

If transmission spectroscopy is new to you, this is the whole idea.

When a planet crosses its star, the star dims by the fraction of its disk the
planet covers, the transit depth :math:`\delta = (R_p/R_s)^2`. At a
wavelength where the planet's atmosphere absorbs, the planet looks slightly
larger, so the transit is slightly deeper. The size of that extra depth is set
by the atmospheric scale height :math:`H = k T_\mathrm{eq} / (\mu m_H g)`:
one scale height of extra radius adds about :math:`2 R_p H / R_s^2` to the
depth. For a hot Jupiter that is a few hundred parts per million (ppm); for a
temperate sub-Neptune, tens of ppm. The transmission spectrum is the transit
depth measured as a function of wavelength, and its bumps are the molecular
bands (water near 1.4 µm, sodium and potassium in the optical) and the
continuum slope of a haze.

``lazuli_exo`` computes two things for such a measurement with Lazuli. The
model spectrum, binned to a resolving power ``R`` (the number of wavelength
bins per octave divided by :math:`\ln 2`; ``R = 100`` gives about 145 bins
over the band). And the 1-sigma uncertainty on the depth in each bin for
``n_transits`` co-added transits, from the host star's brightness, the
transit duration and the instrument's photon and read noise. From those two
arrays it derives a single number, the template S/N, which says how far the
model spectrum sits from a flat line in units of the error bars. It is a
comparison statistic between planets, scenarios and observing strategies;
:doc:`detect` explains why it is not a detection significance.

A few terms used throughout:

``depth``, ``depth_err``
    The binned model transit depth (a fraction; multiply by :math:`10^6`
    for ppm) and its 1-sigma uncertainty per bin.
``R``
    Resolving power of the binned grid, :math:`\lambda / \Delta\lambda`.
    Coarser bins (smaller ``R``) have smaller error bars and resolve less.
``n_transits``
    Transits co-added; error bars shrink as :math:`1/\sqrt{N}` with no floor.
``feature_amplitude_ppm``
    Peak-to-trough of the binned model over 0.40 to 1.70 µm, in ppm.
``template_snr_diagonal``
    The template S/N described above.
``scenario``
    A preset of the atmosphere parameters: ``clear``, ``cloudy``, ``hazy``,
    ``high_metallicity`` or ``flat`` (:doc:`spectra`).

The two-stage architecture
--------------------------

The heavy physics runs once, offline, in environments that never have to
coexist:

.. code-block:: text

   inputs (POSEIDON opacities + fastchem grid, PHOENIX stellar grid)
         |
         v  poseidon conda env
   atmo/zeta_grid.npz     star/phoenix_grid.npz     star/<planet>.npz
   (opacity shape on a (T, [M/H], C/O) grid)             |
         |                                               v  py313 env + slicersim
         |                                         noise/<planet>.npz
         |                                         (single-transit depth_err(lambda))
         |                                               v  py313 env + exotic-ld
         |                                         ld/<planet>.npz
         v                                         (quadratic limb darkening)
   runtime (src/lazuli_exo/) = numpy + these tables, nothing heavier

   precompute/tables/models/<planet>__<scenario>.npz   POSEIDON validation data

POSEIDON and slicersim are never installed together. The runtime imports
neither. This is what lets the package be deployed behind a web app without
shipping tens of GB of opacity inputs.

Known approximations
--------------------

These are the limits that matter. Each is real, each is knowable, and each is
recorded somewhere you can check.

Isothermal equilibrium chemistry
    Every spectrum uses a single :math:`T = T_\mathrm{eq}` isotherm, not a
    real pressure-temperature profile, with equilibrium chemistry throughout
    (the scenario presets fix C/O = 0.55; ``simulate()`` lets you vary it).

Surrogate atmosphere
    The spectrum is an interpolation of a POSEIDON-derived opacity grid
    plus analytic scale-height, column, deck and haze terms, not a live
    radiative-transfer run. Inside its validated box it matches POSEIDON to
    well under the single-transit noise; see :doc:`parametric` for the box
    and the gates.

Cold-planet temperature floor
    The fastchem equilibrium-chemistry grid has a hard floor at 300 K. K2-18 b
    (:math:`T_\mathrm{eq} \approx 284` K), LHS 1140 b (≈ 226 K), and
    TRAPPIST-1 e (≈ 250 K) use 300 K chemistry, while the scale height uses
    the true :math:`T_\mathrm{eq}`. Recorded on every observation as
    ``meta["atmosphere"]["T_clamped"]`` (and ``T_chem_K``).

:math:`\sqrt{N}` S/N scaling
    ``depth_err`` scales as :math:`1/\sqrt{N_\mathrm{transits}}` from a single
    precomputed single-transit uncertainty. There is **no per-transit
    systematics term and no red-noise floor**, so predicted precision improves
    without limit as transits are added. This is the single largest unrealism
    in the package. See :doc:`noise`.

Equal out-of-transit baseline
    The noise model assumes an out-of-transit baseline exposure equal to the
    in-transit one, i.e. :math:`\sigma_\mathrm{depth} = \sqrt{1/\mathrm{SNR}_\mathrm{in}^2 + 1/\mathrm{SNR}_\mathrm{out}^2}`
    with :math:`\mathrm{SNR}_\mathrm{in} = \mathrm{SNR}_\mathrm{out}`, not an
    optimized or telescope-time-constrained baseline.

Template S/N, not detection significance
    :func:`~lazuli_exo.detect.detect_features` returns a matched-template
    amplitude S/N against the best-fit flat line under a diagonal covariance.
    There is no defined null/alternative likelihood ratio, no nuisance model,
    no covariance, and no false-positive calibration. See :doc:`detect`.

Residual detector saturation on bright hosts
    For HD 189733 b, slicersim's detector ramp cannot be fully de-saturated
    even at the minimal MACC configuration (2 groups × 1 frame). The residual
    pixel count is recorded as ``meta["noise"]["nsaturated_detpx"]`` and
    ``observe()`` raises a ``UserWarning``. **Its uncertainties are
    unvalidated**: saturation can bias extraction weights, nonlinearity, and
    wavelength dependence in either direction. Check the flag rather than
    the name; the list changes when the tables are regenerated.

Light-curve geometry
    Quadratically limb-darkened transits with geometry from a frozen NASA
    Exoplanet Archive registry. Planets outside the registry, or without a
    limb-darkening table, fall back to a :math:`b = 0` trapezoid;
    ``meta["shape"]`` records which. See :doc:`lightcurves`.

The caveats live in the data
----------------------------

Two of the approximations above are flagged *per table*, so code can branch on
them rather than relying on someone having read this page:

.. code-block:: python

   obs = lazuli_exo.observe("K2-18 b", scenario="clear")
   obs.meta["atmosphere"]["T_clamped"]       # True: chemistry grid floor hit

   obs = lazuli_exo.observe("HD 189733 b")   # also raises a UserWarning
   obs.meta["noise"]["fully_desaturated"]    # False
   obs.meta["noise"]["nsaturated_detpx"]     # residual saturated pixels

The upgrade paths
-----------------

The two most consequential limitations have concrete, non-speculative fixes
that fit the existing table schema without API changes:

- **Molecule detections.** Replace the band-restricted template norms with a
  likelihood ratio against per-molecule *leave-one-out* model tables. The
  schema already supports additional scenario names per planet.
- **Realistic transit counts.** Add a per-transit systematics or red-noise
  term to the noise tables, which would break the pure :math:`\sqrt{N}`
  scaling and stop transit counts from being optimistic without bound.
