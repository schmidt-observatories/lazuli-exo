Concepts and known approximations
=================================

``lazuli_exo`` is a **comparison tool**, not a forecasting tool. It exists to
let you ask "how do these atmospheres, targets, or transit counts compare
under one consistent set of assumptions?" — and it is deliberately explicit
about which assumptions those are.

.. warning::

   Every scenario is a toy H2/He atmosphere with idealized noise. No number
   produced by this package is a calibrated detection significance or an
   observing-time forecast. Treat outputs as *relative* and *qualitative*.

The two-stage architecture
--------------------------

The heavy physics runs once, offline, in environments that never have to
coexist:

.. code-block:: text

   inputs (POSEIDON opacities + fastchem grid)
         |
         v  poseidon conda env
   models/<planet>__<scenario>.npz      star/<planet>.npz
         |
         v  py313 env + slicersim
   noise/<planet>.npz    (per-planet single-transit depth_err(lambda))
         |
         v  py313 env + exotic-ld
   ld/<planet>.npz       (quadratic limb darkening u1(lambda), u2(lambda))

   runtime (src/lazuli_exo/) = numpy + these tables, nothing heavier

POSEIDON and slicersim are never installed together. The runtime imports
neither. This is what lets the package be deployed behind a web app without
shipping tens of GB of opacity inputs.

Known approximations
--------------------

These are the limits that matter. Each is real, each is knowable, and each is
recorded somewhere you can check.

Isothermal equilibrium chemistry
    Every scenario uses a single :math:`T = T_\mathrm{eq}` isotherm, not a
    real pressure–temperature profile. Equilibrium chemistry throughout, with
    C/O = 0.55 fixed.

Cold-planet temperature floor
    The fastchem equilibrium-chemistry grid has a hard floor at 300 K. K2-18 b
    (:math:`T_\mathrm{eq} \approx 284` K), LHS 1140 b (≈ 226 K), and
    TRAPPIST-1 e (≈ 250 K) are clamped to 300 K for both the isotherm and the
    chemistry lookup. Recorded per table as ``meta["model"]["T_clamped"]``.

``hazy`` is ``deck_haze``, not haze alone
    POSEIDON's ``cloud_type="haze"`` alone is a silent no-op for a 1D model —
    it zeroes the cloudy-sector fraction, so the haze opacity is computed but
    never mixed into the spectrum (confirmed by an empirical A/B check giving
    a spectrum bit-identical to ``clear``). The ``hazy`` scenario therefore
    uses ``cloud_model="MacMad17"``, ``cloud_type="deck_haze"``, with the
    deck's own opaque cloud pinned at the pressure grid floor (100 bar) so it
    never shadows the atmosphere — only the haze term contributes opacity.

:math:`\sqrt{N}` S/N scaling
    ``depth_err`` scales as :math:`1/\sqrt{N_\mathrm{transits}}` from a single
    precomputed single-transit uncertainty. There is **no per-transit
    systematics term and no red-noise floor**, so predicted precision improves
    without limit as transits are added. This is the single largest unrealism
    in the package. See :doc:`noise`.

Equal out-of-transit baseline
    The noise model assumes an out-of-transit baseline exposure equal to the
    in-transit one, i.e. :math:`\sigma_\mathrm{depth} = \sqrt{1/\mathrm{SNR}_\mathrm{in}^2 + 1/\mathrm{SNR}_\mathrm{out}^2}`
    with :math:`\mathrm{SNR}_\mathrm{in} = \mathrm{SNR}_\mathrm{out}` — not an
    optimized or telescope-time-constrained baseline.

Template S/N, not detection significance
    :func:`~lazuli_exo.detect.detect_features` returns a matched-template
    amplitude S/N against the best-fit flat line under a diagonal covariance.
    There is no defined null/alternative likelihood ratio, no nuisance model,
    no covariance, and no false-positive calibration. See :doc:`detect`.

Residual detector saturation on bright hosts
    For HAT-P-11 b, HD 189733 b, and HD 209458 b, slicersim's detector ramp
    cannot be fully de-saturated even at the minimal MACC configuration
    (2 groups × 1 frame). The residual pixel count is recorded as
    ``meta["noise"]["nsaturated_detpx"]`` and ``observe()`` raises a
    ``UserWarning``. **Their uncertainties are unvalidated** — saturation can
    bias extraction weights, nonlinearity, and wavelength dependence in either
    direction.

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
   obs.meta["model"]["T_clamped"]            # True — chemistry grid floor hit

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
