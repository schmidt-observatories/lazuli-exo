The parametric model
====================

:func:`lazuli_exo.simulate` builds a transmission spectrum for *any*
transiting planet. The showcase tables froze one POSEIDON run per planet and
scenario; the parametric model runs POSEIDON once, offline, on a grid that
every planet shares, and applies everything planet-specific analytically at
request time (about 20 ms, numpy only). The noise then comes from a live
slicersim run on a PHOENIX stellar SED.

Inputs
------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Meaning
   * - ``T_eq_K``, ``R_p_RJ``, ``M_p_MJ``, ``R_s_RSun``
     - Planet temperature, **band-median transit radius**, mass; stellar
       radius. Gravity is evaluated at :math:`R_p`.
   * - ``log_met`` (−1…3), ``c_to_o`` (0.3…1.0)
     - Atmospheric metallicity (log, solar = 0) and C/O for the equilibrium
       chemistry.
   * - ``mu``
     - Mean molecular weight [amu]; ``None`` takes the equilibrium-chemistry
       value stored with the grid.
   * - ``log_P_cloud_bar``
     - Gray cloud-deck top pressure; ``None`` = clear.
   * - ``log_a_haze``, ``haze_slope``
     - POSEIDON MacMad17 haze: enhancement over H2 Rayleigh at 0.35 µm (log)
       and power-law slope :math:`\gamma`; ``log_a_haze=None`` = no haze.
   * - ``teff_K``, ``logg``, ``feh``
     - Star for the PHOENIX SED (solar metallicity only, 2000–12000 K,
       log g 3.5–5.5; out-of-range T_eff, log g and any non-zero [Fe/H]
       clamp to the nearest node and are reported in ``info["clamped"]``).
   * - ``f_spot``, ``T_spot_K``, ``f_fac``, ``T_fac_K``
     - Unocculted spots and faculae (transit light source effect, Rackham et
       al. 2018 eq. 2): the model depth is multiplied by
       :math:`\epsilon(\lambda)` from PHOENIX fluxes at the three
       temperatures. Default off (``f_spot = f_fac = 0``);
       ``star.TLSE_PRESETS`` holds literature covering fractions.
   * - ``mag``, ``band`` (J/H/Ks/V), ``mag_system`` (``"vega"`` default, ``"ab"``)
     - Normalisation magnitude. slicersim normalises in AB; archive
       magnitudes are Vega and are converted through sncosmo.
   * - ``t14_s``
     - Transit duration; the noise assumes an equal out-of-transit baseline.

Physics
-------

For an isothermal, well-mixed atmosphere the photospheric pressure at each
wavelength, written as :math:`\zeta(\lambda) = -\ln[P_{\tau=1}(\lambda)/1\,\mathrm{bar}]`,
depends on the gas opacity — i.e. on (T, [M/H], C/O) — and only weakly on
the planet. It is tabulated for two fiducial planets (1 R\ :sub:`J` at
10 m s\ :sup:`-2` and 0.3 R\ :sub:`J` at 30 m s\ :sup:`-2`) at the model's
native R = 10,000 and interpolated trilinearly in (T, [M/H], C/O). Everything
else is analytic:

.. math::

   x_p &= \tfrac{1}{2}\ln\frac{R_p H}{R_{p,0} H_0} - \ln\frac{T_\mathrm{eq}}{T_\mathrm{chem}}
   \qquad\text{(slant-column variable)}\\
   \zeta_p &= \zeta_0 + s(\lambda)\,(x_p - x_0)
   \qquad s = \frac{\zeta_1 - \zeta_0}{x_1 - x_0} \approx \frac{1}{1+a}\ \text{for}\ \tau\propto P^{1+a}\\
   \zeta_g &= \mathrm{logaddexp}(\zeta_p,\ \zeta_\mathrm{haze}),\qquad
   P_\mathrm{haze}(\lambda) = \frac{k T_\mathrm{eq}}{\kappa_\mathrm{haze}\sqrt{2\pi R_p H}}\\
   \zeta &= \zeta_d + s\,\mathrm{Ein}\!\left(e^{(\zeta_g - \zeta_d)/s - \gamma_E}\right),
   \qquad \zeta_d = -\ln P_\mathrm{cloud}\\
   R(\lambda) &= \frac{R_p}{1 - (H/R_p)\,[\zeta - \tilde\zeta]},\qquad
   \mathrm{depth} = \left(\frac{R(\lambda)}{R_s}\right)^2

with :math:`H = kT_\mathrm{eq}/(\mu m_H g)`, :math:`g` at :math:`R_p`, and
:math:`\tilde\zeta` the band median (0.40–1.70 µm) so that :math:`R_p` is the
band-median transit radius. The last line is the exact isothermal
hydrostatic solution for :math:`g\propto r^{-2}`. The deck line is the exact
effective radius of an exponential optical-depth profile above an opaque
deck (POSEIDON integrates :math:`1-e^{-\tau}` over impact parameter, so gas
whose :math:`\tau=1` level lies just below the deck still absorbs); it tends
to :math:`\max(\zeta_g,\zeta_d)` in both limits. Ein is the entire
exponential integral.

Conventions worth knowing
-------------------------

- **Radius.** The showcase tables put the archive radius at the 10 bar
  reference level; the parametric model puts it at the transit photosphere,
  roughly ten scale heights higher. The validation therefore feeds the model
  each table's own band-median radius. For a user the archive radius *is* the
  transit radius, so no correction is needed.
- **Temperature floor.** The chemistry grid starts at 300 K. Colder planets
  use 300 K chemistry but the true :math:`T_\mathrm{eq}` for :math:`H`;
  ``info["T_clamped"]`` records it.
- **Deck discretisation.** POSEIDON applies a deck to layers strictly deeper
  than :math:`P_\mathrm{cloud}`, so on its 100-layer grid a nominal 1 mbar
  deck is realised at 1.38 mbar. The model keeps the user's value literal.
- **Flat preset.** ``presets.showcase(planet, "flat")`` uses a deck at 10 nbar.

Validation
----------

``precompute/validate_zeta_grid.py`` compares the model with the 14 × 5
POSEIDON tables in ``precompute/tables/`` (plus a set of off-registry
holdout runs) at R = 100, after removing a constant offset, against the
single-transit per-bin uncertainty. The gates are: worst bin within 1 sigma
(G2), the error spectrum's own template S/N below 3 (G5), and the template
S/N within 10 % of the table's (G3); ``flat`` is gated on its absolute
amplitude. Expected failures are waivers in the script, each naming the
gate and the GitHub issue that would remove it, and a waiver that no longer
fails is itself an error. CI runs the gate on every push. The notebook
:doc:`../notebooks/06_parametric_model` shows the grid and the comparison.

What it cannot do
-----------------

Non-isothermal profiles, disequilibrium chemistry, patchy clouds, red
noise, and light curves for custom stars (no limb-darkening tables).
Stellar contamination is a user-supplied covering fraction, not a prediction
of the host's activity. Changing ``mu`` by hand does not change the
opacities, which stay at the chemistry implied by ``log_met`` and
``c_to_o``.
