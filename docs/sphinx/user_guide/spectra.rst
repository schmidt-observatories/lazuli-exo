Spectra and scenarios
=====================

Scenarios
---------

Every showcase target offers the same five scenarios. They are presets of
the parametric model (:mod:`lazuli_exo.presets`), evaluated at the registry
values of the planet; ``observe()`` and ``simulate()`` run one model.

.. list-table::
   :header-rows: 1
   :widths: 20 36 44

   * - Scenario
     - Parameters
     - What it represents
   * - ``clear``
     - solar metallicity, no deck, no haze
     - Full-amplitude molecular features, the optimistic case.
   * - ``cloudy``
     - gray deck at 1 mbar (``log_P_cloud_bar = -3``)
     - An opaque deck truncating the features from below.
   * - ``hazy``
     - haze 100× H2 Rayleigh at 0.35 µm, slope −6 (``log_a_haze = 2``, ``haze_slope = -6``)
     - A scattering haze steepening the blue continuum.
   * - ``high_metallicity``
     - 100× solar (``log_met = 2``)
     - Higher mean molecular weight, so a smaller scale height.
   * - ``flat``
     - gray deck at 10 nbar (``log_P_cloud_bar = -8``)
     - A featureless :math:`(R_p/R_s)^2` line, to within a few ppm.

All scenarios use C/O = 0.55, ``mu=None`` (the equilibrium-chemistry mean
molecular weight) and a single isothermal :math:`T = T_\mathrm{eq}` layer.
The opacity grid carries H2O, CO, CO2, CH4, NH3, Na and K.

The parameters are recorded on every observation, so you never have to
trust this table:

.. code-block:: python

   obs = lazuli_exo.observe("WASP-39 b", scenario="hazy")
   obs.meta["params"]
   # {'name': 'WASP-39 b', 'scenario': 'hazy', 'T_eq_K': 1166.0, ...,
   #  'log_met': 0.0, 'c_to_o': 0.55, 'mu': None, 'log_P_cloud_bar': None,
   #  'log_a_haze': 2.0, 'haze_slope': -6.0, ...}
   obs.meta["atmosphere"]
   # {'g_ms2': ..., 'H_m': ..., 'H_over_Rp': ..., 'mu_used': ...,
   #  'mu_source': 'chemistry', 'T_chem_K': 1166.0, 'T_clamped': False,
   #  'feature_amplitude_ppm': ..., ...}

The physics behind the presets is in :doc:`parametric`.

The ``flat`` baseline
---------------------

``flat`` is the reference the template statistic is measured *against*: a
deck above every line core, leaving a constant :math:`(R_p/R_s)^2` line with
under a ppm of ripple. Running :meth:`~lazuli_exo.SimulatedSpectrum.detect`
on it returns a template S/N below 0.01, an invariant the test suite asserts.

The geometric :math:`(R_p/R_s)^2` from the registry radii is also on every
observation as ``obs.flat_depth``, so you can compare a scenario's continuum
level against the featureless baseline without a second call.

The model grid
--------------

The model is evaluated on the opacity grid's constant-resolving-power
wavelengths, :math:`R = 10\,000` over 0.35–1.85 µm, with margin around the
0.40–1.70 µm IFS bandpass so the instrument response never runs off the end
of the model. The grid matches ``POSEIDON.core.wl_grid_constant_R``.

Binning
-------

:func:`~lazuli_exo.binning.bin_observation` turns the native model and the
per-channel noise into the binned spectrum in four steps:

1. Constant-R bin edges (:func:`~lazuli_exo.binning.constant_R_edges`, the
   same fractional width :math:`1 + 1/R` everywhere) are snapped to the
   nearest detector channel edges, so every bin is a whole number of pixels.
2. The model is seen through the spectrograph
   (:func:`~lazuli_exo.binning.instrument_response`): the stellar photon
   flux times :math:`1 - d(\lambda)` and the flux alone are convolved with
   the Gaussian line-spread function (sigma 1.15 px) and integrated per
   pixel, so the per-channel depth is photon weighted within each pixel.
3. The per-channel depths are combined with the channel
   :math:`1/\sigma^2` weights (:func:`~lazuli_exo.binning.bin_weighted`), the
   same combination whose uncertainty step 4 returns.
4. The channel uncertainties are combined in inverse variance
   (:func:`~lazuli_exo.binning.bin_sigma`), which is why coarser ``R`` buys
   precision.

``obs.meta["instrument_response"]`` records the LSF width, the instrument
resolving-power range over the band and which photon weighting was used
(the R = 10 000 star table for ``observe()``, the R = 1000 PHOENIX SED for
``simulate()``).

.. note::

   Only *complete* bins are kept, so coverage can stop up to one bin short of
   the band edge. If you are matching grids between two calls, take
   ``obs.wavelength`` from one call rather than recomputing the edges.

The two resolution limits are set by the instrument and the noise tables,
not by the model; see :doc:`noise`.
