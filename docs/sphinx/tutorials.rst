Tutorials
=========

These tutorials are runnable Jupyter notebooks from the ``notebooks/``
directory of the repository, rendered here with their saved outputs. Each one
is self-contained. Download it and run it interactively to experiment.
Notebooks 6–9 need ``slicersim`` and ``sncosmo`` installed (live noise).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - What it covers
   * - :doc:`notebooks/01_getting_started`
     - The ``observe → detect → plot`` loop, the ``SimulatedSpectrum`` fields,
       reproducible noise realizations, and reading the provenance in ``meta``.
   * - :doc:`notebooks/02_scenarios`
     - The five scenario presets, overlaying them with ``.plot(ax=ax)``, why
       ``hazy`` can out-score ``clear``, and the cold-planet temperature clamp.
   * - :doc:`notebooks/03_transits_and_binning`
     - ``n_transits`` and ``R``: the exact :math:`1/\sqrt{N}` scaling,
       ``.plot_noise()``, the instrument response and the two resolution
       limits.
   * - :doc:`notebooks/04_light_curves`
     - ``.light_curves().plot()`` for white-light and banded transits,
       validation of ``transit_flux`` against the analytic uniform-disk limit,
       limb darkening, and the trapezoid fallback.
   * - :doc:`notebooks/05_tables_and_targets`
     - The table layout, index and sha256 manifest, the loaders, the frozen NEA
       registry, and where the caveats are recorded in the data itself.
   * - :doc:`notebooks/06_parametric_model`
     - The :math:`\zeta` grid, the mean molecular weight from chemistry, the
       reconstruction of the POSEIDON tables, the validation table, and a
       custom sub-Neptune end to end.
   * - :doc:`notebooks/07_custom_planet`
     - A planet that is not in the registry (GJ 9827 d) from a dozen numbers;
       μ and deck variants; the same run through the ``forms``/``report`` web
       contract.
   * - :doc:`notebooks/08_atmosphere_knobs`
     - Sweeps of μ, metallicity, deck pressure, haze and :math:`T_\mathrm{eq}`
       on a hot Jupiter and a sub-Neptune; feature amplitude, template S/N and
       transits needed.
   * - :doc:`notebooks/09_stars_and_noise`
     - The PHOENIX SED grid, Vega→AB offsets, the error bar versus magnitude
       and band, the bright-host saturation floor, CBE vs EOL configurations.

.. note::

   The repository also contains ``notebooks/verification.ipynb``, a developer
   verification artifact that asserts physical sanity checks against the
   committed tables. It is deliberately not part of this tutorial series.
   Each ``notebooks/build_0N.py`` regenerates the corresponding notebook.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/01_getting_started
   notebooks/02_scenarios
   notebooks/03_transits_and_binning
   notebooks/04_light_curves
   notebooks/05_tables_and_targets
   notebooks/06_parametric_model
   notebooks/07_custom_planet
   notebooks/08_atmosphere_knobs
   notebooks/09_stars_and_noise
