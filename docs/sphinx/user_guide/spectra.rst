Spectra and scenarios
=====================

Scenarios
---------

Each of the 14 showcase targets ships five precomputed scenarios. Four are
POSEIDON atmospheres; the fifth is an analytic featureless baseline.

.. list-table::
   :header-rows: 1
   :widths: 20 22 18 40

   * - Scenario
     - Cloud model
     - Metallicity
     - What it represents
   * - ``clear``
     - cloud-free
     - solar (``log_Met = 0``)
     - Full-amplitude molecular features — the optimistic case.
   * - ``cloudy``
     - ``MacMad17`` deck at :math:`10^{-3}` bar
     - solar
     - An opaque grey deck truncating the features from below.
   * - ``hazy``
     - ``MacMad17`` ``deck_haze``
     - solar
     - A scattering haze steepening the blue continuum.
   * - ``high_metallicity``
     - cloud-free
     - 100× solar (``log_Met = 2``)
     - Higher mean molecular weight, so a smaller scale height.
   * - ``flat``
     - —
     - —
     - Analytic featureless :math:`(R_p/R_s)^2` line.

All four physical scenarios use C/O = 0.55 and a single isothermal
:math:`T = T_\mathrm{eq}` layer. The parameterized species are H2O, CO, CO2,
CH4, NH3, Na, and K.

The settings are frozen into every table's ``meta``, so you never have to
trust a table in prose:

.. code-block:: python

   obs = lazuli_exo.observe("WASP-39 b", scenario="hazy")
   obs.meta["model"]
   # {'cloud_model': 'MacMad17', 'cloud_type': 'deck_haze',
   #  'cloud_params': [2.0, -6.0, 2.0], 'C_to_O': 0.55, 'log_Met': 0.0,
   #  'T_eq_K': 1166.0, 'T_iso_K': 1166.0, 'T_clamped': False, ...}

The ``flat`` baseline
---------------------

``flat`` is the reference the template statistic is measured *against*: a
constant :math:`(R_p/R_s)^2` line with zero spread. Running
:meth:`~lazuli_exo.SimulatedSpectrum.detect` on it returns exactly zero, which
is a useful invariant to assert in tests.

Its value is also surfaced directly on every observation as
``obs.flat_depth``, so you can compare a scenario's continuum level against
the featureless baseline without loading a second table.

The model grid
--------------

Model spectra are computed on a constant-resolving-power grid at
:math:`R = 10\,000` over 0.35–1.85 µm — deliberate margin around the
0.40–1.70 µm IFS bandpass, so that binning never runs off the end of the
model. The grid is built to match ``POSEIDON.core.wl_grid_constant_R``
byte for byte.

Binning
-------

:func:`~lazuli_exo.observe` bins the model and the noise onto a common
constant-R grid rather than interpolating one onto the other. The grid is
geometric — every bin has the same fractional width :math:`1 + 1/R`:

.. code-block:: python

   from lazuli_exo import binning

   edges = binning.constant_R_edges(0.40, 1.70, 100)
   centers = binning.bin_centers(edges)     # geometric mean of the edges

Model samples are **averaged** within a bin
(:func:`~lazuli_exo.binning.bin_model`); noise channels are combined in
**inverse variance** (:func:`~lazuli_exo.binning.bin_sigma`). That difference
is the whole reason coarser ``R`` buys precision.

.. note::

   Only *complete* bins are kept, so coverage can stop up to one bin short of
   the requested ``wl_max``. If you are matching grids between two calls, take
   the edges from one call rather than recomputing them.

The resolution ceiling is set by the noise tables, not the model — see
:doc:`noise`.
