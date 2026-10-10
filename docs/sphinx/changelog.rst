Changelog
=========

The project is pre-release; there has been no tagged version yet, so the
entries below group the development history by theme rather than by release.

Unreleased (0.2.0)
------------------

- Absorbed the standalone ``lazuli-transit`` package as
  :mod:`lazuli_exo.transit`: ``FluxModel``, ``TransitModel`` (jaxoplanet),
  ``MoonTransitModel`` (pandoramoon), the NASA Exoplanet Archive cache
  (``download_exoplanet_archive`` / ``load_exoplanet_archive``) and the
  matplotlib helpers. New extras ``transit``, ``archive``, ``pandora``; the
  subpackage itself still imports with numpy only. The archive cache moved
  from ``~/.lazuli_transit/`` to ``~/.lazuli_exo/``. ``wcc_etc`` depends on
  ``lazuli-exo[transit]`` instead of vendoring a copy.

Unreleased (0.1.0 development)
------------------------------

2026-10 (parametric model)
--------------------------

- ``lazuli_exo.simulate(params)``: any transiting planet from a shared
  POSEIDON-derived :math:`\zeta` grid (16 T × 9 [M/H] × 4 C/O × 2 fiducials,
  native R = 10,000, float16) plus analytic scale height, slant column, haze
  and an exact Ein cloud-deck operator; exact hydrostatic radius mapping.
  Validated against the 14 × 5 showcase tables (one documented waiver).
- PHOENIX stellar SED mini-grid (2000–8000 K) and ``star.phoenix_sed``.
- ``slicer.depth_error``: live slicersim noise with the Vega→AB conversion
  through sncosmo, the readout desaturation loop, provenance in ``info``.
  The committed noise tables were regenerated with this conversion (1.4–1.5×
  more conservative for unsaturated hosts).
- ``presets.showcase``, ``orbit.transit_duration_s``, ``forms`` and
  ``report`` (the web contract shared with the Lazuli simulators site).
- Flask applet: Custom-planet mode over ``/api/simulate``.
- Tables moved into the package (``src/lazuli_exo/data/tables``) as package
  data; wheels and VCS pins ship them. Schema version 2.
- Package renamed ``lazuli-exo`` (import ``lazuli_exo``); repository at
  https://github.com/gummiks/lazuli-exo.
- Tutorials 06–09 and this documentation update.

Documentation
~~~~~~~~~~~~~

- Sphinx documentation site under ``docs/sphinx/`` — user guide, quick start,
  auto-generated API reference, and the tutorial notebooks rendered via
  nbsphinx.
- Five tutorial notebooks in ``notebooks/`` covering the runtime end to end.
- Verification notebook (``notebooks/verification.ipynb``) asserting physical
  sanity checks against the committed tables.

Light curves
~~~~~~~~~~~~

- Quadratically limb-darkened transits with geometry from the frozen NASA
  Exoplanet Archive registry — impact parameter from :math:`a/R_\star` and
  :math:`\cos i` with an eccentricity factor, and :math:`z(t)` along a local
  circular arc at the Keplerian conjunction sky speed.
- Limb-darkening tables per planet from exotic-ld (PHOENIX grid, 48
  wavelength chunks); trapezoid fallback recorded in ``meta["shape"]``.
- White-light and wavelength-banded light curves, with per-point noise derived
  from the band depth precision.

Web demo
~~~~~~~~

- Local single-page Flask + Plotly UI over the runtime (``webapp/``), with
  system-parameters and diagnostics panels.
- ``observe()`` validation errors surface as a JSON 400 rather than an HTML
  500.

Runtime
~~~~~~~

- ``observe()`` assembling binned spectra with :math:`1/\sqrt{N}` noise
  scaling.
- Constant-R binning with inverse-variance noise combination.
- Spectral-template S/N against a flat baseline, with per-band masks.
- Spectrum plotting with simulated error bars.
- Frozen showcase-target registry (14 planets), generated from NEA
  ``PSCompPars``.
- Package scaffold and versioned lookup-table I/O; table loads validate
  against the index unconditionally.

Precompute
~~~~~~~~~~

- POSEIDON model tables and PHOENIX stellar spectra for all targets, plus
  analytic ``flat`` baselines.
- slicersim single-transit noise tables, with saturation-free per-target
  readout tuning and the residual saturation recorded in ``meta`` for the
  three bright hosts where it could not be eliminated.
- Scenario definitions pinned in one place, including the ``deck_haze``
  workaround for POSEIDON's silent ``cloud_type="haze"`` no-op.

Renamed
~~~~~~~

- The package carried two earlier working names before settling on ``lazuli-exo`` on 2026-10-05.
