Changelog
=========

The project is pre-release; there has been no tagged version yet, so the
entries below group the development history by theme rather than by release.

Unreleased
----------

Physics review of 2026-10-10 (tracked in GitHub issue #31, milestone
"Quantitative atmospheres"):

- Stellar contamination (transit light source effect, Rackham et al. 2018)
  as ``f_spot``/``T_spot_K``/``f_fac``/``T_fac_K`` keywords of ``simulate``
  and ``observe``; ``star.contamination`` and ``star.TLSE_PRESETS``. Default
  off (#42).
- Light curves use the exact Keplerian sky-projected separation
  (``orbit.projected_separation``) and stretch the orbit clock so the
  first-to-fourth-contact duration equals the registry :math:`t_{14}`;
  removes a 1σ depth bias for WASP-12 b (#43).
- PHOENIX grid rebuilt from the solar-metallicity folder only (the non-solar
  pysynphot folders are R ≈ 740 and unreliable), log g to 5.5 and T_eff to
  12000 K; the blackbody fallback is gone and every out-of-range input
  clamps to the nearest node with ``info["clamped"]``. The live (``simulate``)
  and table (``observe``) noise paths now agree to ≈ 3 % instead of
  0.5–1.6× (#44, part of #34).
- ``.readthedocs.yaml`` so the docs build on Read the Docs.
- Live noise (``slicer.depth_error``, hence ``simulate``) pins the narrow
  spectrograph field by default. slicersim >= 1.3.3 no longer rescales a new
  ``LazuliTarget`` to a field, which halved the throughput and left
  ``simulate`` 1.4x noisier than ``observe``'s tables; the tables themselves
  are unchanged and correct.
- Live noise (``slicer.depth_error``, hence ``simulate``) pins the narrow
  spectrograph field by default. slicersim >= 1.3.3 no longer rescales a new
  ``LazuliTarget`` to a field, which halved the throughput and left
  ``simulate`` 1.4x noisier than ``observe``'s tables; the tables themselves
  are unchanged and correct.

0.1.1 (2026-10-10)
------------------

- Absorbed the standalone ``lazuli-transit`` package as
  :mod:`lazuli_exo.transit`: ``FluxModel``, ``TransitModel`` (jaxoplanet),
  the NASA Exoplanet Archive cache (``download_exoplanet_archive`` /
  ``load_exoplanet_archive``) and the matplotlib helpers. The GPL-backed
  ``MoonTransitModel`` was not moved; it stays in the private lazuli-transit
  repo so this package remains BSD-clean. New extras ``transit``, ``archive``; the
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
  https://github.com/schmidt-observatories/lazuli-exo (moved from
  ``gummiks`` in 2026-10).
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
