Changelog
=========

The project is pre-release; there has been no tagged version yet, so the
entries below group the development history by theme rather than by release.

Unreleased
----------

- ``LightCurveSet.plot_transit(t_hr)`` and ``LightCurveSet.animate()``
  (``plotting.plot_transit`` / ``plotting.animate_transit``): the transit
  seen band by band, a limb-darkened stellar disk per band with the planet
  on the registry orbit over the light curves filling in with time; a still
  for talks, or a ``FuncAnimation`` to save as GIF or MP4. Custom planets
  get a uniform disk and a central chord.

0.1.2 (2026-10-12)
------------------

First PyPI release (``pip install lazuli-exo``). The ``v0.1.1`` tag was
never published; everything below the 0.1.1 heading is the state of that
tag, and everything in this section landed since.

- Docs aligned with wcc-etc: a :doc:`user_guide/plotting` page whose
  figures are drawn at build time, ``docs/MODEL.md`` included in the
  reference as :doc:`model`, a conda section on the installation page, and
  a one-page primer plus glossary at the top of :doc:`user_guide/concepts`
  for readers new to transmission spectroscopy. Prose pass over every page.
- Release decisions (#31): HD 189733 b stays in the tables with its
  saturation warning; the internal design notes under ``docs/superpowers/``
  are no longer tracked in the public tree.
- Release scrub (#31): ``numpy>=2.0`` is declared (the runtime uses
  ``np.trapezoid``); the test workflow runs on Python 3.10-3.13; ``run.sh``
  binds to 127.0.0.1 without ``--debug``; README and
  :doc:`installation` document ``pip install lazuli-exo`` and the extras.
- Plotting as methods on the result objects: ``obs.plot()`` gained
  ``show_points``, ``units`` ("percent", "ppm", "fraction"), ``label`` and
  ``title`` so scenarios overlay on one Axes; new ``obs.plot_noise()``
  (per-bin error against wavelength) and ``LightCurveSet.plot()`` (one
  curve per band, stacked). ``plotting.wavelength_axis`` is public. The
  plotting functions no longer apply a matplotlib style globally.
- Docs refreshed for the parametric runtime: ``meta["model"]`` references
  (gone since #19) now point at ``meta["params"]`` / ``meta["atmosphere"]``;
  scenarios are described as presets; the binning page describes the LSF
  and photon-weighted instrument response; the two resolution limits, the
  single remaining saturated host and the validation gates are current.
  Tutorial notebooks 01-05 are rebuilt from ``notebooks/build_0N.py``
  scripts like 06-09, and 06 is regenerated from its builder.

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
- Release plumbing (#31 section 1): ``publish.yml`` builds sdist + wheel
  from a clean checkout, checks the wheel ships exactly the tracked tables,
  and uploads with PyPI trusted publishing on a ``v*`` tag (manual dry run
  to TestPyPI); the same wheel check runs in CI. ``lazuli_exo.__version__``
  is the single version source; ``pyproject.toml`` gains readme, authors,
  urls and classifiers. ``observe``/``simulate`` reject non-positive or
  non-integer ``R`` and ``n_transits`` (``R=0`` was a ZeroDivisionError,
  ``n_transits=2.5`` was applied but reported as 2).
- Noise tables regenerated (2026-10-11) with saturation tested at the last
  read, the pixel-averaged PHOENIX SED, the reset-frame overhead and the
  narrow field pinned: single-transit errors 2-30 % larger at the median;
  HD 209458 b and HAT-P-11 b now desaturate fully. Tables carry package
  version, git commit and input hashes; ``index.json`` schema 4 adds a
  per-file sha256 manifest (#4, #33, #34, #37).
- ``simulate(star, spectrum=(wl_um, depth))`` accepts a transit spectrum
  from any forward model in place of the parametric atmosphere; the noise,
  contamination and instrument response are unchanged
  (``meta["atmosphere"]["source"] == "user"``). README and index now state
  the scope: quantitative instrument half, surrogate atmosphere half.
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
