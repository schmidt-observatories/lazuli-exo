# slixer — Lazuli IFS Transmission Spectroscopy Simulator

**Date:** 2026-08-29
**Status:** Approved design
**Repo:** `GIT/slixer` (new project)

## Purpose

Simulate exoplanet transmission spectroscopy with the Lazuli IFS. For a
showcase planet, an atmosphere scenario, and a number of transits, slixer
produces (a) the model transmission spectrum binned to the IFS with realistic
error bars, and (b) detection significances for the atmosphere overall and for
individual molecules. The science core is a numpy-only runtime library backed
by precomputed lookup tables, so a later Django frontend can be hosted on
Heroku without shipping POSEIDON or its 71 GB of opacity inputs.

## Scope of this round

Library + precompute pipeline + generated tables + verification notebook.
The Django app is a separate follow-up project. This round is done when the
runtime API works from tables alone and the verification notebook passes
sanity checks.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Core output | Spectrum + error bars, plus per-feature detection significance vs N transits |
| Table parameterization | Curated hand-picked showcase list (~14 planets) |
| Scenarios per planet | Canonical set of 5 (4 POSEIDON + 1 analytic flat) |
| Noise model | Analytic per-bin depth precision from slicersim SNR; 1/√N transit scaling |
| Architecture | Two-stage precompute; POSEIDON isolated in its own conda env; runtime is numpy-only |
| Transit machinery | `lazuli_transit` (jaxoplanet-based, non-GPL) for ephemerides/durations and context light curves |
| Licensing constraint | No GPL dependencies anywhere in the stack |

## Architecture

Three stages, two environments:

1. **Model tables** (dedicated `poseidon` conda env: Python 3.11, `numpy<2`,
   POSEIDON installed editable from `GIT/POSEIDON`, opacity inputs at
   `~/inputs/` via the `POSEIDON_input_data` environment variable).
   `precompute/make_model_tables.py` computes a high-resolution transit-depth
   spectrum for every planet × scenario over the full IFS bandpass and writes
   one `.npz` per (planet, scenario) to `data/tables/models/`.

2. **Noise tables** (the regular py313 env with slicersim, jaxoplanet, and
   lazuli_transit). `precompute/make_noise_tables.py` builds a slicersim
   `LazuliTarget` for each host star (magnitude + spectral template), computes
   IFS SNR(λ) for an exposure equal to the transit duration T₁₄ (from
   lazuli_transit), assumes an equal out-of-transit baseline, and stores the
   single-transit depth uncertainty
   σ_depth(λ) = sqrt(1/SNR_in² + 1/SNR_out²)
   at native IFS sampling in `data/tables/noise/`.

3. **Runtime** (`src/slixer/`, imports numpy only; matplotlib as an optional
   `[plot]` extra). Loads the two table families, bins the model to a
   requested resolving power, attaches errors scaled by 1/√N_transits, and
   computes detection significances.

The runtime package never imports POSEIDON, slicersim, jax, or jaxoplanet.
`data/tables/` (~17 MB) is committed to the repo and shipped with the
package.

## Repo layout

```
slixer/
├── pyproject.toml            # runtime deps: numpy; extras: [plot] matplotlib
├── src/slixer/
│   ├── __init__.py           # observe(), list_planets(), list_scenarios()
│   ├── targets.py            # frozen showcase planet registry
│   ├── tables.py             # table reader; index.json schema + versioning
│   ├── observe.py            # SimulatedSpectrum
│   ├── detect.py             # significance math + molecule band masks
│   └── plotting.py           # spectrum plot (gks style)
├── precompute/               # never shipped to Heroku
│   ├── poseidon_env.yml
│   ├── scenarios.py          # planet list + scenario definitions (single source of truth)
│   ├── make_model_tables.py  # poseidon env
│   ├── make_noise_tables.py  # py313 env
│   └── freeze_targets.py     # NEA → targets.py via lazuli_transit (run rarely)
├── data/tables/
│   ├── index.json            # version, planets, scenarios, provenance
│   ├── models/<planet>__<scenario>.npz
│   └── noise/<planet>.npz
├── notebooks/verification.ipynb
├── tests/
└── docs/superpowers/specs/
```

## Showcase planets (~14)

HD 209458 b, HD 189733 b, WASP-39 b, WASP-96 b, WASP-107 b, WASP-121 b,
WASP-12 b, HAT-P-11 b, GJ 1214 b, GJ 3470 b, K2-18 b, TOI-270 d,
TRAPPIST-1 e, LHS 1140 b.

System parameters (R_p, M_p, R_s, T_eff, log g_s, T_eq, period, T₁₄, depth,
host magnitudes, host spectral type) are pulled once from the NASA Exoplanet
Archive via `lazuli_transit` helpers by `freeze_targets.py` and written into
`targets.py` as literal data. Runtime has no astroquery dependency. A planet
whose archive parameters are incomplete gets values filled from the
literature by hand in `scenarios.py` overrides, with provenance noted.

## Scenarios (5 per planet)

| Key | Description | POSEIDON setup |
|---|---|---|
| `clear` | Clear, solar metallicity | eq-chem, isothermal T = T_eq |
| `cloudy` | Grey cloud deck | as `clear` + opaque deck at P_cloud ~ 1 mbar |
| `high_metallicity` | 100× solar | eq-chem at 100× solar |
| `hazy` | Enhanced Rayleigh haze | as `clear` + enhanced-slope haze |
| `flat` | Featureless baseline | analytic constant depth = (R_p/R_s)² from targets.py (no POSEIDON call) |

Chemistry uses POSEIDON equilibrium-chemistry grids (present in
`~/Downloads/inputs/chemistry_grids`). Forward models are computed at high
resolution (R ≳ 5000) over the IFS bandpass read from the slicersim config,
with margin on both ends for rebinning. Scenario parameters (P_cloud, haze
strength, metallicity) live in `scenarios.py` as constants so a re-run is
reproducible. ~14 × 4 = 56 POSEIDON forward models; total precompute is
minutes.

## Runtime API

```python
import slixer

slixer.list_planets()                 # -> [str]
slixer.list_scenarios("WASP-39 b")    # -> [str]

obs = slixer.observe("WASP-39 b", scenario="cloudy", n_transits=5, R=100)
obs.wavelength          # bin centers [µm]
obs.depth               # binned model transit depth
obs.depth_err           # per-bin 1σ uncertainty
obs.model_wavelength, obs.model_depth   # native-resolution model
obs.detect()            # {"atmosphere": 11.2, "H2O": 7.4, ...} in σ
obs.simulate_points(seed=...)  # noisy realization for plotting
obs.plot()         # requires [plot] extra
```

`observe()` steps: load model + noise tables → rebin both to R (flux-
conserving binning for the model; inverse-variance combination for σ) →
scale σ by 1/√N_transits.

## Detection significance

- **`atmosphere`**: χ² of (scenario − flat baseline) against σ(λ), converted
  to a Gaussian-equivalent σ.
- **Per molecule**: the same statistic restricted to a wavelength band mask
  for that molecule (masks defined per molecule in `detect.py`). This is an
  approximation where bands overlap; the docstring says so explicitly. The
  clean upgrade path is per-molecule leave-one-out model tables (a scenario
  variant per removed molecule), which slots into the same table schema
  without API changes.

## Table schema

`models/<planet>__<scenario>.npz`: `wavelength` (µm, float64),
`transit_depth` (dimensionless), plus string metadata (scenario params,
POSEIDON version, creation date). `noise/<planet>.npz`: `wavelength` (µm,
native IFS sampling), `depth_err_1transit`, plus metadata (slicersim version,
T₁₄, magnitudes used). `index.json` carries a schema `version` integer;
`tables.py` refuses to load a mismatched version with a clear message.

## Error handling

- Unknown planet/scenario → `KeyError` listing valid options.
- Missing table file for an indexed entry → explicit error naming the file
  and the precompute script that regenerates it.
- Requested R above the native noise-table sampling → error, not silent
  upsampling.

## Testing

pytest, all runnable in py313 without POSEIDON (tests use the committed
tables plus tiny synthetic fixtures):

- table reader: schema/version checks, error paths
- binning: flux conservation, σ inverse-variance combination
- scaling: depth_err(N) = depth_err(1)/√N exactly
- detect: flat-vs-flat → 0σ; injected known offset → analytic expected σ
- API: every indexed (planet, scenario) loads and produces finite outputs

Verification notebook (built with the notebook-demo skill, gks plot style):
WASP-39 b and K2-18 b spectra with error bars, σ-vs-N_transits curves,
and sanity checks (e.g., water feature amplitudes of expected order,
hot Jupiter ≫ temperate sub-Neptune SNR per transit).

## Non-goals (this round)

- Django app, Heroku deployment (follow-up project; this design keeps the
  runtime Heroku-compatible but builds no web code)
- POSEIDON retrievals on simulated data
- Full light-curve injection/recovery (analytic noise only; a Monte Carlo
  spot-check may be added later)
- Generic parameter-grid interpolation between planets
