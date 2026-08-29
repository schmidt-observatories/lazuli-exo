# slixer

slixer simulates exoplanet transmission spectroscopy with the Lazuli IFS. For
a showcase planet, an atmosphere scenario, and a number of transits, it
returns the model transmission spectrum binned to the instrument with
realistic error bars, plus detection significances for the atmosphere
overall and for individual molecules. The science core is a numpy-only
runtime backed by precomputed lookup tables — no POSEIDON, slicersim, jax,
or astropy at import time — so it can later sit behind a Django app on
Heroku without shipping POSEIDON's opacity inputs (tens of GB).

## Install

```bash
pip install -e ".[plot,test]"
```

`plot` pulls in matplotlib (for `.plot()`); `test` pulls in pytest.

## Quickstart

```python
import slixer

slixer.list_planets()                 # -> [str], 14 showcase targets
slixer.list_scenarios("WASP-39 b")    # -> ["clear", "cloudy", "flat", "hazy", "high_metallicity"]

obs = slixer.observe("WASP-39 b", scenario="cloudy", n_transits=5, R=100)
obs.wavelength      # bin centers [um]
obs.depth           # binned model transit depth
obs.depth_err       # per-bin 1-sigma uncertainty (scaled by 1/sqrt(n_transits))
obs.detect()        # {"atmosphere": 11.2, "H2O": 7.4, ...} in sigma
obs.simulate_points(seed=0)  # one noisy realization, for plotting
obs.plot()          # requires the [plot] extra
```

## Two-stage precompute

```
inputs (POSEIDON opacities + fastchem grid)
      |
      v  poseidon conda env
models/<planet>__<scenario>.npz   star/<planet>.npz
      |
      v  py313 env + slicersim
noise/<planet>.npz   (per-planet single-transit depth_err(lambda))

runtime (src/slixer/) = numpy + these tables, nothing heavier
```

Model/star tables and noise tables are built in separate steps, in separate
conda environments, because POSEIDON and slicersim never need to be
installed together. `data/tables/` (a few MB) is committed to the repo and
shipped with the package; the runtime only ever reads it.

## Regenerating the tables

```bash
# 1. model + star tables (poseidon env; needs the opacity/chemistry inputs)
POSEIDON_input_data=$HOME/inputs/ PYSYN_CDBS=$HOME/inputs/stellar_grids \
  ~/anaconda3/envs/poseidon/bin/python precompute/make_model_tables.py [planet ...]

# 2. flat (featureless) baseline tables (analytic, py313 env)
~/anaconda3/envs/py313/bin/python precompute/make_flat_tables.py

# 3. noise tables (py313 env; needs slicersim)
~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py
```

Each script calls `tables.rebuild_index()` when it finishes, so `index.json`
always reflects what's actually on disk.

## Table schema

Under a tables directory:

```
index.json                              # version, planets, scenarios, provenance
models/<planet-slug>__<scenario>.npz    # wavelength_um, transit_depth, meta
star/<planet-slug>.npz                  # wavelength_um, f_lambda, meta
noise/<planet-slug>.npz                 # wavelength_um, depth_err_1transit, meta
```

`index.json` carries a schema `version` integer; `slixer.tables` refuses to
load a mismatched version with a clear error. `observe()` resolves the
tables directory as: an explicit `tables_dir=` argument, else the
`SLIXER_TABLES` environment variable, else the `data/tables/` shipped in the
repo/package.

## Known approximations

- **Isothermal equilibrium chemistry**: every scenario uses a single T = T_eq
  isotherm, not a real P-T profile.
- **Cold-planet temperature floor**: the fastchem equilibrium-chemistry grid
  has a hard floor at 300 K. K2-18 b (T_eq ~ 284 K), LHS 1140 b (~226 K), and
  TRAPPIST-1 e (~250 K) are clamped to 300 K for both the isotherm and the
  chemistry lookup; this is recorded per-table as `T_clamped` in `meta`.
- **"hazy" is deck_haze, not haze-alone**: POSEIDON's `cloud_type="haze"`
  alone is a no-op for a 1D model (it silently zeroes the cloudy-sector
  fraction, so the haze opacity is computed but never mixed into the
  spectrum — confirmed by an empirical A/B check). The `hazy` scenario
  instead uses `cloud_model="MacMad17"`, `cloud_type="deck_haze"`, with the
  deck's own opaque cloud pinned at the pressure grid floor (100 bar) so it
  never actually shadows the atmosphere — only the haze term contributes
  opacity.
- **Band-mask molecule significances**: per-molecule detection significance
  restricts the same chi-squared statistic to a fixed wavelength band mask
  per molecule (`slixer.detect.BAND_MASKS`). Bands overlap (e.g. H2O and CH4
  near 1.15 um), so these are approximate, not a joint fit. A rigorous
  upgrade path is per-molecule leave-one-out model tables, which slots into
  the existing table schema without API changes.
- **sqrt(N) SNR scaling**: `depth_err` scales as `1/sqrt(n_transits)` from a
  single precomputed single-transit uncertainty; there's no per-transit
  systematics or red-noise floor.
- **Equal out-of-transit baseline**: the noise model assumes an out-of-transit
  baseline exposure equal to the in-transit one, i.e.
  `sigma_depth = sqrt(1/SNR_in^2 + 1/SNR_out^2)` with `SNR_in == SNR_out`,
  not an optimized or telescope-time-constrained baseline.
- **Residual detector saturation on bright hosts**: for HAT-P-11 b,
  HD 189733 b, and HD 209458 b, slicersim's detector ramp cannot be fully
  de-saturated even at the minimal MACC configuration (2 groups x 1 frame).
  The residual `nsaturated_detpx` is recorded in the noise table's `meta`
  rather than silently ignored; `depth_err` for these three hosts is mildly
  pessimistic relative to a hypothetical unsaturated readout.

## Django/Heroku (planned)

The runtime ships with its lookup tables and no heavy dependencies (numpy,
plus matplotlib as an optional extra), so a future Django frontend can be
deployed on Heroku importing `slixer` directly — no POSEIDON, slicersim,
jax, or opacity data needs to reach the web dyno. That app is a separate,
not-yet-started follow-up project.
