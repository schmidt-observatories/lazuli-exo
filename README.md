# terminator

terminator simulates exoplanet transmission spectroscopy with the Lazuli IFS. For
a showcase planet, an atmosphere scenario, and a number of transits, it
returns the model transmission spectrum binned to the instrument with
statistical (photon/read-noise-only) error bars, plus a spectral-template
S/N for the atmosphere overall and per-band diagnostics. These are toy
H2/He scenarios with idealized noise — qualitative comparison material, not
calibrated detection significances or observing-time forecasts. The science
core is a numpy-only
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
import terminator

terminator.list_planets()                 # -> [str], 14 showcase targets
terminator.list_scenarios("WASP-39 b")    # -> ["clear", "cloudy", "flat", "hazy", "high_metallicity"]

obs = terminator.observe("WASP-39 b", scenario="cloudy", n_transits=5, R=100)
obs.wavelength      # bin centers [um]
obs.depth           # binned model transit depth
obs.depth_err       # per-bin 1-sigma uncertainty (scaled by 1/sqrt(n_transits))
obs.detect()        # {"template_snr_diagonal": 11.2, "band_snr": {"H2O": 7.4, ...}}
obs.simulate_points(seed=0)  # one noisy realization, for plotting
obs.plot()          # requires the [plot] extra
```

## Parametric model (any planet)

`terminator.simulate(params)` builds a spectrum for any transiting planet from a
shared POSEIDON-derived opacity grid (`src/terminator/data/tables/atmo/zeta_grid.npz`) and a
PHOENIX stellar grid (`src/terminator/data/tables/star/phoenix_grid.npz`), with slicersim
computing the noise live (needs `slicersim` and `sncosmo` at call time;
`import terminator` itself stays numpy-only):

```python
import terminator
p = terminator.presets.showcase("WASP-39 b", "cloudy")   # or build the dict by hand
p.update(mu=None, log_P_cloud_bar=-2.0)
obs = terminator.simulate(p, R=100, n_transits=5)
obs.detect()
```

Knobs: `T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun, log_met (-1..3), c_to_o (0.3..1.0),
mu (None = equilibrium-chemistry value), log_P_cloud_bar, log_a_haze,
haze_slope, teff_K, logg, feh, mag, band (J/H/Ks/V), mag_system ("vega"
default, or "ab"), t14_s`. `R_p` is the band-median (0.40-1.70 um) transit
radius, which replaces the P_ref convention for this path. The physics is

    zeta(lambda) = -ln P_photo / bar      (tabulated: T, [M/H], C/O; two fiducials)
    zeta_p = zeta + 0.5 ln[(R_p H) / (R_p,fid H_fid)]   (blended between the fiducials)
    zeta_eff = zeta_d + s Ein(exp((logaddexp(zeta_p, zeta_haze) - zeta_d)/s - gamma_E))
    R(lambda) = R_p / (1 - (H / R_p) (zeta_eff - median zeta_eff)),   depth = (R / R_s)^2

(the deck line is the exact effective radius of an exponential tau profile
over an opaque deck at zeta_d = -ln P_cloud, with s the slant-column exponent
read off the two fiducials; the last line is the exact isothermal hydrostatic
solution with g ~ 1/r^2; H = k T_eq / (mu m_H g) with g at R_p). zeta is stored at the model's native
R = 10,000 so that a cloud deck clips individual line cores, as in POSEIDON,
and on two fiducial planets (1 R_J at g = 10 m/s^2; 0.3 R_J at g = 30 m/s^2) so that the
pressure dependence of the line cores is captured to first order.

Fidelity is unchanged from the showcase tables: isothermal, equilibrium
chemistry, gray deck, power-law haze (POSEIDON MacMad17 form, anchored at
0.35 um), photon + read noise only. Validation against the 14 POSEIDON
planets is in `notebooks/06_parametric_model.ipynb`
(`precompute/validate_zeta_grid.py`). Note the convention difference: the
per-planet tables put the archive radius at 10 bar, the parametric path puts
it at the transit photosphere, ~10 scale heights higher; the validation
therefore feeds the model each table's own band-median radius.

**Magnitudes.** slicersim normalizes in AB; archive J magnitudes are Vega.
`terminator.slicer.depth_error` converts (J: +0.90 mag via sncosmo) and reports
the offset in `info`. The committed noise tables were regenerated with this
conversion on 2026-10-04 and are 1.4-1.5x more conservative than before for
the eleven unsaturated hosts; the three bright hosts that sit at the readout
floor (HAT-P-11 b now desaturated, HD 189733 b and HD 209458 b still flagged)
are saturation-limited and essentially unchanged.

## Documentation

Full docs (user guide, quick start, API reference, tutorials) are built with
Sphinx under `docs/sphinx/`:

```bash
pip install -e ".[docs]"   # plus a pandoc binary, for nbsphinx
cd docs/sphinx && make html
```

Then open `docs/sphinx/_build/html/index.html`.

Five runnable tutorial notebooks live in `notebooks/` and are rendered into
the docs with their committed outputs (they are not re-executed at build
time):

| Notebook | Covers |
|---|---|
| `01_getting_started` | `observe → detect → plot`, `SimulatedSpectrum` fields, reproducible noise |
| `02_scenarios` | The five scenarios, and why `hazy` can out-score `clear` |
| `03_transits_and_binning` | `n_transits` and `R`, the √N scaling, the resolution ceiling |
| `04_light_curves` | White-light and banded transits, `transit_flux` validation, limb darkening |
| `05_tables_and_targets` | Table layout, the index, the frozen NEA registry, caveat flags |

`notebooks/verification.ipynb` is a separate developer artifact (physical
sanity checks against the committed tables) and is not part of the docs.

To refresh the rendered outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
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

runtime (src/terminator/) = numpy + these tables, nothing heavier
```

Model/star tables and noise tables are built in separate steps, in separate
conda environments, because POSEIDON and slicersim never need to be
installed together. `src/terminator/data/tables/` (~50 MB) is committed to the repo, ships in the wheel as package data, and
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

# 4. limb-darkening tables (py313 env; needs exotic-ld, network on first run;
#    PHOENIX files cache to $EXOTIC_LD_DATA, default ~/inputs/exotic_ld_data)
~/anaconda3/envs/py313/bin/python precompute/make_ld_tables.py
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

`index.json` carries a schema `version` integer; `terminator.tables` refuses to
load a mismatched version with a clear error. `observe()` resolves the
tables directory as: an explicit `tables_dir=` argument, else the
`TERMINATOR_TABLES` environment variable, else the `data/tables/` shipped inside the installed
repo/package.

## Known approximations

- **Isothermal equilibrium chemistry**: every scenario uses a single T = T_eq
  isotherm, not a real P-T profile.
- **Light-curve shape** (`terminator.lightcurve`): quadratically
  limb-darkened transit (numpy-only numerical disk integration, validated
  against the analytic uniform-disk and small-planet limits) with geometry
  from the frozen NEA registry (b from a/R*, cos i and the eccentricity
  factor; z(t) along a local circular arc of radius a(1-e^2)/(1+e sin w),
  the conjunction star-planet distance, so z(0) = b, traversed at the
  Keplerian conjunction sky speed).
  LD coefficients come from per-planet tables (`src/terminator/data/tables/ld/`) computed
  by `precompute/make_ld_tables.py` with exotic-ld (PHOENIX grid, trilinear
  interpolation, 48 wavelength chunks). Planets without a registry entry or
  LD table fall back to a b = 0 trapezoid; `meta["shape"]` records which.
  Per-point noise is derived from the band's depth precision via
  sigma_pt = sigma_band * sqrt(t14 / 2c), consistent with the noise tables'
  equal out-of-transit baseline assumption.
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
- **Template S/N, not detection significance**: `detect()` returns a
  matched-template amplitude S/N against the best-fit flat line under a
  diagonal covariance (`template_snr_diagonal`) — there is no defined
  null/alternative likelihood ratio, nuisance model, covariance, or
  false-positive calibration. Per-band values (`band_snr`, masks in
  `terminator.detect.BAND_MASKS`) restrict the same statistic to fixed
  wavelength masks; they mix every species in the mask plus haze/cloud/
  continuum structure, and bands overlap (e.g. H2O and CH4 near 1.15 um) —
  they are not molecule detections. The rigorous upgrade path is a
  likelihood ratio against per-molecule leave-one-out model tables, which
  slots into the existing table schema without API changes.
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
  rather than silently ignored, and `observe()` raises a `UserWarning` for
  these hosts: their uncertainties are **unvalidated** (saturation can bias
  extraction weights, nonlinearity, and wavelength dependence in either
  direction), so treat their forecasts as indicative only until an
  unsaturated readout strategy or an injection/recovery validation exists.

## Local web demo

A single-page Flask + Plotly UI over the runtime (`webapp/`): pick planet,
scenario, transits, R, and noise seed; get the interactive spectrum, the
spectral-template S/N diagnostics, and simulated transit light curves
(white-light or split into wavelength bands). Local exploration tool only —
not the deployable frontend.

```bash
pip install -e ".[web]"
./run.sh                     # then open http://127.0.0.1:5000
```

Noise realizations happen server-side (`simulate_points(seed)`), so a given
seed reproduces the notebook figures exactly. Design notes:
`docs/superpowers/specs/2026-08-29-webapp-design.md`.

## Django/Heroku (planned)

The runtime ships with its lookup tables and no heavy dependencies (numpy,
plus matplotlib as an optional extra), so a future Django frontend can be
deployed on Heroku importing `terminator` directly — no POSEIDON, slicersim,
jax, or opacity data needs to reach the web dyno. That app is a separate,
not-yet-started follow-up project.

The tables live at `src/terminator/data/tables` and are declared as package
data in `pyproject.toml`, so an editable install, a repo checkout, a VCS pin
(`pip install git+https://...@<sha>`) and a built wheel all ship them (~50 MB).
`TERMINATOR_TABLES` overrides the location when a deployment wants the tables
somewhere else.
