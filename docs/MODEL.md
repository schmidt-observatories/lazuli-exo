# lazuli-exo: model, tables and caveats

Reference material moved out of the README. The README stays short; this
file keeps the physics, the table pipeline and the known approximations.

## Parametric model

`lazuli_exo.simulate(params)` builds a spectrum for any transiting planet from a
shared POSEIDON-derived opacity grid (`src/lazuli_exo/data/tables/atmo/zeta_grid.npz`) and a
PHOENIX stellar grid (`src/lazuli_exo/data/tables/star/phoenix_grid.npz`), with slicersim
computing the noise live (needs `slicersim` and `sncosmo` at call time;
`import lazuli_exo` itself stays numpy-only):

```python
import lazuli_exo
p = lazuli_exo.presets.showcase("WASP-39 b", "cloudy")   # or build the dict by hand
p.update(mu=None, log_P_cloud_bar=-2.0)
obs = lazuli_exo.simulate(p, R=100, n_transits=5)
obs.detect()
```

Knobs: `T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun, log_met (-1..3), c_to_o (0.3..1.0),
mu (None = equilibrium-chemistry value), log_P_cloud_bar, log_a_haze,
haze_slope, teff_K, logg, feh, mag, band (J/H/Ks/V), mag_system ("vega"
default, or "ab"), t14_s`. `R_p` is the band-median (0.40-1.70 um) transit
radius, which replaces the P_ref convention for this path. The physics is

    zeta(lambda) = -ln P_photo / bar      (tabulated: T, [M/H], C/O; two fiducials)
    zeta_p = zeta + 0.5 ln[(R_p H) / (R_p,fid H_fid)]   (blended between the fiducials)
           + 1.5 s [ln(r(lambda)/R_p) - ln(r_0(lambda)/R_fid)]   (local slant column, iterated)
    tau(z) = exp((zeta_p - s gamma_E - z)/s) + exp(zeta_haze + 1.5 ln(r/R_p) - z)
    zeta_eff = zeta_d + int_0^inf [1 - exp(-tau(zeta_d + u))] du
    R(lambda) = R_p / (1 - (H / R_p) (zeta_eff - median zeta_eff)),   depth = (R / R_s)^2

(zeta_p is an effective-radius level, s gamma_E above the gas tau_chord = 1
level for an exponential tau profile with slant-column exponent s read off
the two fiducials; zeta_haze = -ln P_haze is the haze tau_chord = 1 level
with exponent 1; the integral is the thin-atmosphere effective radius of the
summed profile over an opaque deck at zeta_d = -ln P_cloud, which for pure
gas is zeta_d + s Ein(exp((zeta_p - zeta_d)/s - gamma_E)); the last line is
the exact isothermal hydrostatic solution with g ~ 1/r^2; H = k T_eq /
(mu m_H g) with g at R_p). The chord column at wavelength lambda is
sqrt(2 pi r H(r)) ~ r^1.5 at that wavelength's own radius r(lambda), not at
R_p, so the gas level gains 1.5 s ln(r/R_p) less the same term for the
fiducial's own radius r_0(lambda) (already in the grid) and the haze level
gains 1.5 ln(r/R_p); r is the effective radius of the previous iterate and
the fixed point converges to 1e-4 in ln r in 2-4 steps (contraction
~1.5 s H/R_p; `info["column_iterations"]`, `info["column_correction_max_H"]`).
This matters for puffy planets (H/R_p >= 0.015; a 1 R_J, g = 1.7 m/s^2,
1000 K planet goes from 0.93 to 0.98 of the brute-force amplitude); the
residual 0.1-0.2 H at H/R_p = 0.033 is the second-order sphericity term
(the chord sampling r(z) across the tau profile, not just at r_eff), which
is not modelled. zeta is stored at the model's native
R = 10,000 so that a cloud deck clips individual line cores, as in POSEIDON,
and on two fiducial planets (1 R_J at g = 10 m/s^2; 0.3 R_J at g = 30 m/s^2) so that the
pressure dependence of the line cores is captured to first order.

`lazuli_exo.observe(planet, scenario)` runs the same model at the frozen
registry values (`presets.showcase`) and combines it with the precomputed
noise table, so the showcase and parametric paths share one model and one
radius convention (issue #3). The 14 x 5 POSEIDON spectra in
`precompute/tables/models/` are validation data only: isothermal, equilibrium
chemistry, gray deck, power-law haze (MacMad17 form, anchored at 0.35 um).
Validation is in `notebooks/06_parametric_model.ipynb`
(`precompute/validate_zeta_grid.py`). Note the convention difference: the
POSEIDON tables put the archive radius at 10 bar, the parametric path puts
it at the transit photosphere, ~10 scale heights higher; the validation
therefore feeds the model each table's own band-median radius. The transit
depth baseline `k = R_p/R_s` is taken from the registry radii (`R_p_RJ`,
`R_s_RSun`) on both paths and recorded as `meta["k_source"]`; the registry
`depth` field (archive `pl_ratror^2`) is informational.

**Magnitudes.** slicersim normalizes in AB; archive J magnitudes are Vega.
`lazuli_exo.slicer.depth_error` converts (J: +0.90 mag via sncosmo) and reports
the offset in `info`. The committed noise tables were regenerated with this
conversion on 2026-10-04 and are 1.4-1.5x more conservative than before for
the eleven unsaturated hosts; the three bright hosts that sit at the readout
floor (HAT-P-11 b now desaturated, HD 189733 b and HD 209458 b still flagged)
are saturation-limited and essentially unchanged.

**Instrument setup.** By default the readout is slicersim's and is shortened
(groups, then frames halved) until the host no longer saturates; the metadata
records `readout_mode = "auto"`, `default_nmd` and `nmd_used`. A caller that
shows this result next to its own single-exposure ETC can instead pass the
same observation: `depth_error(..., nmd=(N, M, D), tframe=, spectrograph_mode=,
guiding_sigma=, zodi_ab_mag=)` (also as keywords of `simulate`). A given `nmd`
is used as is and its saturation is *reported*, not removed
(`readout_mode = "fixed"`, `fully_desaturated`, `nsaturated_detpx`). The
native-noise cache keys on the setup, so the knobs are free to vary per call.

### Live instrument-noise cache

`simulate()` reuses up to 32 native slicersim results per process and caches
Vega-to-AB zero-point offsets for up to 32 bands. Native-result cache keys
include the exact stellar wavelength/flux arrays, converted AB magnitude,
band, instrument, and slicersim version/target implementation. Changing the
atmosphere, output resolution, duration, or transit count reuses the native
result; duration and transit-count scaling and metadata are recomputed for
each call. Different SED contents (including a different stellar grid) miss
the cache. Returned arrays and metadata can be edited without affecting it.

The cache assumes instrument/calibration files and registered bandpasses
stay fixed while the process runs. After changing them, call
`lazuli_exo.slicer.clear_noise_cache()` or restart the process. For an
independent reference calculation, call `slicer.depth_error(...,
use_cache=False)`; this bypasses both cache lookup and storage. The cache
is bounded and process-local, not persistent or shared across web workers.

## Two-stage precompute

```
inputs (POSEIDON opacities + fastchem grid)
      |
      v  poseidon conda env
atmo/zeta_grid.npz   star/phoenix_grid.npz   star/<planet>.npz
      |                                            |
      |                                            v  py313 env + slicersim
      |                                      noise/<planet>.npz
      v
runtime (src/lazuli_exo/) = numpy + these tables, nothing heavier

precompute/tables/models/<planet>__<scenario>.npz   POSEIDON validation data
```

Grids/star tables and noise tables are built in separate steps, in separate
conda environments, because POSEIDON and slicersim never need to be
installed together. `src/lazuli_exo/data/tables/` (~34 MB) is committed to
the repo and ships in the wheel as package data; the runtime only ever reads
it. The per-planet POSEIDON spectra (14 MB) live in `precompute/tables/` and
are read only by the validation script and notebook 06.

## Regenerating the tables

```bash
# 1. POSEIDON validation tables (-> precompute/tables/models) + star tables
#    (poseidon env; needs the opacity/chemistry inputs)
POSEIDON_input_data=$HOME/inputs/ PYSYN_CDBS=$HOME/inputs/stellar_grids \
  ~/anaconda3/envs/poseidon/bin/python precompute/make_model_tables.py [planet ...]

# 2. flat (featureless) validation tables (analytic, py313 env)
~/anaconda3/envs/py313/bin/python precompute/make_flat_tables.py

# 3. noise tables (py313 env; needs slicersim)
~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py

# 4. parametric grids (zeta: poseidon env, ~14 min, 6 GB; PHOENIX: poseidon env, ~1 min)
POSEIDON_input_data=$HOME/inputs/ ~/anaconda3/envs/poseidon/bin/python precompute/make_zeta_grid.py
PYSYN_CDBS=$HOME/inputs/stellar_grids ~/anaconda3/envs/poseidon/bin/python precompute/make_phoenix_grid.py
~/anaconda3/envs/py313/bin/python precompute/validate_zeta_grid.py   # 14 x 5 vs POSEIDON tables

# 5. limb-darkening tables (py313 env; needs exotic-ld, network on first run;
#    PHOENIX files cache to $EXOTIC_LD_DATA, default ~/inputs/exotic_ld_data)
~/anaconda3/envs/py313/bin/python precompute/make_ld_tables.py
```

Each script calls `tables.rebuild_index()` when it finishes, so `index.json`
always reflects what's actually on disk.

## Table schema

Under a tables directory:

```
index.json                              # version, planets (slug, has_noise, has_ld), grids
star/<planet-slug>.npz                  # wavelength_um, f_lambda, meta
noise/<planet-slug>.npz                 # wavelength_um, depth_err_1transit, meta
ld/<planet-slug>.npz                    # wavelength_um, u1, u2, meta
atmo/zeta_grid.npz, star/phoenix_grid.npz
```

Scenarios are not table files: `list_scenarios()` returns `presets.SCENARIOS`.

`index.json` carries a schema `version` integer; `lazuli_exo.tables` refuses to
load a mismatched version with a clear error. `observe()` resolves the
tables directory as: an explicit `tables_dir=` argument, else the
`LAZULI_EXO_TABLES` environment variable, else the `data/tables/` shipped inside the installed
repo/package.

## Known approximations

- **Isothermal equilibrium chemistry**: every scenario uses a single T = T_eq
  isotherm, not a real P-T profile.
- **Light-curve shape** (`lazuli_exo.lightcurve`): quadratically
  limb-darkened transit (numpy-only numerical disk integration, validated
  against the analytic uniform-disk and small-planet limits) with geometry
  from the frozen NEA registry (b from a/R*, cos i and the eccentricity
  factor; z(t) along a local circular arc of radius a(1-e^2)/(1+e sin w),
  the conjunction star-planet distance, so z(0) = b, traversed at the
  Keplerian conjunction sky speed).
  LD coefficients come from per-planet tables (`src/lazuli_exo/data/tables/ld/`) computed
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
  `lazuli_exo.detect.BAND_MASKS`) restrict the same statistic to fixed
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

## Deployment notes

The runtime ships with its lookup tables and no heavy dependencies (numpy,
plus matplotlib as an optional extra), so a future Django frontend can be
deployed on Heroku importing `lazuli-exo` directly — no POSEIDON, slicersim,
jax, or opacity data needs to reach the web dyno. That app is a separate,
not-yet-started follow-up project.

The tables live at `src/lazuli_exo/data/tables` and are declared as package
data in `pyproject.toml`, so an editable install, a repo checkout, a VCS pin
(`pip install git+https://...@<sha>`) and a built wheel all ship them (~50 MB).
`LAZULI_EXO_TABLES` overrides the location when a deployment wants the tables
somewhere else.

### Atmosphere-grid smoke builds

`precompute/make_zeta_grid.py --smoke` writes only
`atmo/zeta_grid.smoke.npz`, with a separate
`atmo/zeta_grid.smoke.partial.npz` checkpoint. Add `--resume` to resume that
smoke checkpoint. Production builds retain `zeta_grid.npz` and
`zeta_grid.partial.npz`; a smoke run never changes the production grid or
index. Completed grids are serialized to a temporary file beside the
destination and atomically replaced, so a failed write preserves the last
published grid. A failed build retains its checkpoint for resuming.
