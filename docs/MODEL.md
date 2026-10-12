# lazuli-exo: model, tables and caveats

Reference material moved out of the README. The README stays short; this
file keeps the physics, the table pipeline and the known approximations.

## Noise tables: 2026-10-11 regeneration and provenance

The 14 noise tables were rebuilt in one pass on `main` with slicersim 1.3.4
and the narrow field pinned, picking up #33 (saturation at the last read),
#34 (one PHOENIX SED for both paths), #37 (pixel-averaged SED, one reset
frame per ramp) and #44 (solar-metallicity grid, log g to 5.5). Against the
2026-10-04 tables the single-transit errors rose by 2-30 % at the median
(shorter ramps and the reset overhead), most at the bright hosts; HD 209458 b
and HAT-P-11 b now desaturate fully, HD 189733 b keeps 324 saturated pixels
at the (2,1,0) floor with the 0.7 s subframe and still carries the warning.
Every table's `meta` records `package_version`, `git_commit` (of the code
that built it), `slicersim_version`, `instrument_config_sha256` and
`phoenix_grid_sha256` (#4), and `index.json` (schema 4) holds a sha256 per
table file. `make_noise_tables.py` refuses nothing but marks the commit
`-dirty` when generator code is uncommitted.

## Bring your own spectrum

`simulate(star, spectrum=(wl_um, depth))` bypasses the parametric atmosphere
and runs the rest of the pipeline on a user transit depth (R_p/R_s)^2: the
PHOENIX star, live slicersim noise, stellar contamination and the LSF /
photon-weighted binning are identical to the parametric path (feeding the
parametric model's own native-grid output back in reproduces its binned
depths to 0 ppm). Only `teff_K, logg, feh, mag, t14_s` are required;
`R_p_RJ`/`R_s_RSun` set `flat_depth` when given, else it is the median
depth. The spectrum must be increasing in wavelength and reach into the
first and last bin; give it a few pixels of margin beyond 0.40-1.70 um,
because the LSF convolution reaches past the band and the interpolation
holds the edge value there (a spectrum cut exactly at the band edge moves
the last bin by ~100 ppm for WASP-39 b). This is the intended route to an
accurate atmosphere; a POSEIDON backend inside the package is not planned,
because POSEIDON and slicersim do not share an environment.

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
Validation is `precompute/validate_zeta_grid.py` (run by
`tests/test_validation_report.py` against the committed reference sets; it
fails, never skips, if they are missing). Each of the 70 showcase cases is
binned to R = 100 over 0.40-1.70 um with the constant offset removed and
gated against that planet's single-transit per-bin sigma: worst bin < 1 sigma
(G2), error-spectrum template S/N `sqrt(sum (res/sigma)^2)` < 3 (G5),
template S/N ratio model/table within 10 % (G3), and for the featureless
`flat` scenario a model amplitude < 10 ppm. A second set,
`precompute/tables/holdouts/zeta_holdouts.npz`, holds direct POSEIDON zeta
and depths at 15 off-node points (T = 650/1100/2450 K midpoints, 567/736/
1166 K, [M/H] 1.75, C/O 0.9, x = +0.63) so that grid interpolation itself is
gated, not only the node values every showcase planet sits on. Expected
failures are listed in `WAIVED` with the gate they waive and the issue that
removes them (#35 grid regeneration); a waiver
whose gate passes is itself an error. Note the convention difference: the
POSEIDON tables put the archive radius at 10 bar, the parametric path puts
it at the transit photosphere, ~10 scale heights higher; the validation
therefore feeds the model each table's own band-median radius. The transit
depth baseline `k = R_p/R_s` is taken from the registry radii (`R_p_RJ`,
`R_s_RSun`) on both paths and recorded as `meta["k_source"]`. The registry
freezes one radius ratio per planet (issue #7): `k` is the archive
`pl_ratror` (fallback `pl_radj/st_rad`, see `k_source`), `R_p_RJ` is derived
as `k R_s`, and `depth` is `k^2` by construction; the archive `pl_radj` is
kept as the informational `R_p_archive_RJ` (WASP-39 b: 1.279 vs 1.331 R_J
derived, a 7.7 % depth difference). `T_eq_K` is the archive `pl_eqt` as
frozen, not recomputed, because the POSEIDON validation tables were built at
those temperatures; `T_eq_conv_K = T_eff sqrt(R_s / 2a)` (A = 0, full
redistribution) is stored alongside, and four planets deviate from it by
more than 4 %: GJ 3470 b (0.87), HAT-P-11 b (1.04), WASP-107 b (0.955) and
K2-18 b (1.06, via the registry a/R* of 83.83 rather than its T_eq).

**Stellar SED.** `star.phoenix_sed(teff_K, logg, feh)` interpolates the
pysynphot PHOENIX `phoenixm00` folder (2000-12000 K, log g 3.5-5.5, R = 1000
storage, trilinear in log F / log T_eff). It is **solar metallicity by
construction**: the non-solar pysynphot folders are a different, low-resolution
(R ~ 741) product with missing log g columns and cool-star spectra wrong by
1-3 mag in V-J, and Gaia XP prefers the solar node for every metal-rich FGK
host (issue #34). The `feh` argument is accepted for API stability but any
non-zero value is clamped to 0 and reported in `info["clamped"]["feh"]`; T_eff
below 2000 K and log g outside 3.5-5.5 clamp to the edge node the same way
(`info["clamped"]["teff"/"logg"]`), T_eff above 12000 K raises. log g 5.5 is
only tabulated below 5100 K and log g 5.0 below 10200 K; hotter nodes repeat
the nearest available column (`meta["filled_from_nearest_logg"]`).
`precompute/make_noise_tables.py` builds the noise tables from this same SED,
so `observe()` and `simulate()` share one stellar spectrum (the committed
noise tables were regenerated this way on 2026-10-11, issue #34); the POSEIDON
star tables `star/<slug>.npz` are validation data only.

**Spectrograph field.** `depth_error` rescales the slicersim target to the
narrow field by default (`spectrograph_mode="narrow"`) instead of relying on
slicersim's default: from slicersim 1.3.3 a freshly built `LazuliTarget` is
left un-rescaled, with half the narrow-field throughput, which made the live
path 1.4x noisier than the committed tables (built in the narrow field with
slicersim 1.0.0). Pass `spectrograph_mode="wide"` for the wide field.

**Magnitudes.** slicersim normalizes in AB; archive J magnitudes are Vega.
`lazuli_exo.slicer.depth_error` converts (J: +0.90 mag via sncosmo) and reports
the offset in `info`. The committed noise tables were regenerated with this
conversion on 2026-10-04 and are 1.4-1.5x more conservative than before for
the eleven hosts slicersim reported as unsaturated. That report was wrong for
12 of the 14 hosts (saturation was tested at the (n-1)-group signal, see
"Known approximations"); the committed tables still carry it until the
regeneration tracked in #31, which will also apply the pixel-averaged SED and
the ramp reset overhead below (median `depth_err` x1.09-1.13 for GJ 1214 b,
TRAPPIST-1 e and WASP-39 b, x1.6-1.9 at 0.45 um; x1.14-1.30 for the three
bright hosts at the 0.7 s subframe).

**Instrument response (binning, GH #5).** A noise-table channel is one
detector pixel column (525 px, 0.400-1.699 um); slicersim's LSF is a Gaussian
of sigma = 1.15 px, constant in pixel space (resolution element 2.3 px,
`config/lazuli_cbe.toml` `dispersion_resolution`), so the physical resolving
power is `R_instr = lambda / (2.3 dlambda_px)` = 101 (0.99 um) to 369
(0.40 um), median 156 (`binning.R_instr`). `binning.instrument_response`
puts the R=10,000 model and the stellar photon flux on a fine grid uniform in
pixel coordinate (100 points/px), convolves `F (1 - d)` and `F` with the LSF,
integrates per pixel and returns `1 - sum_in/sum_out` per channel; the
within-pixel photon weight is the R=10,000 star table on the `observe()` path
(uniform if the table is missing) and the R=1000 `phoenix_sed` on the
`simulate()` path (coarser; ~76 ppm for GJ 1214 b at R=100 per the review).
The constant-R bin edges are snapped to channel edges so every bin is an
integer group of pixels, and the per-channel depths are combined with the
same `1/sigma^2` weights whose uncertainty `bin_sigma` returns, so `depth`
and `depth_err` describe the same estimator (this removes the 0.87-1.18
`depth_err` sawtooth at R=100). Before this, the unconvolved, unweighted mean
overshot the Na/K cores by 2-4 sigma per bin at R >= 100 (HD 189733 b R=200
3.7 sigma, WASP-39 b R=100 3.0 sigma); against the review's full forward
model the residual is now <= 0.1 sigma. Requesting `R > R_instr.min()` over
the binned range emits a `UserWarning` (not an error): the bins are then
narrower than a resolution element and their depths are correlated by the
LSF, which is fine for oversampled display but not for a statistical
forecast. `meta["instrument_response"]` records `lsf_sigma_px`,
`resolution_px`, `R_instr_min/max`, `photon_weighting` and `edges_snapped`;
the LSF constants are read from the noise-table meta when present
(`lsf_sigma_px`, `resolution_px`, to be written on the next table
regeneration, #31) and default to 1.15 / 2.3 otherwise.

**Instrument setup.** By default the readout is slicersim's and is shortened
(groups, then frames halved, then the 0.7 s rapid subframe `tframe` unless
the caller fixed it) until no detector pixel exceeds the full well at the
last read of the ramp; the metadata records `readout_mode = "auto"`,
`default_nmd`, `nmd_used` and `tframe_s`. Before slicersim sees the SED it
is box-averaged over each instrument pixel (`slicer.pixel_average_sed`),
because slicersim point-samples the spectrum at the pixel centres. A caller that
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
PYSYN_CDBS=$HOME/inputs/stellar_grids PYTHONPATH=src python precompute/make_phoenix_grid.py   # numpy + astropy only
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
  from the frozen NEA registry: z(t) is the exact sky-projected separation
  of the Keplerian orbit (`orbit.projected_separation`, Kepler's equation
  by Newton iteration, mid-transit at inferior conjunction so z(0) = b with
  the eccentricity factor; agrees with jaxoplanet to < 0.02 ppm in flux for
  every registry planet). The registry `t14_s` is the measured archive
  duration and sets the noise and the window, but the geometric duration
  from (P, a/R*, k, i, e, w) can differ (HAT-P-11 b: 7184 s vs 8480 s), so
  the orbit's clock is scaled by `t14_geom / t14_s` and the model's
  first-to-fourth-contact span equals `t14_s` (`meta["t14_geom_s"]`,
  `meta["time_stretch"]`).
  LD coefficients come from per-planet tables (`src/lazuli_exo/data/tables/ld/`) computed
  by `precompute/make_ld_tables.py` with exotic-ld (PHOENIX grid, trilinear
  interpolation, 48 wavelength chunks, fit over mu >= 0.1); each chunk is
  refitted under the Kipping (2013) constraints (u1 >= 0, u1 + u2 <= 1,
  u1 + 2 u2 >= 0) so the limb intensity is never negative, with the affected
  chunks listed in `meta["constrained_chunks"]` (only TRAPPIST-1 e, at
  0.41-0.46 um and 0.74 um). TRAPPIST-1 e's (2566 K) LD is uncertain at the
  0.1-0.4 level in u1 (it depends on the fit's mu_min and the grid vertex),
  and a 0.1 error in u1 biases a fitted white-light k^2 by ~150 ppm =
  6 sigma_band, so an empirical/JWST LD prior is recommended for that
  target. Planets without a registry entry or
  LD table fall back to a b = 0 trapezoid; `meta["shape"]` records which.
  Per-point noise is derived from the band's depth precision via
  sigma_pt = sigma_band * sqrt(t14 / 2c), consistent with the noise tables'
  equal out-of-transit baseline assumption; a depth + baseline fit to the
  light curve recovers sigma_band to ~10 % depending on the template
  (0.83-1.07x measured), and to 1/sqrt(2) with the baseline held fixed.
- **Stellar SED is solar metallicity, PHOENIX only**: see "Stellar SED"
  above. For M-dwarf hosts PHOENIX is 15-25 % too bright at 0.45 um and up to
  2x in the 0.60 um TiO bins against Gaia XP, so blue-end noise for GJ 1214 b,
  LHS 1140 b, GJ 3470 b and TRAPPIST-1 e is optimistic by ~10-25 % in sigma
  (review 2026-10-10, F4).
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
- **Stellar contamination (transit light source effect)**: off by default,
  i.e. forecasts for M/K hosts assume an immaculate photosphere unless
  `f_spot`/`f_fac` are set; with f_spot >= 0.03 the stellar signal is >= 1
  sigma per bin for every M/K showcase host and >= 10 % of the H2O amplitude
  for M dwarfs. `star.contamination` implements Rackham et al. 2018
  (AJ 853, 122) eq. 2,
  `eps = 1 / (1 - f_spot (1 - F_spot/F_phot) - f_fac (1 - F_fac/F_phot))`,
  with the PHOENIX surface fluxes from `star.phoenix_sed` at the host
  log g / [Fe/H]; defaults T_spot = 0.86 T_eff (T_eff < 4000 K, Rackham 2018)
  or 0.80 T_eff (Rackham 2019, FGK), T_fac = T_eff + 100 K. `simulate()` and
  `observe()` multiply the native-resolution depth by eps before binning
  (`meta["atmosphere"]["tlse"]` records the temperatures, fractions and the
  eps range over 0.40-1.70 um); `flat_depth` stays the geometric
  (R_p/R_s)^2, so the observed flat depth is eps-weighted. Literature
  covering fractions are in `star.TLSE_PRESETS` (data, not defaults).
  Contamination amplitude depth x (eps - 1), peak-to-peak 0.4-1.7 um, R=100,
  clear scenario, 1 transit (2026-10-10 review):

  | host | f_spot | ptp (ppm) |
  |---|---|---|
  | TRAPPIST-1 e | 0.03 / 0.39 | 103 / 2393 |
  | GJ 1214 b | 0.03 / 0.12 | 283 / 1274 |
  | LHS 1140 b | 0.03 / 0.12 | 108 / 490 |
  | K2-18 b | 0.03 / 0.10 | 50 / 184 |
  | GJ 3470 b | 0.03 / 0.10 | 108 / 393 |
  | WASP-107 b | 0.03 / 0.08 | 337 / 956 |
  | HAT-P-11 b | 0.03 / 0.08 | 60 / 172 |
  | HD 189733 b | 0.03 / 0.08 | 450 / 1283 |

- **Saturation is tested at the last read, not by slicersim's count**:
  slicersim 1.0.0 flags a pixel on `rate * (n-1)(m+d) * tframe`, the slope
  signal between group means, while the last read of the ramp accumulates
  `rate * nframes * tframe` — every (2, m, 0) ramp is tested at half its
  real charge (#33, upstream #6). `slicer._nsat_lastread` recomputes the
  count at the last read from slicersim's own projected cube (same pixel
  rate: cube / anamorphose x QE + effective dark) and drives the readout
  ladder and `fully_desaturated`; slicersim's own count is kept as
  `nsaturated_detpx_slicersim`. The committed tables predate this fix.
- **Residual detector saturation on bright hosts**: for HD 189733 b the
  ramp still saturates ~320 px at the (2, 1, 0) floor with the 0.7 s
  subframe (HD 209458 b and HAT-P-11 b are clean there). The residual
  `nsaturated_detpx` is recorded in the noise table's `meta` rather than
  silently ignored, and `observe()` raises a `UserWarning` for such hosts:
  their uncertainties are **unvalidated** (saturation can bias extraction
  weights, nonlinearity, and wavelength dependence in either direction), so
  treat their forecasts as indicative only until an unsaturated readout
  strategy (e.g. the wide field) or an injection/recovery validation exists.
  Whether a 0.7 s subframe exists for the full IFS field is a Pearl question.
- **Ramp reset overhead**: slicersim's observing time is `nframes * tframe *
  nramps` with no dead time between ramps. `depth_error` assumes one reset
  frame per MACC ramp (`n_reset_frames=1`, JWST-style accounting), so the
  wall clock entering the `sqrt(t14 / t_obs)` scaling is `(nframes + 1) *
  tframe * nramps` (`info["t_obs_s"]`; slicersim's value is `t_int_s`):
  x1.22 on `depth_err` for (2, 1, 0), x1.06 for (2, 4, 0), x1.03 for
  (2, 8, 0). Pearl should confirm the H4RG MACC reset accounting; set
  `n_reset_frames=0` to reproduce slicersim's zero-overhead timing.
- **SED pixel averaging**: slicersim samples the input SED at the pixel
  centres (`np.interp`). With the R=10,000 star tables (~28 samples per
  pixel) that is one sample of a spectrum with structure inside the pixel:
  the per-channel signal of an M dwarf was off by ~14 % rms and the R=100
  binned sigma by up to 33 % (#37). `depth_error` now box-averages the SED
  over each pixel first (exact for the piecewise-linear SED; a SED coarser
  than a pixel is unchanged), so slicersim's centre sample is the pixel
  mean and its LSF is applied once.

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
