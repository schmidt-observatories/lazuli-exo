# Parametric transit simulator on the Lazuli simulators site

**Date:** 2026-10-04
**Status:** Draft for review
**Scope:** Two repos. `terminator` gains a parametric atmosphere model and a
stellar SED mini-grid (both numpy-only at runtime) plus the two precompute
scripts that build them. `lazuli-etc-web2` (the Heroku app behind
`simulators.schmidtobservatorysystem.org`) gains one page that drives them,
with slicersim computing the noise live and the NASA Exoplanet Archive
picker prefilling the form. No deploy is part of this work; Gummi deploys.

## Purpose

Today `observe(planet, scenario)` can only show 14 showcase planets under 5
canned atmospheres, because every spectrum is a POSEIDON run frozen into a
table. The goal is a public page where any transiting planet, archive or
hand-entered, can be simulated with the atmosphere knobs that matter
(T_eq, mean molecular weight, metallicity, C/O, cloud deck, haze) and the
stellar knobs that set the noise (T_eff, log g, [Fe/H], R_s, magnitude), at
a cost per request of about one second and a few MB of new data on the dyno.

Fidelity class is unchanged: isothermal, equilibrium chemistry, photon and
read noise only. The page says so.

## Why no big grid is needed

For an isothermal, well-mixed atmosphere the wavelength-dependent transit
radius is

    R(λ) = R_p + H · (ζ(λ) − ζ_ref),     H = k T_eq / (μ m_H g)

where ζ(λ) = −ln P_photo(λ)/1 bar is the log of the pressure at which the
slant optical depth reaches unity. ζ depends on the gas opacity, i.e. on
(T, [M/H], C/O), and is independent of g, R_p, R_s and (to first order) μ.
Everything a user types other than those three enters analytically:

| Input | Enters through |
|---|---|
| T_eq, μ, g (= G M_p / R_p²) | H |
| R_p, R_s | the quadratic, depth = (R(λ)/R_s)² |
| cloud-top pressure P_c | ζ ← max(ζ, −ln P_c) |
| haze (P_h at 0.5 µm, slope γ) | 1/P_tot = 1/P_gas + 1/P_haze(λ), P_haze = P_h (λ/0.5 µm)^(−γ) |
| [M/H] → μ | μ_chem([M/H]) stored on the grid; user may override |

So POSEIDON runs once, offline, for one fiducial planet on a 3-D grid, and
that grid serves every planet. The 14 committed POSEIDON tables become the
validation set for this approximation.

## Data products (built offline, shipped in `data/tables/`)

### `atmo/zeta_grid.npz` — opacity-shape grid

- Axes: `T_K` = 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600,
  1800, 2000, 2300, 2600, 3000 (16); `log_met` = −1 … 3 step 0.5 (9);
  `c_to_o` = 0.3, 0.55, 0.8, 1.0 (4). 576 POSEIDON spectra.
- `wavelength_um`: constant R = 2000 over 0.35–1.85 µm (~3,300 points).
- `zeta[T, met, co, wl]` float32: ζ binned by mean from the R = 10,000 run.
- `mu[T, met, co]` float32: POSEIDON's mean molecular weight at the
  photospheric layers, so [M/H] gives a default μ.
- `meta`: fiducial planet (R_p = 1 R_J, g = 10 m s⁻², R_s = 1 R_☉,
  P_ref = 10 bar, pressure grid, species list, POSEIDON and fastchem
  versions, creation date).
- Built by `precompute/make_zeta_grid.py` in the `poseidon` env. For each
  grid point: `make_atmosphere` → `compute_spectrum` → convert the depth to
  R(λ) → map R(λ) to P through the atmosphere's own r(P) column →
  ζ = −ln P. T below 300 K is not on the grid; runtime clamps the chemistry
  lookup to 300 K and reports it (same behaviour as today's `T_clamped`).
- Size: 576 × 3,300 × 4 B ≈ 7.6 MB. Run time ≈ 1 h on the laptop; opacities
  are loaded once with `T_fine` spanning the grid, chunked by T if memory
  forces it.

### `star/phoenix_grid.npz` — stellar SED mini-grid

- Source: the pysynphot PHOENIX grid already at
  `~/inputs/stellar_grids/grid/phoenix` (`PYSYN_CDBS`), the same grid the
  current star tables come from.
- Axes: `teff_K` = every grid step from 2300 to 8000 K (~45);
  `logg` = 3.5, 4.0, 4.5, 5.0 (the grid tops out at 5.0; M dwarfs at
  5.0–5.1 clamp); `feh` = −1.0, −0.5, 0.0, +0.3, +0.5 (folders m10, m05,
  m00, p03, p05).
- `wavelength_um`: constant R = 1000 over 0.30–2.50 µm (~2,100 points),
  wide enough to normalize in V, J, H and Ks.
- `log_flam[teff, logg, feh, wl]` float32, F_λ in erg s⁻¹ cm⁻² Å⁻¹ at the
  stellar surface (absolute level is irrelevant; slicersim renormalizes).
- Built by `precompute/make_phoenix_grid.py` (poseidon env, pysynphot).
- Size ≈ 45 × 4 × 5 × 2,100 × 4 B ≈ 7.6 MB.

Both files are read by `terminator.tables` with the existing version check;
`index.json` lists them under new `atmo` and `star_grid` keys. Schema
`version` bumps.

## terminator runtime API (numpy only)

```python
terminator.atmosphere.transmission_depth(
    wl_um, *, T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun,
    log_met=0.0, c_to_o=0.55, mu=None,
    log_P_cloud_bar=None, log_P_haze_bar=None, haze_slope=-4.0,
    tables_dir=None) -> (depth, info)
```

- Trilinear interpolation of ζ (and μ) in (T_eq clamped to grid, log_met,
  c_to_o); out-of-range log_met / c_to_o raise `ValueError` with the range.
- `mu=None` → μ_chem from the grid. `info` returns `g`, `H_m`, `H_over_Rp`,
  `mu_used`, `mu_source` ("chemistry" or "user"), `T_chem_K`, `T_clamped`,
  `zeta_ref`. ζ_ref is the median ζ over 0.40–1.70 µm, so **R_p is the
  band-median transit radius**, the quantity the archive reports. This
  replaces the open P_ref calibration question for the new path.
- Interpolated onto the caller's `wl_um` by `np.interp`; the web view asks
  for the grid's own wavelengths and bins with the existing
  `binning.bin_model`.

```python
terminator.star.phoenix_sed(teff_K, logg, feh, tables_dir=None)
    -> (wl_angstrom, f_lambda, info)
```

- Trilinear in (log T_eff, log g, [Fe/H]) on `log_flam`; log g and [Fe/H]
  clamp to the grid edges and say so in `info`; T_eff below 2300 K falls
  back to a Planck curve with `info["fallback"] = "blackbody"`, as the
  current star-table script does for TRAPPIST-1; above 8000 K raises.

```python
terminator.slicer.depth_error(wl_A, f_lambda, *, mag, band, mag_system="vega",
                              t14_s, n_transits=1, instrument="lazuli_cbe")
    -> (wl_um, depth_err, info)
```

- Lazy `import slicersim` and `import sncosmo` inside the function, so
  `import terminator` stays numpy-only.
- Vega → AB per band: `m_AB = m_Vega + vega.refspectrum.bandmag(band, "ab")`
  with sncosmo's Vega reference spectrum. No hard-coded offsets. The offset
  used is returned in `info`.
- Desaturation loop moves here verbatim from `make_noise_tables.py`
  (shrink groups, then frames per group; floor at 2 groups × 1 frame;
  residual `nsaturated_detpx` reported, never hidden).
- `depth_err = sqrt(2) / (snr · sqrt(t14 / t_obs)) / sqrt(n_transits)`,
  the same equal-baseline assumption as today.
- `make_noise_tables.py` is rewritten to call this function, so the
  per-planet tables and the live page share one code path.

```python
terminator.simulate(params: dict, R=100, n_transits=1, tables_dir=None)
    -> SimulatedSpectrum
```

- Thin assembler: `transmission_depth` + `depth_error` + binning, returning
  the existing `SimulatedSpectrum` so `detect()`, `simulate_points()` and
  `plot()` work unchanged. `planet` and `scenario` fields carry the user's
  label and `"custom"`.
- `observe(planet, scenario)` keeps working on the committed tables in
  this version. Its presets are also expressible as `simulate` parameters
  (see Presets). Removing the per-planet model tables is a follow-up once
  validation passes, not part of this change.

### Presets

`terminator.presets.showcase(planet, scenario) -> params` maps the frozen
registry plus scenario to `simulate` inputs:

| scenario | log_met | c_to_o | log_P_cloud | haze |
|---|---|---|---|---|
| clear | 0 | 0.55 | none | none |
| cloudy | 0 | 0.55 | −3 | none |
| high_metallicity | 2 | 0.55 | none | none |
| hazy | 0 | 0.55 | none | (P_h, γ) fitted in validation to match the POSEIDON hazy tables |
| flat | 0 | 0.55 | −6 (opaque above the photosphere) | none |

### Validation (precompute/validate_zeta_grid.py, run before merge)

For each of the 14 planets × 5 scenarios, bin both the committed POSEIDON
table and the `transmission_depth` reconstruction (registry values, preset
params) to R = 100 and report the maximum |Δdepth| and the error in
feature amplitude (max − min). Acceptance: feature amplitude within 10 %
for clear and high_metallicity, within 20 % for cloudy and hazy, after a
constant offset is removed (the offset is the P_ref convention, which the
new path defines away). Results are written to the spec's companion
notebook `notebooks/06_parametric_model.ipynb`. If a planet misses, the
escape hatch is a fourth grid axis in log g; the schema allows it without
API change.

## lazuli-etc-web2 page

Route `/ifs/transit/`, template `templates/main/ifs_transit.html`, app
module `ifs/transit_views.py` (+ `ifs/transit_share.py` mirroring
`ifs/share.py`). Styling, Bokeh theme, nav, and permalink conventions follow
the existing pages.

### Form

1. **Target lookup.** Datalist of transiting archive planets and a Lookup
   button, reusing `wcc.transit_help.planet_names` and
   `_planet_star_value` (promoted to a public `planet_value`). A JSON
   endpoint `/ifs/transit/planet-params/?name=` returns every field below
   that the archive has; the page fills the boxes and lists what it could
   not find, using the WCC page's banner JavaScript moved into
   `static/js/archive_lookup.js` so both pages share it.
2. **Preset** dropdown: the 14 showcase planets × 5 scenarios, filled from
   `terminator.presets`.
3. **Star:** T_eff [K], log g, [Fe/H], R_s [R_☉], magnitude, band
   (2MASS J default; V, H, Ks), magnitude system (**Vega default**, AB).
   Archive source columns: `st_teff`, `st_logg`, `st_met`, `st_rad`,
   `sy_jmag` / `sy_vmag` / `sy_hmag` / `sy_kmag` (all Vega).
4. **Planet:** R_p [R_J], M_p [M_J], T_eq [K], t14 [h]. t14 comes from
   `pl_trandur`, else from P, a/R_s, i, e, ω with the formula in
   `freeze_targets.py` (moved into `terminator.targets` so both use it).
5. **Atmosphere:** [M/H] (log, −1…3), C/O (0.3…1.0), μ (box shows the
   chemistry value, editable; an "auto" checkbox restores it), cloud-top
   pressure (log bar slider, −6…1, with "clear" at the top), haze off/on
   with log P_h at 0.5 µm and slope γ (default −4).
6. **Observation:** number of transits (1…1000), R (10…500), noise seed,
   instrument configuration (CBE default, EOL).

Validation errors return the form with a message, never a 500, following
the IFS view's pattern.

### Output

- Bokeh plot: native model, binned model, one noisy realization with error
  bars, flat baseline line. Hover shows wavelength, depth, σ in ppm.
- Table: template S/N and per-band S/N from `detect()`, with the README's
  caveat text that these are not molecule detections.
- Derived-quantities panel: g, H, H/R_p, mean feature amplitude [ppm],
  μ used and its source, T clamp if any, Vega→AB offset and the AB
  magnitude handed to slicersim, t_obs, readout `nmd` used, residual
  saturation flag (red if non-zero), t14 source.
- CSV download (wavelength, depth, depth_err, realization) via the cached
  CSV pattern already in `ifs/views.py`.
- Permalink with `v=1` and every form field, via `transit_share.py`.
- Light curves are **not** in this version: they need per-planet
  limb-darkening tables that do not exist for custom stars. Note in the
  page; candidate follow-up with exotic-ld coefficients on the same
  PHOENIX axes.

### Cost on the dyno

| Item | Budget |
|---|---|
| New data in the slug | ~15 MB (two npz files) |
| Resident memory | +15 MB on first use, lazily loaded, cached in-process |
| Per request | ζ and SED interpolation < 10 ms; slicersim 0.2 s warm, up to ~1.5 s for bright hosts through the desaturation loop |
| Cache | slicersim result keyed on rounded (T_eff, log g, [Fe/H], m_AB, band, config) in the Django cache, 1 h |

### Dependencies and pins

- `requirements.txt` adds `terminator @ git+https://github.com/<org>/terminator.git@<sha>`,
  pinned by SHA like wcc-etc. Prerequisite: terminator is pushed to a
  GitHub repo the Heroku build can clone (today the local repo has no
  remote). The tables ship in the repo under `data/tables`, which the
  editable/VCS install includes; `TERMINATOR_TABLES` is not needed on
  Heroku as long as the install is from the git URL (same as today's
  README note).
- slicersim and sncosmo are already installed and pinned.

## Audit items closed or carried

- **Vega/AB (audit #1):** closed by construction for the page; the default
  path converts archive Vega J to AB per band. `make_noise_tables.py` gets
  the same conversion when it is rewritten over `depth_error`, so the
  committed tables are corrected in the same regeneration pass.
- **P_ref calibration:** defined away for the parametric path (R_p is the
  band-median radius). Still open for `observe()` until the tables are
  regenerated.
- **slicersim saturation undercount:** upstream; carried, and the page
  shows the residual saturation count rather than hiding it.
- **Noise table provenance:** `depth_error`'s `info` records the slicersim
  version and instrument config; the page displays them.

## Testing

terminator (`pytest`):
- `transmission_depth`: exact recovery at grid nodes; doubling T_eq at
  fixed chemistry doubles the feature amplitude; cloud deck lowers the
  amplitude monotonically with decreasing P_c; haze with γ = −4 steepens
  the blue end; `mu=None` equals the grid μ; ranges raise.
- `phoenix_sed`: node recovery; blackbody fallback below 2300 K;
  clamping reported.
- `depth_error`: skipped unless slicersim imports; one canary that the
  Vega→AB J offset is 0.85–0.95 and that a J = 7 star reaches the
  desaturation floor with the residual count reported.
- `simulate` end to end against a showcase preset within the validation
  tolerance.

lazuli-etc-web2 (`manage.py test ifs`):
- Form parsing and validation (mocked `terminator.simulate`), the
  planet-params endpoint for a known planet, an unknown name, and a planet
  missing a magnitude; share-field round trip; one unmocked slicersim
  canary, as the IFS page does.

## Out of scope

Light curves, non-isothermal P-T profiles, stellar activity, red noise,
per-molecule detections, user-uploaded spectra, and the Heroku deploy
itself.
