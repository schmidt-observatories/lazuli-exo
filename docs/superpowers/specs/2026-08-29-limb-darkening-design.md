# Limb darkening + orbital geometry for light curves

**Date:** 2026-08-29
**Status:** Approved (chat, exotic-ld chosen over Claret-grid interpolation)
**Scope:** New per-planet LD tables (precompute stage), quadratic-LD transit
model in `terminator.lightcurve`, orbital geometry from the frozen target
registry. Spectrum/noise tables and their generators are untouched.

## What already exists

`src/terminator/targets.py` (frozen from NEA PSCompPars by
`precompute/freeze_targets.py`) already carries the orbital and stellar
parameters: `a_over_rs`, `inc_deg`, `ecc`, `w_deg`, `period_d`, `t14_s`,
`depth`, `T_eff_K`, `log_g_star`, `Met_star`. No POSEIDON or slicersim
re-runs are needed; only LD coefficients are new data.

## LD tables

- Generator: `precompute/make_ld_tables.py` (py313 env, `pip install
  exotic-ld`, network on first run). For each registry planet, exotic-ld
  v3.2 computes quadratic (u1, u2) from **PHOENIX** stellar models —
  consistent with the noise tables' `stellar_grid: phoenix`, and PHOENIX
  covers the cool hosts (TRAPPIST-1, 2566 K) that MPS-ATLAS's 3500 K floor
  would clamp. Stellar data auto-downloads to `~/inputs/exotic_ld_data`
  (only the files each star needs), matching the `~/inputs` convention.
- Sampling: 48 contiguous top-hat chunks log-spaced over 0.40–1.70 µm
  (`mode="custom"`, flat throughput per chunk). LD coefficients vary
  smoothly, so runtime bands interpolate `u1(λ), u2(λ)` at bin centers and
  average.
- Storage: `data/tables/ld/<slug>.npz` with `wavelength_um` (chunk centers),
  `u1`, `u2`, meta (grid name, Teff/logg/M_H used, any input clamping,
  exotic-ld version). New `tables.write_ld_table` / `tables.load_ld`;
  `rebuild_index` records `has_ld`. Additive — `SCHEMA_VERSION` stays 1.
- Tables are generated once and committed (~KB each); tests never need
  network or exotic-ld.

## Transit model (`terminator.lightcurve`)

- New public `transit_flux(z, k, u1, u2, n_r=501)`: normalized flux of a
  quadratically limb-darkened star occulted by an opaque disk of radius
  ratio `k` at projected separations `z` (stellar radii). Numpy-only
  numerical disk integration: midpoint quadrature over stellar radius with
  the analytic overlap-arc weight per annulus — exact to quadrature
  precision for any μ-law, no scipy/batman. Validated against the analytic
  uniform-disk overlap area (u1=u2=0) and the small-planet approximation.
- Geometry: `b = a/R* · cos i · (1−e²)/(1+e·sin ω)`;
  `z(t) = a/R* · sqrt(sin²θ + cos²i·cos²θ)` with `θ = 2π t f_e / P` and the
  eccentric velocity factor `f_e = (1+e·sin ω)/sqrt(1−e²)` (circular-orbit
  arc traversed at the conjunction speed; ω taken as NEA `pl_orblper`).
  Band depth sets `k_band = sqrt(depth_band)`; geometry is shared.
- `light_curves()` uses the registry + LD table when the planet has both;
  otherwise it falls back to the previous trapezoid (synthetic fixture
  planets, missing LD table). `meta["shape"]` records
  `"quadratic-ld"` or `"trapezoid"`; the LD path also reports `b`,
  `a_over_rs`, and per-band `u1`, `u2` on the LightCurveSet.
- Noise model unchanged: `sigma_pt = sigma_band · sqrt(t14/2c)` (the
  equal-baseline box assumption; documented approximation).

## Webapp

`/api/lightcurve` passes through `shape`, `b`, `a_over_rs`, and per-group
`u1`, `u2`; the UI shows a system-parameters line under the light-curve
panel and drops the "no limb darkening" caveat when the LD path is active.

## Testing

- `tests/test_lightcurve.py` additions: `transit_flux` vs analytic uniform
  disk; small-planet LD check; LD path on real tables (b, u1, u2 finite,
  mid-transit deeper than box depth for u1>0); fixture planets still take
  the trapezoid fallback (existing tests unchanged).
- `tests/test_tables.py` additions: ld write/load round-trip, index
  `has_ld`.
- Webapp round-trip test extended with the new fields.
