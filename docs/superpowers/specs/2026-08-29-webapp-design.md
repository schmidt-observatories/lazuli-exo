# terminator webapp — local Flask + Plotly demo

**Date:** 2026-08-29
**Status:** Approved (chat), implemented same day
**Scope:** Local exploration tool only. Not the deployable frontend; no
Heroku packaging, no persistence, no auth, no config. Nothing under
`src/terminator/` changes.

## Purpose

A single-page browser UI over the `terminator` runtime: pick a planet,
scenario, number of transits, spectral resolution, and noise seed; see the
simulated transmission spectrum (interactive Plotly plot) and the predicted
detection significances.

## Layout

```
webapp/
  __init__.py        # empty; lets tests import webapp.app
  app.py             # Flask app factory + two JSON endpoints
  templates/index.html
  static/app.js      # controls + Plotly wiring
```

Run: `flask --app webapp/app run` (or `python webapp/app.py`).
Install: `pip install -e ".[web]"` — new `web = ["flask"]` optional
dependency in `pyproject.toml`. Plotly loads from CDN in the template; no
JS build step.

## Endpoints

- `GET /` — the page.
- `GET /api/targets` — `{planet: [scenario, ...], ...}` from
  `list_planets()` + `list_scenarios()`. One call at page load. "flat" is a
  real scenario in the tables and is included.
- `GET /api/observe?planet=&scenario=&n_transits=&R=&seed=` — everything one
  render needs:
  - `wavelength`, `depth`, `depth_err` — binned model + 1σ per bin
  - `points` — `simulate_points(seed)`; noise is realized **server-side**
    so numpy's RNG is the single source of randomness and a given seed
    reproduces the notebook exactly
  - `model_wavelength`, `model_depth` — native-resolution curve
  - `detections` — `.detect()` output (σ per molecule + "atmosphere")
  - `flat_depth`, plus the echoed inputs for the title
  - Defaults: `scenario=clear`, `n_transits=1`, `R=100`, `seed=0`

- `GET /api/lightcurve?planet=&scenario=&n_transits=&R=&seed=&groups=` —
  simulated transit light curves from `terminator.lightcurve` (added
  2026-08-29 by request): `time_hr`, `t14_s`, `cadence_s`, and per band a
  `{label, wl_lo, wl_hi, depth, sigma_point, flux_model, flux}` record.
  `groups=1` (default, cap 12) is white light; `k` splits the spectral bins
  into k contiguous bands. Trapezoid shape (central transit, no limb
  darkening — the tables carry no geometry or LD coefficients); per-point
  noise `sigma_band * sqrt(t14/2c)`, self-consistent with the spectrum's
  error bars.

Validation → HTTP 400 + `{"error": msg}`: unknown planet or scenario,
non-integer or `< 1` `n_transits` (cap 1000), `R` outside [10, 500],
non-integer seed, `groups` outside [1, 12]. `observe()`'s own ValueError
(R beyond the noise tables' native sampling) also maps to a JSON 400 —
never Flask's HTML 500, which the JSON-parsing front end can't render.

## Page behavior

Dropdowns for planet/scenario (scenario list repopulates per planet),
number inputs for `n_transits` and `R`, a "re-roll noise" button (bumps
seed), a show-model checkbox (client-side trace visibility toggle, no
refetch). Any other input change refetches `/api/observe` and redraws.

Plot mirrors `plot_spectrum`: log-x with fixed ticks
[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7] labeled as plain numbers, transit depth
in %, model line + error-bar markers, same title format. Beside it, a table
of detection significances with the standing caveat that overlapping band
masks make per-molecule values approximate.

## Testing

`tests/test_webapp.py`, Flask test client, real repo tables (same as the
end-to-end suite): targets shape; an observe call round-trips against a
direct `terminator.observe()` call (arrays, points, detections); seed
reproducibility; 400s on each invalid input. No browser automation.
