![Logo](docs/logo_schmidt_observatory_system.png)

[![Tests](https://github.com/schmidt-observatories/lazuli-exo/actions/workflows/tests.yml/badge.svg)](https://github.com/schmidt-observatories/lazuli-exo/actions/workflows/tests.yml)

# lazuli-exo
Exoplanet transmission spectroscopy simulator for the Lazuli IFS

lazuli-exo returns the model transmission spectrum of a transiting planet binned
to the Lazuli IFS, with photon/read-noise error bars for a chosen number of
transits, plus a spectral-template S/N. Two ways in:

- **Showcase planets**: 14 well-known planets x 5 atmosphere scenarios from
  the frozen NASA Exoplanet Archive registry, with precomputed slicersim
  noise tables (`lazuli_exo.observe`).
- **Parametric model**: any planet from its star, radius, mass, T_eq, transit
  duration and atmosphere knobs ([M/H], C/O, mu, cloud deck, haze), with
  slicersim computing the noise live (`lazuli_exo.simulate`).

Both paths evaluate the same parametric model (a POSEIDON-derived opacity
grid), so a showcase planet and the same numbers typed by hand give the same
spectrum.

The runtime is numpy-only with its lookup tables shipped as package data.
These are isothermal, equilibrium-chemistry model atmospheres (gray cloud deck, power-law haze) with statistical noise
only: comparison material, not calibrated detection significances. Details,
physics and caveats: [docs/MODEL.md](docs/MODEL.md).

# Installation

## **1. Clone the repository**
```sh
git clone git@github.com:gummiks/lazuli-exo.git
cd lazuli-exo
```

## **2. Install the package**
```sh
pip install -e .
```

Optional extras: `pip install -e ".[plot,web,test,docs]"` (matplotlib for
`.plot()`, Flask for the applet, pytest, Sphinx), and `[transit]` /
`[archive]` / `[pandora]` for the transit models below. The parametric model's live
noise needs `slicersim` and `sncosmo` installed in the same environment; the
showcase tables do not.

# Quickstart

```python
import lazuli_exo

# showcase planet (precomputed noise table)
obs = lazuli_exo.observe("WASP-39 b", scenario="cloudy", n_transits=5, R=100)
obs.wavelength, obs.depth, obs.depth_err      # binned spectrum + 1-sigma per bin
obs.detect()                                  # {"template_snr_diagonal": ..., "band_snr": {...}}
obs.simulate_points(seed=0)                   # one noisy realization
obs.plot()                                    # [plot] extra

# any planet (needs slicersim + sncosmo)
p = lazuli_exo.presets.showcase("WASP-39 b", "clear")   # or build the dict by hand
p.update(T_eq_K=900.0, mu=None, log_P_cloud_bar=-2.0)
obs = lazuli_exo.simulate(p, R=100, n_transits=5)
```

Parameter names for `simulate`: `T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun, teff_K,
logg, feh, mag, band (J/H/Ks/V), mag_system ("vega" default or "ab"), t14_s,
log_met, c_to_o, mu (None = chemistry), log_P_cloud_bar (None = clear),
log_a_haze (None = no haze), haze_slope`.

# Transit light-curve models

`lazuli_exo.transit` (formerly the standalone `lazuli-transit` package) holds
the instrument-agnostic transit models the Lazuli simulators share. The WCC
ETC (`wcc_etc`) consumes them through `relative_flux(time)`.

```python
from lazuli_exo.transit import TransitModel, download_exoplanet_archive

model = TransitModel(per=1.09, rp=0.117, a=3.04, inc=83.3, u=(0.1, 0.3))
flux = model.relative_flux(time_days)                 # [transit]: jaxoplanet

df = download_exoplanet_archive()                     # [archive]: astroquery, cached in ~/.lazuli_exo/
model = TransitModel.from_planet("WASP-12 b", df=df)
```

`MoonTransitModel` (exomoon transits) needs the `[pandora]` extra.

# Tutorial
See the `notebooks/` directory:

| Notebook | Covers |
|---|---|
| `01_getting_started` | `observe -> detect -> plot`, `SimulatedSpectrum` fields, reproducible noise |
| `02_scenarios` | The five scenarios, and why `hazy` can out-score `clear` |
| `03_transits_and_binning` | `n_transits` and `R`, the sqrt(N) scaling, the resolution ceiling |
| `04_light_curves` | White-light and banded transits, limb darkening |
| `05_tables_and_targets` | Table layout, the index, the frozen NEA registry, caveat flags |
| `06_parametric_model` | The zeta grid, the parametric model, validation against POSEIDON |
| `07_custom_planet` | A planet not in the registry from a dozen numbers; the forms/report web path |
| `08_atmosphere_knobs` | Sweeps of mu, metallicity, deck, haze and T_eq; transits needed |
| `09_stars_and_noise` | PHOENIX SEDs, Vega->AB, error bar vs magnitude/band, saturation floor |

# Running the Flask applet

```bash
pip install -e ".[web]"
./run.sh
```

Then open your browser and go to:

    http://127.0.0.1:5000

Pick a showcase planet and scenario, or switch Mode to "Custom planet" to
prefill from a preset and edit the star, planet and atmosphere. This is a
local exploration tool; the public page lives on the Lazuli simulators site.

# Notes
- Tables (~34 MB) live in `src/lazuli_exo/data/tables/` and ship with the
  package; `LAZULI_EXO_TABLES` overrides the location. The POSEIDON
  validation tables stay in `precompute/tables/` and are not runtime data.
- Regenerating tables needs POSEIDON (conda env `poseidon`) and slicersim;
  see [docs/MODEL.md](docs/MODEL.md).
- Tests: `pytest` (the slicersim-dependent tests skip when it is absent).

## Viewing documentation locally

```bash
pip install -e ".[docs]"      # plus a pandoc binary, for nbsphinx
./docs/rundocs.sh             # live-reloading server (sphinx-autobuild) or a one-shot build
```

Then open `docs/sphinx/_build/html/index.html`. The tutorial notebooks are
rendered into the docs with their saved outputs; `notebooks/build_0N.py`
regenerates notebook N.
