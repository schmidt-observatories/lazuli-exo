"""Builder for 01_getting_started.ipynb (run, then execute the notebook):

    ~/anaconda3/envs/py313/bin/python notebooks/build_01.py
    cd notebooks && ~/anaconda3/envs/py313/bin/jupyter nbconvert --to notebook \
        --execute --inplace --ExecutePreprocessor.timeout=600 01_getting_started.ipynb
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PREAMBLE = r'''import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use("gks")
except OSError:
    pass

import lazuli_exo

print("lazuli_exo", lazuli_exo.__file__)'''

cells = [
new_markdown_cell(r"""# 1. Getting started

`lazuli-exo` simulates transmission spectroscopy with the Lazuli IFS. Give it a
showcase planet, an atmosphere scenario and a number of transits, and it returns
the model transmission spectrum binned to the instrument, the 1-sigma error bar
per bin, and a spectral-template S/N.

`import lazuli_exo` needs numpy and nothing else. The atmosphere is a fast
surrogate for POSEIDON (an interpolated opacity grid plus analytic scale-height,
cloud and haze terms) and the noise comes from precomputed slicersim tables.

<div class="alert alert-warning">

**What the numbers mean.** The instrument half (photon and read noise through the
real throughput, readout and line-spread function) is quantitative. The atmosphere
half is a 1D isothermal, equilibrium-chemistry surrogate with statistical noise
only: no P-T profile, no systematics floor. Use the outputs for rankings and
trade studies. Do not quote the template S/N as a detection significance.
[Known approximations](../user_guide/concepts.rst) lists what the model leaves
out.

</div>"""),
new_code_cell(PREAMBLE),
new_markdown_cell(r"""## What is available

Fourteen showcase targets, each with the same five scenarios. `list_planets()`
reads the table index; `list_scenarios()` returns the preset names."""),
new_code_cell(r'''planets = lazuli_exo.list_planets()
print(f"{len(planets)} showcase targets\n")
for p in planets:
    print(f"  {p:14s} {lazuli_exo.list_scenarios(p)}")'''),
new_markdown_cell(r"""## A single observation

`observe()` takes a planet name, a scenario, the number of transits to co-add
and the resolving power `R` to bin the spectrum to."""),
new_code_cell(r'''obs = lazuli_exo.observe("WASP-39 b", scenario="clear",
                         n_transits=1, R=100)
obs'''),
new_markdown_cell(r"""## Anatomy of a `SimulatedSpectrum`

The result is a plain dataclass. You will use `wavelength`, `depth` and
`depth_err` most; `model_wavelength` and `model_depth` keep the native-resolution
model for overplotting."""),
new_code_cell(r'''for name in ["wavelength", "depth", "depth_err",
             "model_wavelength", "model_depth"]:
    arr = getattr(obs, name)
    print(f"{name:18s} shape={str(arr.shape):8s} "
          f"[{arr.min():.6g}, {arr.max():.6g}]")

print()
print(f"{'flat_depth':18s} {obs.flat_depth:.6g}   "
      "# geometric (Rp/Rs)^2 from the registry radii")
print(f"{'R':18s} {obs.R}")
print(f"{'n_transits':18s} {obs.n_transits}")'''),
new_markdown_cell(r"""The bins are constant resolving power (snapped to whole detector pixels), so
the bin width grows with wavelength. `depth` is the noiseless model seen through
the instrument and averaged into those bins; `depth_err` is the per-bin 1-sigma
uncertainty."""),
new_code_cell(r'''dlam = np.diff(obs.wavelength)
print(f"{obs.wavelength.size} bins from {obs.wavelength[0]:.3f} "
      f"to {obs.wavelength[-1]:.3f} um")
print(f"bin width: {dlam[0] * 1e3:.2f} nm at the blue end, "
      f"{dlam[-1] * 1e3:.2f} nm at the red end")
print(f"R implied at mid-band: "
      f"{obs.wavelength[len(dlam) // 2] / dlam[len(dlam) // 2]:.0f}")

print(f"\ndepth_err: median {np.median(obs.depth_err) * 1e6:.0f} ppm, "
      f"best {obs.depth_err.min() * 1e6:.0f} ppm, "
      f"worst {obs.depth_err.max() * 1e6:.0f} ppm")'''),
new_markdown_cell(r"""## Plotting

`obs.plot()` draws the native-resolution model with one simulated noisy
realization at the binned resolution. It needs the `[plot]` extra (matplotlib).
Every plot method takes an optional `ax` and returns the Axes, so you can pass
your own subplot or overlay several calls; notebook 2 does that."""),
new_code_cell(r'''obs.plot(seed=0)
plt.show()'''),
new_markdown_cell(r"""`obs.plot_noise()` shows where the instrument is precise: the per-bin 1-sigma
error against wavelength, with `n_transits` already included."""),
new_code_cell(r'''obs.plot_noise()
plt.show()'''),
new_markdown_cell(r"""## Noise realizations

`depth` and `depth_err` are deterministic. The noise is drawn on demand by
`simulate_points(seed)`, and a seed makes a realization reproducible, which is how
the web demo and these notebooks stay stable across runs."""),
new_code_cell(r'''a = obs.simulate_points(seed=0)
b = obs.simulate_points(seed=0)
c = obs.simulate_points(seed=1)

assert np.array_equal(a, b), "same seed must give the same realization"
assert not np.array_equal(a, c), "different seeds must differ"

resid = (a - obs.depth) / obs.depth_err
print(f"same seed reproduces:      {np.array_equal(a, b)}")
print(f"different seed differs:    {not np.array_equal(a, c)}")
print(f"\nresiduals in sigma units: mean {resid.mean():+.3f}, "
      f"std {resid.std():.3f}  (expect ~0 and ~1)")'''),
new_markdown_cell(r"""## Detection diagnostics

`obs.detect()` returns a matched-template amplitude S/N of the noiseless model
against the best-fit flat line, under a diagonal covariance.

<div class="alert alert-warning">

`template_snr_diagonal` is a comparison statistic, not a calibrated detection
significance: there is no likelihood ratio, no nuisance model, no covariance and
no false-positive calibration. The per-band values are the same statistic
restricted to fixed wavelength masks. They mix every species in the mask with the
haze, cloud and continuum structure, and the masks overlap, so they are not
molecule detections either.

</div>"""),
new_code_cell(r'''result = obs.detect()

print(f"template_snr_diagonal: {result['template_snr_diagonal']:.1f}\n")
print("band_snr (band-restricted template norms, NOT detections):")
for mol, snr in sorted(result["band_snr"].items(),
                       key=lambda kv: -kv[1]):
    print(f"  {mol:5s} {snr:7.1f}")'''),
new_markdown_cell(r"""The band masks live in `lazuli_exo.detect.BAND_MASKS`. H2O and CH4 both cover
1.10 to 1.20 µm, which is one reason these are not per-molecule detections."""),
new_code_cell(r'''from lazuli_exo.detect import BAND_MASKS

for mol, spans in BAND_MASKS.items():
    pretty = ", ".join(f"{lo:.3f}-{hi:.3f}" for lo, hi in spans)
    print(f"  {mol:5s} {pretty} um")'''),
new_markdown_cell(r"""## Where the numbers came from

Every observation carries its provenance in `obs.meta`: the parameters the model
was evaluated at (`params`), what the atmosphere model derived from them
(`atmosphere`), the slicersim settings behind the error bars (`noise`) and how
the model was put through the spectrograph (`instrument_response`). This is the
first place to look when a number surprises you."""),
new_code_cell(r'''SHOW = {
    "params": ["T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "log_met", "c_to_o",
               "mu", "log_P_cloud_bar", "log_a_haze"],
    "atmosphere": ["g_ms2", "H_m", "H_over_Rp", "mu_used", "mu_source",
                   "T_chem_K", "T_clamped", "feature_amplitude_ppm"],
    "noise": ["instrument", "nmd_used", "t14_s", "t_obs_s", "mag_input", "mag_ab",
              "fully_desaturated", "nsaturated_detpx", "slicersim_version",
              "git_commit"],
    "instrument_response": None,          # all keys
}
for section, keys in SHOW.items():
    d = obs.meta[section]
    print(f"meta[{section!r}]:")
    for k in keys or d:
        v = d[k]
        print(f"  {k:22s} {v:.6g}" if isinstance(v, float) else f"  {k:22s} {v}")
    print()'''),
new_markdown_cell(r"""## Summary

- `lazuli_exo.observe(planet, scenario, n_transits, R)` returns a
  `SimulatedSpectrum`.
- `.depth` and `.depth_err` are the noiseless model and its 1-sigma errors;
  `.simulate_points(seed)` draws a reproducible noisy realization.
- `.plot()`, `.plot_noise()` and (notebook 4) `.light_curves().plot()` all take
  `ax=` and return the Axes.
- `.detect()` gives a template S/N: a comparison statistic, not a significance.
- `.meta` records the parameters, the derived atmosphere quantities and the noise
  settings that produced the numbers.

Next: [scenarios](02_scenarios.ipynb), what the five atmosphere scenarios mean
and how they differ."""),
]
nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
out = Path(__file__).with_name("01_getting_started.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, len(cells), "cells")
