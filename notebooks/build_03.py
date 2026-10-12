"""Builder for 03_transits_and_binning.ipynb (run, then execute the notebook):

    ~/anaconda3/envs/py313/bin/python notebooks/build_03.py
    cd notebooks && ~/anaconda3/envs/py313/bin/jupyter nbconvert --to notebook \
        --execute --inplace --ExecutePreprocessor.timeout=600 03_transits_and_binning.ipynb
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
new_markdown_cell(r"""# 3. Transits and binning

Two knobs control the precision of a simulated observation.

- `n_transits`: how many transits are co-added. The noise tables store a
  single-transit depth uncertainty per detector channel; `observe()` scales it
  by 1/√N.
- `R`: the resolving power the model and noise are binned to. Coarser bins
  average more channels, so each bin is more precise and resolves less.

This notebook shows both, then opens the `lazuli_exo.binning` module that
implements them, including the instrument response and the two resolution
limits."""),
new_code_cell(PREAMBLE),
new_markdown_cell(r"""## √N scaling

`depth_err` is the single-transit error divided by √N. Nothing else changes; the
model `depth` is identical."""),
new_code_cell(r'''PLANET = "WASP-39 b"
n_grid = np.array([1, 2, 5, 10, 20, 50])

obs1 = lazuli_exo.observe(PLANET, scenario="clear", n_transits=1, R=100)
errs = []
for n in n_grid:
    o = lazuli_exo.observe(PLANET, scenario="clear", n_transits=int(n), R=100)
    errs.append(np.median(o.depth_err))
    assert np.array_equal(o.depth, obs1.depth), "model must not change with N"

errs = np.array(errs)
ratio = errs[0] / errs

print(f"{'N':>4s}{'median sigma (ppm)':>22s}{'sigma_1/sigma_N':>18s}"
      f"{'sqrt(N)':>10s}")
for n, e, r in zip(n_grid, errs, ratio):
    print(f"{n:4d}{e * 1e6:22.2f}{r:18.4f}{np.sqrt(n):10.4f}")

assert np.allclose(ratio, np.sqrt(n_grid))
print("\nOK: errors scale exactly as 1/sqrt(N).")'''),
new_markdown_cell(r"""The statistic is linear in 1/σ, so the template S/N inherits the same scaling.
This is the clearest illustration of what the model leaves out: there is no
per-transit systematics term and no red-noise floor, so the curve never
flattens."""),
new_code_cell(r'''fig, ax = plt.subplots(figsize=(8.5, 4.5))

for planet in ["WASP-39 b", "GJ 3470 b", "K2-18 b"]:
    snr = [lazuli_exo.observe(planet, scenario="clear", n_transits=int(n))
           .detect()["template_snr_diagonal"] for n in n_grid]
    ax.plot(n_grid, snr, "o-", ms=4, label=planet)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of transits")
ax.set_ylabel("Template S/N")
ax.set_title("Pure sqrt(N): no systematics or red-noise floor")
ax.legend(frameon=False)
plt.show()'''),
new_markdown_cell(r"""## What `R` does to the error bar

`obs.plot_noise(ax=ax)` overlays the per-bin error of several binnings. Coarser
bins combine more channels in inverse variance, so the whole curve drops as the
bins widen; the sawtooth at high `R` is real and comes from bins being whole
numbers of pixels. R = 200 also triggers a `UserWarning` about the instrument
resolving power, which the section on resolution limits below explains; the
warnings are left visible in this notebook."""),
new_code_cell(r'''fig, ax = plt.subplots(figsize=(9, 4.2))
for R in [30, 100, 200]:
    o = lazuli_exo.observe(PLANET, scenario="clear", n_transits=1, R=R)
    o.plot_noise(ax=ax, label=f"R = {R}, {o.wavelength.size} bins")
ax.set_title(f"{PLANET}, 1 transit: per-bin error vs binning")
plt.show()'''),
new_markdown_cell(r"""## Choosing `R`

Coarser bins mean fewer, more precise points. The total template S/N is roughly
preserved as long as the bins still resolve the features, and only degrades once
a bin is wide enough to average a band's peak against its continuum.

Above R ≈ 100 the cells below emit the same `UserWarning`. That is the first of
two resolution limits, explained further down: the requested bins are narrower
than the spectrograph's resolution element, so neighbouring bins are correlated
by the line-spread function and the error bars are no longer independent."""),
new_code_cell(r'''R_grid = [30, 50, 100, 150, 200]

print(f"{'R':>5s}{'bins':>7s}{'median sigma (ppm)':>22s}"
      f"{'template S/N':>15s}")
for R in R_grid:
    o = lazuli_exo.observe(PLANET, scenario="clear", n_transits=1, R=R)
    print(f"{R:5d}{o.wavelength.size:7d}{np.median(o.depth_err) * 1e6:22.2f}"
          f"{o.detect()['template_snr_diagonal']:15.1f}")'''),
new_code_cell(r'''fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)

for ax, R in zip(axes, [30, 100, 200]):
    o = lazuli_exo.observe(PLANET, scenario="clear", n_transits=1, R=R)
    o.plot(ax=ax, seed=0, label=f"R = {R}, {o.wavelength.size} bins")
    ax.legend(frameon=False, loc="upper left")
for ax in axes[:-1]:
    ax.set_xlabel("")
axes[0].set_title(f"{PLANET}, clear, 1 transit: resolution trade")
plt.show()'''),
new_markdown_cell(r"""## The binning module

`observe()` builds its grid with `lazuli_exo.binning`. The edges are geometric,
so every bin has the same fractional width 1 + 1/R, and only complete bins are
kept, so the grid can stop up to one bin short of the requested `wl_max`."""),
new_code_cell(r'''from lazuli_exo import binning

edges = binning.constant_R_edges(0.40, 1.70, 100)
centers = binning.bin_centers(edges)

print(f"{len(centers)} complete bins over [{edges[0]:.4f}, "
      f"{edges[-1]:.4f}] um")
print(f"edge ratio: {edges[1] / edges[0]:.6f}  "
      f"(expected {1 + 1 / 100:.6f})")

frac = np.diff(edges) / centers
print(f"fractional width: constant at {frac.mean():.6f} "
      f"(spread {frac.std():.2e})")
print(f"shortfall below 1.70 um: {(1.70 - edges[-1]) * 1e3:.2f} nm "
      f"(< one bin width of {np.diff(edges)[-1] * 1e3:.2f} nm)")

# bin_centers is the geometric mean of the edges
assert np.allclose(centers, np.sqrt(edges[:-1] * edges[1:]))'''),
new_markdown_cell(r"""## The instrument response

`observe()` does not simply average the model into bins. `bin_observation`
snaps the constant-R edges to detector channel edges, convolves the stellar
photon flux and the in-transit flux with the Gaussian line-spread function
(sigma 1.15 px), integrates per pixel, and then combines the per-channel depths
with the channel 1/σ² weights. `obs.meta["instrument_response"]` records what
was done."""),
new_code_cell(r'''from lazuli_exo import tables

tdir = tables.resolve_tables_dir(None)
wl_n, err_n, meta_n = tables.load_noise(tdir, PLANET)

print(obs1.meta["instrument_response"])
print()
R_inst = binning.R_instr(wl_n)
print(f"{wl_n.size} native noise channels over [{wl_n.min():.4f}, {wl_n.max():.4f}] um")
print(f"instrument resolving power (2.3 px per element): "
      f"{R_inst[0]:.0f} at {wl_n[0]:.2f} um, "
      f"minimum {R_inst.min():.0f} at {wl_n[R_inst.argmin()]:.2f} um, "
      f"{R_inst[-1]:.0f} at {wl_n[-1]:.2f} um")

# the LSF smooths the model: a plain bin average differs from the instrument
# response in the line cores, where the model has structure narrower than a pixel
edges100 = binning.snap_edges_to_channels(
    binning.constant_R_edges(wl_n.min(), wl_n.max(), 100), wl_n)
plain = binning.bin_model(obs1.model_wavelength, obs1.model_depth, edges100)
d = (plain - obs1.depth) * 1e6
i = np.argmax(np.abs(d))
print(f"\nplain bin average minus instrument response: median |diff| {np.median(np.abs(d)):.1f} ppm, "
      f"largest {d[i]:+.0f} ppm at {obs1.wavelength[i]:.3f} um")'''),
new_markdown_cell(r"""## The two resolution limits

Both are set where the dispersion makes the pixels coarsest in resolving
power, which for Lazuli is near 1.0 µm rather than at either end of the band
(the cell above printed the numbers).

1. The instrument resolving power (minimum about 100). Above it the call still
   runs but warns: bins narrower than a resolution element are correlated by
   the line-spread function.
2. The native channel sampling (minimum about 230, one pixel per bin). Above
   it the call raises a `ValueError`."""),
new_code_cell(r'''import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    lazuli_exo.observe(PLANET, scenario="clear", n_transits=1, R=150)
print(f"R=150: {caught[0].category.__name__}: {caught[0].message}\n")

try:
    lazuli_exo.observe(PLANET, scenario="clear", n_transits=1, R=400)
except ValueError as exc:
    print(f"R=400: ValueError: {exc}")
else:
    raise AssertionError("expected a ValueError")'''),
new_markdown_cell(r"""`n_transits` is validated too."""),
new_code_cell(r'''try:
    lazuli_exo.observe(PLANET, n_transits=0)
except ValueError as exc:
    print(f"ValueError: {exc}")
else:
    raise AssertionError("expected a ValueError")'''),
new_markdown_cell(r"""## How many transits to reach a target S/N

A practical use of the √N scaling is to invert it. The statistic is exactly
linear in √N, so the number of transits needed to reach a target follows in
closed form from a single-transit run.

<div class="alert alert-warning">

These are not observing-time forecasts. The statistic is not a calibrated
significance, and the noise model has no systematics floor, so the real number of
transits would be larger by an amount this model cannot tell you.

</div>"""),
new_code_cell(r'''TARGET_SNR = 100.0

rows = []
with warnings.catch_warnings():
    # one bright host warns about residual detector saturation;
    # see notebook 05 for what that means
    warnings.simplefilter("ignore", UserWarning)
    for p in lazuli_exo.list_planets():
        snr1 = lazuli_exo.observe(
            p, scenario="clear",
            n_transits=1).detect()["template_snr_diagonal"]
        rows.append((p, snr1, int(np.ceil((TARGET_SNR / snr1) ** 2))))

rows.sort(key=lambda r: r[2])
print(f"transits for template S/N = {TARGET_SNR:.0f}, clear scenario\n")
print(f"{'planet':14s}{'S/N (1 transit)':>17s}{'N needed':>11s}")
for p, snr1, n in rows:
    print(f"{p:14s}{snr1:17.2f}{n:11d}")'''),
new_code_cell(r'''# verify the closed form against an actual re-run for one target
p, snr1, n = rows[len(rows) // 2]
actual = lazuli_exo.observe(p, scenario="clear",
                            n_transits=n).detect()["template_snr_diagonal"]
print(f"{p}: predicted N={n} -> S/N {snr1 * np.sqrt(n):.3f}, "
      f"actual {actual:.3f}")
assert np.isclose(snr1 * np.sqrt(n), actual)
print("OK: the closed form is exact.")'''),
new_markdown_cell(r"""## Summary

- `n_transits` scales `depth_err` by exactly 1/√N and nothing else; the model
  `depth` is untouched.
- Template S/N therefore grows as √N without limit. The model has no systematics
  or red-noise floor, which is its main unrealism here.
- `R` sets a geometric bin grid snapped to whole pixels; the model goes through
  the line-spread function before binning, and channels combine in inverse
  variance.
- Above the instrument resolving power (about 100, set near 1.0 µm) you get a
  warning; above the pixel sampling (about 230) you get a `ValueError`.

Next: [light curves](04_light_curves.ipynb), the time domain."""),
]
nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
out = Path(__file__).with_name("03_transits_and_binning.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, len(cells), "cells")
