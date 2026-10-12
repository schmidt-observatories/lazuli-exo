"""Builder for 05_tables_and_targets.ipynb (run, then execute the notebook):

    ~/anaconda3/envs/py313/bin/python notebooks/build_05.py
    cd notebooks && ~/anaconda3/envs/py313/bin/jupyter nbconvert --to notebook \
        --execute --inplace --ExecutePreprocessor.timeout=600 05_tables_and_targets.ipynb
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

print("lazuli_exo", lazuli_exo.__version__ if hasattr(lazuli_exo, "__version__") else "(dev)")'''

cells = [
new_markdown_cell(r"""# 5. Tables and targets

Everything `lazuli-exo` computes at runtime comes out of a small directory of
`.npz` lookup tables plus a frozen target registry, and nothing heavier than
numpy is imported. This notebook opens the tables directly, so you can see what
the runtime reads and where its caveats are recorded.

```
index.json                 # schema version, planets, grids, sha256 manifest
atmo/zeta_grid.npz         # parametric model: zeta(T, [M/H], C/O, fiducial, lambda), mu
star/phoenix_grid.npz      # PHOENIX SEDs on (T_eff, log g, [Fe/H])
star/<planet-slug>.npz     # wavelength_um, f_lambda, meta      (R = 10 000 host SED)
noise/<planet-slug>.npz    # wavelength_um, depth_err_1transit, meta
ld/<planet-slug>.npz       # wavelength_um, u1, u2, meta
```

Scenarios are not files: they are presets of the parametric model. The
per-planet POSEIDON spectra the model is validated against live under
`precompute/tables/` and are not shipped."""),
new_code_cell(PREAMBLE),
new_markdown_cell(r"""## Locating the tables

`resolve_tables_dir()` looks in three places, in order: an explicit
`tables_dir=` argument, then the `LAZULI_EXO_TABLES` environment variable, then
the `data/tables/` shipped inside the package."""),
new_code_cell(r'''from pathlib import Path
from lazuli_exo import tables

tdir = tables.resolve_tables_dir(None)
PKG = Path(lazuli_exo.__file__).parent
print(f"tables dir: <package>/{tdir.relative_to(PKG)}\n")

for sub in ["atmo", "star", "noise", "ld"]:
    files = sorted((tdir / sub).glob("*.npz"))
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"  {sub:8s} {len(files):3d} files  {mb:6.2f} MB")

total = sum(f.stat().st_size for f in tdir.rglob("*")
            if f.is_file()) / 1e6
print(f"\n  {'total':8s} {total:16.2f} MB")'''),
new_markdown_cell(r"""## The index

`index.json` is the catalogue. It carries a schema `version`, and `load_index()`
refuses a mismatch rather than silently misreading tables. Since schema 4 it
also holds a sha256 manifest of every file, so a table that was edited or
truncated after generation can be caught."""),
new_code_cell(r'''index = tables.load_index(tdir)

print(f"schema version: {index['version']} "
      f"(supported: {tables.SCHEMA_VERSION})")
print(f"planets:        {len(index['planets'])}")
print(f"grids:          {index['atmo']}")
print(f"manifest:       {len(index['files'])} files\n")

print("index['planets']['WASP-39 b']:", index["planets"]["WASP-39 b"])

rel = "noise/wasp-39-b.npz"
print(f"\n{rel}: manifest {index['files'][rel][:16]}..., "
      f"on disk {tables.sha256_file(tdir / rel)[:16]}...")
assert tables.sha256_file(tdir / rel) == index["files"][rel]'''),
new_markdown_cell(r"""The index is regenerated from what is on disk by `tables.rebuild_index()`,
which every precompute script calls when it finishes, so the catalogue cannot
drift from the files."""),
new_code_cell(r'''missing = [rel for rel in index["files"] if not (tdir / rel).exists()]
print(f"indexed files missing from disk: {missing or 'none'}")
assert not missing

bad = [rel for rel in index["files"]
       if tables.sha256_file(tdir / rel) != index["files"][rel]]
print(f"files whose hash differs from the manifest: {bad or 'none'}")
assert not bad'''),
new_markdown_cell(r"""## What each table holds

Each per-planet loader returns its arrays plus the `meta` dict that records how
the table was produced; the grid loaders return a dict. The model itself is not
a table any more: `observe()` evaluates the parametric model on the ζ grid's
wavelengths."""),
new_code_cell(r'''PLANET = "WASP-39 b"

obs_m = lazuli_exo.observe(PLANET, scenario="clear", tables_dir=tdir)
wl_m, depth_m = obs_m.model_wavelength, obs_m.model_depth
wl_s, flam_s, meta_s = tables.load_star(tdir, PLANET)
wl_n, err_n, meta_n = tables.load_noise(tdir, PLANET)
wl_l, u1_l, u2_l, meta_l = tables.load_ld(tdir, PLANET)
zeta = tables.load_zeta_grid(tdir)
phoenix = tables.load_phoenix_grid(tdir)

print(f"{'array':8s}{'samples':>9s}{'wavelength range (um)':>26s}")
for lab, w in [("model", wl_m), ("star", wl_s),
               ("noise", wl_n), ("ld", wl_l)]:
    print(f"{lab:8s}{w.size:9d}{w.min():14.4f} - {w.max():.4f}")

print(f"\nzeta grid: zeta{zeta['zeta'].shape} on T x [M/H] x C/O x fiducial x lambda, "
      f"{zeta['wavelength_um'].size} wavelengths")
print(f"PHOENIX:   log_flam{phoenix['log_flam'].shape} on T_eff x log g x [Fe/H] x lambda, "
      f"T_eff {phoenix['teff_K'].min():.0f}-{phoenix['teff_K'].max():.0f} K")'''),
new_markdown_cell(r"""The ζ grid spans 0.35 to 1.85 µm at R = 10 000 (margin around the 0.40 to
1.70 µm IFS bandpass); `obs.model_wavelength` is the slice of it inside the
binned range, which is why the model row above starts at 0.40 µm. The noise
table is sampled at the instrument's own channel spacing. That difference is why `observe()` puts the
model through the instrument response onto the channels rather than
interpolating one grid onto the other."""),
new_code_cell(r'''fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)

axes[0].plot(wl_m, depth_m * 100.0, lw=0.7)
axes[0].set_ylabel("Transit depth (%)")
axes[0].set_title(f"{PLANET}: model (clear), star table and noise table")

axes[1].plot(wl_s, flam_s / flam_s.max(), lw=0.7)
axes[1].set_ylabel("Stellar $F_\\lambda$ (normalized)")

axes[2].plot(wl_n, err_n * 1e6, lw=1.0)
axes[2].set_ylabel("1-transit $\\sigma_{depth}$ (ppm)")
axes[2].set_yscale("log")
axes[2].set_xlabel("Wavelength (µm)")

plt.show()

print(f"star grid: {meta_s['stellar_grid']}, "
      f"T_eff = {meta_s['T_eff_K']} K, units = {meta_s['units']}")'''),
new_markdown_cell(r"""## What the noise table records

The noise `meta` is the provenance of every error bar: the instrument, the
normalization band and the Vega to AB conversion, the readout actually used, the
assumption the depth uncertainty rests on, and the package version, git commit
and input hashes the table was generated from."""),
new_code_cell(r'''for k, v in meta_n.items():
    print(f"  {k:28s} {v}")'''),
new_markdown_cell(r"""## Noise vs. host brightness

Single-transit depth precision is driven mostly by how bright the host is.
Plotting the median against the registry's J magnitude shows the photon-limited
trend, with the sub-Neptune and terrestrial hosts at the faint end."""),
new_code_cell(r'''from lazuli_exo.targets import get_target

rows = []
for p in lazuli_exo.list_planets():
    _, e, _ = tables.load_noise(tdir, p)
    rows.append((p, get_target(p)["J_mag"], np.median(e)))

fig, ax = plt.subplots(figsize=(9, 5))
for p, j, e in rows:
    ax.plot(j, e * 1e6, "o", ms=6)
    ax.annotate(p, (j, e * 1e6), fontsize=8,
                xytext=(4, 3), textcoords="offset points")

ax.set_yscale("log")
ax.set_xlabel("Host J magnitude")
ax.set_ylabel("Median 1-transit $\\sigma_{depth}$ (ppm)")
ax.set_title("Single-transit depth precision vs. host brightness")
plt.show()'''),
new_markdown_cell(r"""## The target registry

`lazuli_exo.targets` is a frozen snapshot of the NASA Exoplanet Archive
`PSCompPars` table (plus a few overrides), generated by
`precompute/freeze_targets.py`. It is a plain dict: no network access at runtime
and no dependency on the archive being up."""),
new_code_cell(r'''fields = ["T_eq_K", "R_p_RJ", "R_s_RSun", "T_eff_K", "J_mag",
          "period_d", "a_over_rs", "inc_deg", "depth"]

print(f"{'planet':14s}" + "".join(f"{f:>11s}" for f in fields))
for p in lazuli_exo.list_planets():
    t = get_target(p)
    print(f"{p:14s}" + "".join(f"{t[f]:11.4g}" for f in fields))'''),
new_code_cell(r'''t = get_target("K2-18 b")
print("full registry entry for K2-18 b:\n")
for k, v in sorted(t.items()):
    print(f"  {k:12s} {v}")'''),
new_markdown_cell(r"""Unknown names fail loudly, listing what is available. `observe()` behaves the
same way for an unknown planet or scenario."""),
new_code_cell(r'''for call, label in [
    (lambda: get_target("Kepler-999 b"), "get_target"),
    (lambda: lazuli_exo.observe("Kepler-999 b", "clear", tables_dir=tdir), "observe"),
    (lambda: lazuli_exo.observe("WASP-39 b", "stormy", tables_dir=tdir), "scenario"),
]:
    try:
        call()
    except KeyError as exc:
        print(f"{label:12s} KeyError: {str(exc)[:88]}...")
    else:
        raise AssertionError(f"{label} should have raised")'''),
new_markdown_cell(r"""## The saturation caveat

The brightest hosts push the detector toward saturation. slicersim shortens the
readout until the ramp is clean, and when even the shortest ramp (2 groups x 1
frame) leaves saturated pixels at the last read, the residual count is recorded
in the noise table and `observe()` raises a `UserWarning`. With the current
tables one host is affected.

<div class="alert alert-warning">

For a flagged target the uncertainties are unvalidated: saturation can bias
extraction weights, nonlinearity and wavelength dependence in either direction.
Treat its forecasts as indicative only. Check the flag in the data rather than
remembering a name; the list changes when the tables are regenerated.

</div>"""),
new_code_cell(r'''print(f"{'planet':14s}{'desaturated':>13s}{'saturated px':>14s}"
      f"{'nmd used':>10s}")
flagged = []
for p in lazuli_exo.list_planets():
    _, _, m = tables.load_noise(tdir, p)
    ok = m.get("fully_desaturated")
    if ok is False:
        flagged.append(p)
    print(f"{p:14s}{str(ok):>13s}{m.get('nsaturated_detpx', 0):14d}"
          f"{str(m.get('nmd_used')):>10s}")

print(f"\nflagged: {flagged}")'''),
new_code_cell(r'''import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    lazuli_exo.observe(flagged[0], scenario="clear", n_transits=1)

for w in caught:
    print(f"{w.category.__name__}: {w.message}")

assert caught and caught[0].category is UserWarning'''),
new_markdown_cell(r"""## Pointing at a different tables directory

For a deployment that keeps the tables elsewhere, set `LAZULI_EXO_TABLES` or
pass `tables_dir=` per call. Both take precedence over the packaged default, in
that order."""),
new_code_cell(r'''import os

print(f"default (package-relative): <package>/"
      f"{tables.resolve_tables_dir(None).relative_to(PKG)}")
print(f"explicit argument:          "
      f"{tables.resolve_tables_dir('/some/other/tables')}")

os.environ["LAZULI_EXO_TABLES"] = "/from/the/environment"
print(f"from $LAZULI_EXO_TABLES:    {tables.resolve_tables_dir(None)}")
del os.environ["LAZULI_EXO_TABLES"]

# an explicit argument still wins over the environment variable
os.environ["LAZULI_EXO_TABLES"] = "/from/the/environment"
print(f"argument beats env:         "
      f"{tables.resolve_tables_dir('/explicit/wins')}")
del os.environ["LAZULI_EXO_TABLES"]

# observe() takes the same argument
obs = lazuli_exo.observe(PLANET, scenario="clear", n_transits=1,
                         tables_dir=tdir)
print(f"\nobs.meta['tables_dir']:     <package>/"
      f"{Path(obs.meta['tables_dir']).relative_to(PKG)}")'''),
new_markdown_cell(r"""## Summary

- Per-planet star, noise and limb-darkening tables plus two grids (ζ and
  PHOENIX) under one directory, catalogued by a version-stamped `index.json`
  with a sha256 manifest that `rebuild_index()` regenerates from disk.
- Every table carries a `meta` dict recording how it was produced: stellar
  grid, readout mode, LD grid point, package version and git commit.
- The registry (`lazuli_exo.targets`) is a frozen NEA snapshot; no network at
  runtime.
- The caveats live in the data: `T_clamped` in `meta["atmosphere"]` on cold
  planets, `fully_desaturated` in the noise table on bright hosts (which also
  raises a `UserWarning`).
- `tables_dir=` beats `$LAZULI_EXO_TABLES` beats the packaged default.

That covers the showcase tables. Notebooks 6 to 9 cover the parametric model:
any planet from a dozen numbers, with the noise computed live."""),
]
nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
out = Path(__file__).with_name("05_tables_and_targets.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, len(cells), "cells")
