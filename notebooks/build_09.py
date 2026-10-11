"""Builder for 09_stars_and_noise.ipynb."""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = [
new_markdown_cell(r"""# 9. Stars and noise

The noise side of the parametric model: a PHOENIX stellar SED from
(T_eff, log g, [Fe/H]), normalised to a magnitude, run through slicersim to
get the per-channel depth uncertainty. This notebook shows the SED grid, the
Vega→AB conversion, how the error bar scales with brightness and band, the
bright-host saturation floor, and the CBE vs EOL instrument configurations.
`lazuli_exo.slicer.depth_error` imports slicersim lazily, so it must be
installed to run this notebook."""),
new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
try:
    plt.style.use("gks")
except OSError:
    pass

from lazuli_exo import star, slicer, tables

g = tables.load_phoenix_grid()
print("PHOENIX grid:", g["log_flam"].shape, "| T_eff", g["teff_K"].min(), "-", g["teff_K"].max(), "K",
      "| log g", list(g["logg"]), "| [Fe/H]", list(g["feh"]))
print("native resolution per folder:", g["meta"]["native_R"])"""),
new_markdown_cell("## Stellar SEDs\n\nTrilinear interpolation in (log T_eff, log g, [Fe/H]). The shipped grid is solar metallicity only (2000–12000 K, log g 3.5–5.5; issue #34): out-of-range T_eff and log g and any non-zero [Fe/H] clamp to the nearest node and are reported in `info[\"clamped\"]`. Above 12000 K raises."),
new_code_cell("""fig, ax = plt.subplots(figsize=(8, 4))
for teff in (2566, 3400, 4300, 5778, 7000):
    wl_A, f, info = star.phoenix_sed(teff, 4.5, 0.0)
    ax.plot(wl_A / 1e4, f / f[(wl_A > 12000) & (wl_A < 13000)].mean(), lw=1, label=f"{teff} K ({info['grid']})")
ax.set_xlim(0.3, 2.5); ax.set_yscale("log"); ax.set_ylim(1e-2, 30)
ax.set_xlabel("wavelength [µm]"); ax.set_ylabel("F_lambda, normalised at 1.25 µm"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
for teff, logg, feh in ((1800, 5.0, 0.0), (3000, 5.3, 0.6)):
    print((teff, logg, feh), "->", star.phoenix_sed(teff, logg, feh)[2])"""),
new_markdown_cell("## Vega → AB\n\nslicersim normalises magnitudes in the AB system. Archive magnitudes (2MASS J/H/Ks, Johnson V) are Vega. The offsets come from sncosmo's band definitions and are independent of the spectrum; `depth_error` applies them for `mag_system=\"vega\"` and reports the offset in `info`."),
new_code_cell("""for band in ("V", "J", "H", "Ks"):
    print(f"{band:2s}: m_AB - m_Vega = {slicer.vega_to_ab_offset(band):+.3f}")"""),
new_markdown_cell("## Error bar vs brightness\n\nFor a Sun-like star and a 3 h transit: the median single-transit depth error against J magnitude. Faint stars are photon-limited (σ ∝ 10^(0.2 J)); bright stars hit slicersim's shortest readout (2 groups × 1 frame) and residual saturation is reported rather than hidden."),
new_code_cell("""wl_A, flam, _ = star.phoenix_sed(5778, 4.44, 0.0)
mags = np.arange(5.0, 13.1, 1.0)
rows = []
for J in mags:
    wl, err, info = slicer.depth_error(wl_A, flam, mag=J, band="J", mag_system="vega", t14_s=3 * 3600)
    rows.append((J, np.median(err) * 1e6, info["nmd_used"], info["nsaturated_detpx"], info["t_obs_s"]))
    print(f"J = {J:4.1f}  median sigma = {rows[-1][1]:7.0f} ppm  nmd = {info['nmd_used']}  saturated px = {info['nsaturated_detpx']}")
fig, ax = plt.subplots(figsize=(6, 3.8))
r = np.array([(x[0], x[1], x[3]) for x in rows])
ax.plot(r[:, 0], r[:, 1], "o-", ms=4)
ax.plot(r[r[:, 2] > 0, 0], r[r[:, 2] > 0, 1], "rx", ms=9, label="residual saturation")
ax.plot(mags, rows[-1][1] * 10 ** (0.2 * (mags - mags[-1])), "0.6", lw=0.8, label="photon-limited slope")
ax.set_yscale("log"); ax.set_xlabel("J (Vega)"); ax.set_ylabel("median depth error, 1 transit [ppm]"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout()"""),
new_markdown_cell("## Which band to quote the magnitude in\n\nThe same star given in J, H, Ks or V. A consistent SED gives nearly the same answer whichever band is used, since slicersim renormalises the whole spectrum; small differences come from the band offsets and the SED shape."),
new_code_cell("""# a 5778 K star with J = 10 has roughly V = 11.1, H = 9.7, Ks = 9.65 (Vega); use the SED itself to be consistent
import sncosmo
spec = sncosmo.Spectrum(wl_A, flam)
J = 10.0
for band in ("V", "J", "H", "Ks"):
    m = J + (spec.bandmag(slicer.BANDS[band], "vega") - spec.bandmag(slicer.BANDS["J"], "vega"))
    _, err, info = slicer.depth_error(wl_A, flam, mag=m, band=band, mag_system="vega", t14_s=3 * 3600)
    print(f"given {band:2s} = {m:6.2f} Vega -> {info['mag_ab']:6.2f} AB: median sigma {np.median(err)*1e6:5.0f} ppm")"""),
new_markdown_cell("## CBE vs EOL\n\nslicersim ships a current-best-estimate (`lazuli_cbe`) and an end-of-life (`lazuli_eol`) Lazuli configuration. `depth_error` turns any configuration problem into a `ValueError` naming the instrument, so a web page can show it instead of failing; the cell below either plots the EOL/CBE error ratio or prints why EOL is unavailable in the installed slicersim."),
new_code_cell("""wl_c, err_c, _ = slicer.depth_error(wl_A, flam, mag=10.0, band="J", t14_s=3 * 3600, instrument="lazuli_cbe")
try:
    wl_e, err_e, _ = slicer.depth_error(wl_A, flam, mag=10.0, band="J", t14_s=3 * 3600, instrument="lazuli_eol")
except ValueError as e:
    print("EOL unavailable:", e)
else:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(wl_c, err_c * 1e6, lw=0.8, label="CBE"); ax[0].plot(wl_e, err_e * 1e6, lw=0.8, label="EOL")
    ax[0].set_yscale("log"); ax[0].set_xlabel("wavelength [µm]"); ax[0].set_ylabel("depth error per channel [ppm]"); ax[0].legend(frameon=False)
    common = np.interp(wl_c, wl_e, err_e)
    ax[1].plot(wl_c, common / err_c, lw=0.8); ax[1].set_xlabel("wavelength [µm]"); ax[1].set_ylabel("EOL / CBE error")
    fig.tight_layout()
    print(f"median EOL/CBE error ratio: {np.median(common / err_c):.2f}")"""),
new_markdown_cell("## Summary\n\n- The PHOENIX mini-grid covers 2000–12000 K at solar metallicity; outside it the code clamps and tells you.\n- Vega→AB is +0.90 mag in J; passing an archive J without it would make every error bar ~1.4× too optimistic.\n- Bright hosts (J ≲ 7) reach the readout floor and carry a residual-saturation flag; treat those error bars as indicative."),
]
nb = new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
with open(Path(__file__).with_name("09_stars_and_noise.ipynb"), "w") as f:
    nbf.write(nb, f)
print("wrote 09")
