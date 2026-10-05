"""All memo figures, from the check results and data snapshot (memo/figures/).

Design rules (from the 2026-10-05 internal review): gray = measured data, teal
= Lazuli prediction, everywhere; one message per plot; measured error bars and
Lazuli's predicted single-transit error at the same bins side by side; the
clear and cloudy toy scenarios drawn as the two brackets; no native-resolution
model spaghetti; panels ordered so the message groups.
"""
import numpy as np

from common import (A_HST_M2, A_JWST_M2, A_LAZULI_M2, PLANET_NAMES, RESULTS, TARGETS, fnum,
                    gks_style, host, read_csv, savefig)
from lazuli_exo import binning
from checkA_sed import EDGES_NM, xp_spectrum
from checkB_photon import _bin_edges, lazuli_sigma_for_bins
from checkD_spectra import all_datasets, bin_to, fits, models

TEAL, RED, AMBER, BLUE, NAVY, GREY = "#00798c", "#d1495b", "#edae49", "#30638e", "#003d5b", "0.3"


# ---------------------------------------------------------------- check A
def figure_a():
    """Model / Gaia XP in the 50 nm bins of Table A, both stellar models, coolest host first."""
    plt = gks_style()
    rows = {(r["planet"], r["model"]): r for r in read_csv(RESULTS / "checkA_sed.csv")}
    planets = sorted([p for p in PLANET_NAMES if xp_spectrum(p) is not None], key=lambda p: TARGETS[p]["T_eff_K"])
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.8), sharex=True, sharey=True)
    centers = 0.5 * (EDGES_NM[:-1] + EDGES_NM[1:]) / 1e3
    handles = None
    for ax, planet in zip(axes.flat, planets):
        wx, fx, ex = xp_spectrum(planet)
        hs = []
        for model, color, label, mk in (("table", BLUE, "star table (POSEIDON PHOENIX; feeds the noise tables)", "o"),
                                        ("phoenix_grid", RED, "PHOENIX mini-grid (parametric path)", "s")):
            r = rows[(planet, model)]
            ratio = np.array([fnum(r.get(f"r{lo}")) for lo in EDGES_NM[:-1]])
            h, = ax.plot(centers, ratio, mk + "-", ms=4, lw=1.2, color=color, label=label)
            hs.append(h)
            over = ratio > 1.98
            for cx, rv in zip(centers[over], ratio[over]):
                ax.annotate(f"{rv:.1f}", (cx, 1.95), ha="center", va="top", fontsize=7, color=color)
        handles = handles or hs
        ax.axhspan(0.9, 1.1, color="0.88", lw=0, zorder=0)
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.set_title(f"{host(planet)}  ({TARGETS[planet]['T_eff_K']:.0f} K)", fontsize=10)
        ax.set_ylim(0.5, 2.0)
        ax.set_xlim(0.38, 1.05)
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength (µm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("model / Gaia XP")
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Check A: J-normalized stellar model over the measured Gaia DR3 XP flux, 50 nm bins, hosts ordered by T$_{\\rm eff}$ (gray band: ±10 %)",
                 y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, "checkA_sed_vs_xp")


# ---------------------------------------------------------------- check B(ii)
SHORT_INST = {"Wide Field": "WFC3", "Space Tele": "STIS", "Near Infrared Imager": "NIRISS", "Near Infrared Spectrog": "PRISM",
              "Advanced C": "ACS", "Near Infrared Camera a": "NICMOS"}


def short_inst(i):
    return next((v for k, v in SHORT_INST.items() if i.startswith(k)), i[:6])


def figure_b():
    """Lazuli predicted single-transit error over the achieved error, per dataset (horizontal dot plot)."""
    plt = gks_style()
    rows = read_csv(RESULTS / "checkB_achieved_precision.csv")
    rows.sort(key=lambda r: (r["facility"] != "HST", fnum(r["ratio_lazuli_over_achieved"])))   # HST at bottom, JWST on top
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 8.5))
    for fac, kw in (("JWST", dict(mfc=NAVY, mec=NAVY)), ("HST", dict(mfc="white", mec="0.35"))):
        m = np.array([r["facility"] == fac for r in rows])
        ax.plot([fnum(r["ratio_lazuli_over_achieved"]) for r, k in zip(rows, m) if k], y[m], "o", ms=7, mew=1.3,
                label=f"{fac} dataset" + (" (indicative: N transits not in the archive)" if fac == "HST" else " (single transit each)"), **kw)
    r_j, r_h = np.sqrt(A_JWST_M2 / A_LAZULI_M2), np.sqrt(A_HST_M2 / A_LAZULI_M2)
    ax.axvline(r_j, color=NAVY, ls="--", lw=1.2, label=f"area-only expectation vs JWST: {r_j:.2f}")
    ax.axvline(r_h, color="0.35", ls="--", lw=1.2, label=f"area-only expectation vs HST: {r_h:.2f}")
    ax.axvline(1.0, color="0.6", ls=":", lw=1)
    ax.text(1.0, len(rows) - 0.3, "equal precision", rotation=90, va="top", ha="right", fontsize=8, color="0.45")
    for yi, r in zip(y, rows):
        ratio, line = fnum(r["ratio_lazuli_over_achieved"]), (r_j if r["facility"] == "JWST" else r_h)
        if ratio > line:
            ax.annotate(f"{ratio:.2f}", (ratio, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['planet']}  {r['reference'].split(' et al')[0]} {r['pubdate'] if 'pubdate' in r else ''}{short_inst(r['instrument'])}" for r in rows], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Lazuli predicted σ (1 transit) / published σ, same bins")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Check B(ii): predicted Lazuli per-bin precision against published HST and JWST precision", fontsize=11)
    fig.tight_layout()
    savefig(fig, "checkB_precision_ratio")


# ---------------------------------------------------------------- check D
# (planet, reference substring, instrument substring, label, N transits, display R, text-box corner)
PANELS = [("WASP-39 b", "Feinstein", "Near Infrared Imager", "JWST NIRISS/SOSS", 1, 100, "lower left"),
          ("WASP-39 b", "Rustamkulov", "Near Infrared Spectrog", "JWST NIRSpec/PRISM", 1, None, "upper right"),
          ("HD 189733 b", "McCullough", "Wide Field", "HST WFC3", None, None, "lower right"),
          ("HAT-P-11 b", "Fraine", "Wide Field", "HST WFC3", None, None, "lower left"),
          ("K2-18 b", "Benneke", "Wide Field", "HST WFC3", None, None, "lower left"),
          ("WASP-107 b", "Spake", "Wide Field", "HST WFC3", None, None, "upper left")]
CORNER = {"lower left": (0.02, 0.03, "bottom", "left"), "lower right": (0.98, 0.03, "bottom", "right"),
          "upper left": (0.02, 0.97, "top", "left"), "upper right": (0.98, 0.97, "top", "right")}


def rebin(wl, bw, d, s, R):
    """Inverse-variance rebinning of a published spectrum to constant R (display only)."""
    edges = binning.constant_R_edges(wl.min() * 0.999, wl.max() * 1.001, R)
    idx = np.digitize(wl, edges) - 1
    out = []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.any():
            w = 1 / s[m] ** 2
            out.append((np.sum(w * wl[m]) / w.sum(), edges[i + 1] - edges[i], np.sum(w * d[m]) / w.sum(), 1 / np.sqrt(w.sum())))
    return map(np.array, zip(*out))


def figure_d():
    plt = gks_style()
    data = {(p, r, i): (wl, bw, d, s) for (p, r, i, k), wl, bw, d, s in all_datasets()}
    summary = {(r["planet"], r["reference"], r["instrument"]): r for r in read_csv(RESULTS / "checkD_spectra_summary.csv")}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    handles = None
    for ax, (planet, ref_sub, inst_sub, inst_label, n_tr, R_disp, corner) in zip(axes.flat, PANELS):
        key = next(k for k in data if k[0] == planet and ref_sub in k[1] and inst_sub in k[2])
        wl, bw, d, s = data[key]
        srow = summary[(planet, key[1], key[2][:20])] if (planet, key[1], key[2][:20]) in summary else \
            next(v for k, v in summary.items() if k[0] == planet and k[1] == key[1] and key[2].startswith(k[2][:10]))
        if R_disp:
            wl, bw, d, s = rebin(wl, bw, d, s, R_disp)
        lo, hi = _bin_edges(wl, bw)
        sig_laz = lazuli_sigma_for_bins(planet, lo, hi)
        ref = float(np.average(d, weights=1 / s ** 2))
        ppm = lambda v: (v - ref) * 1e6
        lines = {}
        for scen in ("clear", "cloudy"):
            wl_m, dt, dp = models(planet, scen)
            m = bin_to(wl_m, dt, lo, hi)
            c, *_ = fits(d, s, m)
            lines[scen] = ppm(m + c)
        h_band = ax.fill_between(wl, ppm(d) - sig_laz * 1e6, ppm(d) + sig_laz * 1e6, step=None, color=TEAL, alpha=0.25, lw=0,
                                 label="Lazuli predicted ±1σ, 1 transit, same bins (band around the data)")
        h_fill = ax.fill_between(wl, lines["clear"], lines["cloudy"], color=AMBER, alpha=0.12, lw=0,
                                 label="range spanned by the clear and cloudy toy scenarios")
        h_clear, = ax.plot(wl, lines["clear"], "-", lw=1.2, color=AMBER, label="clear H$_2$/He toy scenario (offset-fitted)")
        h_cloudy, = ax.plot(wl, lines["cloudy"], "--", lw=1.2, color=BLUE, label="cloudy (1 mbar deck) toy scenario (offset-fitted)")
        h_data = ax.errorbar(wl, ppm(d), yerr=s * 1e6, xerr=(hi - lo) / 2, fmt="o", ms=3.5, lw=1, color=GREY, capsize=0, zorder=5,
                             label="measured spectrum, published error bars")
        handles = handles or [h_data, h_band, h_clear, h_cloudy, h_fill]
        lo_y = min(np.percentile(ppm(d) - 2 * s * 1e6, 2), np.min(lines["cloudy"]))
        hi_y = max(np.percentile(ppm(d) + 2 * s * 1e6, 98), np.max(lines["cloudy"]))
        span = hi_y - lo_y
        lo_y, hi_y = max(lo_y, np.min(lines["clear"]) if np.min(lines["clear"]) > lo_y - 1.5 * span else lo_y - 1.5 * span), \
            min(max(hi_y, np.max(lines["clear"])), hi_y + 1.5 * span)
        ax.set_ylim(lo_y - 0.05 * span, hi_y + 0.05 * span)
        clipped = np.max(lines["clear"]) > hi_y or np.min(lines["clear"]) < lo_y
        n_txt = f"{n_tr} transit" if n_tr else "N transits not in archive"
        box = (f"nearest toy: {srow['table_best'].replace('_', ' ')}\n"
               f"$b_{{\\rm clear}}$ = {fnum(srow['table_b_clear']):.2f}{'  (clipped)' if clipped else ''},  $b_{{\\rm cloudy}}$ = {fnum(srow['table_b_cloudy']):.2f}\n"
               f"σ measured {np.median(s) * 1e6:.0f} ppm ({n_txt})\n"
               f"σ Lazuli {np.median(sig_laz) * 1e6:.0f} ppm (1 transit)")
        bx, by, va, ha = CORNER[corner]
        ax.text(bx, by, box, transform=ax.transAxes, va=va, ha=ha, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.92), zorder=6)
        ax.set_title(f"{planet}: {key[1].split(' et al')[0]} et al., {inst_label}" + (f" (binned to R = {R_disp} for display)" if R_disp else ""), fontsize=9.5)
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Depth − weighted mean (ppm)")
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Check D: measured transmission spectra, Lazuli's predicted single-transit error at the same bins, and the two bracketing toy scenarios",
                 y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "checkD_spectra")


# ---------------------------------------------------------------- check G
def figure_g():
    plt = gks_style()
    visits = [(k[1], wl, bw, d, s) for k, wl, bw, d, s in all_datasets() if k[0] == "TRAPPIST-1 e" and "PRISM" in k[2]]
    wl0, bw0 = visits[0][1], visits[0][2]
    D = np.vstack([np.interp(wl0, wl, d) for _, wl, bw, d, s in visits])
    S = np.vstack([np.interp(wl0, wl, s) for _, wl, bw, d, s in visits])
    mean = np.average(D, axis=0, weights=1 / S ** 2)
    lo, hi = _bin_edges(wl0, bw0)
    sig_laz = lazuli_sigma_for_bins("TRAPPIST-1 e", lo, hi)
    g = read_csv(RESULTS / "checkG_tlse.csv")
    per_channel = fnum(next(r for r in g if r["band_um"] == "1.0-1.7")["lazuli_1transit_sigma_ppm_median_channel"])
    fig, ax = plt.subplots(figsize=(10, 5))
    red = wl0 >= 0.8
    ax.fill_between(wl0[red], (mean - sig_laz)[red] * 1e6, (mean + sig_laz)[red] * 1e6, color=TEAL, alpha=0.25, lw=0,
                    label=f"Lazuli predicted ±1σ, 1 transit, at the PRISM bins (median {np.median(sig_laz[wl0 > 1.0]) * 1e6:.0f} ppm per bin, "
                          f"{per_channel:.0f} ppm per native channel, 1.0–1.7 µm)")
    ax.plot(wl0[red], mean[red] * 1e6, "-", color=TEAL, lw=1.3, label="weighted mean of the four visits")
    styles = [("o", "0.15"), ("s", "0.4"), ("^", "0.6"), ("D", RED)]
    for (name, wl, bw, d, s), (mk, c), dx in zip(visits, styles, (-0.006, -0.002, 0.002, 0.006)):
        m = wl >= 0.8
        ax.errorbar(wl[m] + dx, d[m] * 1e6, yerr=s[m] * 1e6, fmt=mk, ms=4, lw=0.9, color=c, capsize=0, label=name.split(", ")[-1])
    ax.axvline(1.0, color="0.7", ls=":", lw=1)
    ax.text(1.005, 5560, "Table G bands: 0.6–1.0 | 1.0–1.7 µm", fontsize=8, color="0.45", va="top")
    ax.set_xlim(0.78, 1.72)
    ax.set_ylim(4000, 5600)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Transit depth (ppm)")
    ax.set_title("Check G: TRAPPIST-1 e, four JWST/NIRSpec PRISM visits (Espinoza et al. 2025) vs Lazuli's predicted single-transit noise", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    savefig(fig, "checkG_trappist1e_visits")


if __name__ == "__main__":
    figure_a()
    figure_b()
    figure_d()
    figure_g()
