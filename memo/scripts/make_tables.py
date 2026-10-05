"""Render the check results (memo/results/*.csv) as LaTeX tables in memo/results/tables.tex."""
import numpy as np

from common import RESULTS, TARGETS, fnum, read_csv


def tex(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def f(x, nd=2, dash="--"):
    x = fnum(x)
    return dash if not np.isfinite(x) else f"{x:.{nd}f}"


def table(cols, rows, caption, label, align=None, size=r"\small"):
    align = align or "l" + "r" * (len(cols) - 1)
    out = [r"\begin{table}[H]", r"\centering" + size, r"\begin{tabular}{@{}" + align + "@{}}", r"\toprule",
           " & ".join(r"\textbf{" + c + "}" for c in cols) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\caption*{\footnotesize\textbf{" + label + ".} " + caption + "}", r"\end{table}", ""]
    return "\n".join(out)


def table_a():
    rows = {}
    for r in read_csv(RESULTS / "checkA_sed.csv"):
        rows.setdefault(r["planet"], {})[r["model"]] = r
    out = []
    for p, d in rows.items():
        t, g = d["table"], d["phoenix_grid"]
        out.append([tex(t["host"]), f"{fnum(t['Teff']):.0f}", tex(t["grid"]),
                    f(t["d(V-J)"]), f(t["d(J-H)"]), f(t.get("photon_ratio_0.40-0.70")), f(t.get("photon_ratio_0.70-1.02")),
                    f(g["d(V-J)"]), f(g.get("photon_ratio_0.40-0.70")), f(g.get("photon_ratio_0.70-1.02"))])
    return table(["Host", r"$T_{\rm eff}$", "table star", r"$\Delta(V{-}J)$", r"$\Delta(J{-}H)$", "blue", "red",
                  r"$\Delta(V{-}J)$", "blue", "red"], out,
                 r"Check A. Columns 4--7: the committed star table (POSEIDON PHOENIX, or blackbody); columns 8--10: the parametric path's PHOENIX mini-grid. "
                 r"$\Delta$ = synthetic minus SIMBAD colour (Vega). ``blue'' and ``red'' are the $J$-normalized model photon flux over the Gaia DR3 XP flux "
                 r"integrated over 0.40--0.70 and 0.70--1.02\,$\mu$m (1.00 = perfect). K2-18 and TRAPPIST-1 have no XP spectrum; GJ~3470 and HAT-P-11 have no SIMBAD $V$. Derived here.",
                 "Table A", align="lll" + "r" * 7, size=r"\footnotesize")


def table_b1():
    out = []
    for r in read_csv(RESULTS / "checkB_photon_budget.csv"):
        out.append([tex(r["planet"]), f(r["J_vega"]), f(r["J_ab"]), r["nmd"], f"{fnum(r['t_int_s']):.1f}",
                    f(r["ratio_median"], 3), f(r["ratio_p16"], 3) + "--" + f(r["ratio_p84"], 3), f(r["var_over_signal_median"])])
    return table(["Planet", r"$J_{\rm Vega}$", r"$J_{\rm AB}$", "MACC", r"$t_{\rm int}$ (s)", "slicersim / hand", "16--84\\%", r"var / signal"], out,
                 r"Check B(i). Extracted slicersim signal over the hand photon budget $F_\lambda\,\lambda/(hc)\,A\,\eta\,{\rm QE}\,\Delta\lambda\,t_{\rm int}$ "
                 r"(median over 0.40--1.70\,$\mu$m, with the 16--84\% range), where $F_\lambda$ is the star table normalized here to the catalog $J$ with sncosmo "
                 r"and $A=7.07$\,m$^2$, $\eta$ and QE are read from the \texttt{lazuli\_cbe} configuration. The last column is the Rauscher (2007) up-the-ramp "
                 r"variance over the Poisson variance of the accumulated signal. Derived here.", "Table B1", align="l" + "r" * 7)


def table_b2():
    out = []
    for r in read_csv(RESULTS / "checkB_achieved_precision.csv"):
        flag = r"$\downarrow$" if r["lazuli_better_than_area_scaling"] == "True" else ""
        out.append([tex(r["planet"]), tex(r["reference"][:26]), r["facility"], r["n_bins"], f(r["median_bin_width_um"], 3),
                    f"{fnum(r['achieved_sigma_ppm']):.0f}", f"{fnum(r['lazuli_1transit_sigma_ppm']):.0f}", f(r["ratio_lazuli_over_achieved"]) + flag,
                    f(r["area_only_expectation"])])
    return table(["Planet", "Dataset", "Fac.", r"$n$", r"bin ($\mu$m)", r"achieved $\sigma$", r"Lazuli $\sigma_1$", "ratio", "area-only"], out,
                 r"Check B(ii). Published per-bin depth precision inside 0.40--1.70\,$\mu$m (median, ppm, as tabulated in the NASA Exoplanet Archive) against "
                 r"lazuli-exo's predicted single-transit precision for the same bins (committed noise tables; statistical noise only). ``area-only'' is "
                 r"$\sqrt{A_{\rm other}/A_{\rm Lazuli}}$ with $A_{\rm HST}=4.0$, $A_{\rm JWST}=25.4$, $A_{\rm Lazuli}=7.07$\,m$^2$; $\downarrow$ marks datasets where "
                 r"Lazuli's predicted error is smaller than that scaling. The archive does not record the number of visits combined, so ratios are indicative. Derived here.",
                 "Table B2", align="lllr" + "r" * 5, size=r"\scriptsize")


def table_c():
    out = []
    sol = read_csv(RESULTS / "checkC_solutions.csv")
    for r in read_csv(RESULTS / "checkC_registry.csv"):
        mine = [s for s in sol if s["planet"] == r["planet"]]
        last = mine[-1] if mine else None
        out.append([tex(r["planet"]), f(r["ratio"], 3), f(r["b"]), r["n_reference_solutions"], f(r["k_spread_over_k"], 3),
                    f"{fnum(r['depth_spread_ppm']):.0f}" if mine else "--",
                    tex(last["reference"][:22]) if last else "--",
                    f(last["k_over_registry"], 3) if last else "--", f(last["a_rs_over_registry"], 3) if last else "--",
                    f(last["t14_geometry_over_registry_t14"], 3) if last else "--"])
    t1 = table(["Planet", r"$t_{14}^{\rm geo}/t_{14}$", r"$b$", r"$n_{\rm ref}$", r"$\Delta k/k$", r"$\Delta\delta$ (ppm)", "latest reference", r"$k/k_{\rm reg}$", r"$a/R_\star$ ratio", r"$t_{14}^{\rm geo}/t_{14}^{\rm reg}$"], out,
               r"Check C, part 1. Column 2: $t_{14}$ recomputed from the registry's own $(a/R_\star, i, e, \omega, k)$ over the registry $t_{14}$. Columns 4--6: number of "
               r"archive solutions with $(k, a/R_\star, i)$, their spread in $k$ and in white-light depth. Columns 7--10: the most recent such solution against the registry. Derived here.",
               "Table C1", align="l" + "r" * 5 + "l" + "r" * 3, size=r"\footnotesize")
    out = []
    for r in read_csv(RESULTS / "checkC_lightcurve.csv"):
        out.append([tex(r["planet"]), f(r["solver_vs_batman_max_ppm"]), f"{fnum(r['local_arc_vs_kepler_max_ppm']):.1f}",
                    f"{fnum(r['sigma_point_ppm']):.0f}", f(r["batman_t14_over_registry"], 3), r["ld_chunks_negative_intensity"], f(r["ld_u1_plus_u2_max"], 3)])
    t2 = table(["Planet", "solver (ppm)", "local arc (ppm)", r"$\sigma_{\rm pt}$ (ppm)", r"$t_{14}^{\rm batman}/t_{14}^{\rm reg}$", r"LD $I<0$", r"max $u_1{+}u_2$"], out,
               r"Check C, part 2. Numpy transit solver vs batman at identical separations (max $|\Delta F|$); lazuli-exo's local-arc $z(t)$ vs batman's Keplerian $z(t)$ "
               r"for the white-light curve; the per-sample noise at native cadence for one transit; batman's $t_{14}$ from the registry geometry over the registry $t_{14}$; "
               r"number of LD chunks with negative limb intensity. Derived here.", "Table C2")
    return t1 + t2


def table_d():
    rows = read_csv(RESULTS / "checkD_spectra_summary.csv")
    cap = (r"Check D. Every dataset with $\ge 5$ points and errors inside 0.40--1.70\,$\mu$m (NASA Exoplanet Archive; TRAPPIST-1\,e from Espinoza et al.\ 2025). "
           r"``amp'' and $\sigma$: measured peak-to-peak depth and median error (ppm). ``best'' is the scenario with the lowest offset-fit $\chi^2_\nu$ "
           r"(also given for the flat scenario); $b$ is the fitted data/model amplitude ratio (1 = the scenario has the measured amplitude); "
           r"``brk.'' = the four atmospheric scenarios' $b$ values straddle 1. Derived here.")
    out = {"table": [], "parametric": []}
    for r in rows:
        ds = tex(r["reference"][:22] + ", " + r["instrument"][:12])
        for path in out:
            out[path].append([tex(r["planet"]), ds, r["facility"][:4], r["n_bins"], f"{fnum(r['data_amp_ppm']):.0f}", f"{fnum(r['median_sigma_ppm']):.0f}",
                              tex(r[f"{path}_best"].replace("high_metallicity", "high-met")), f(r[f"{path}_chi2_best"], 1), f(r[f"{path}_chi2_flat"], 1),
                              f(r[f"{path}_b_clear"]), f(r[f"{path}_b_cloudy"]), f(r[f"{path}_b_hazy"]), f(r[f"{path}_b_highmet"]),
                              "yes" if r[f"{path}_bracketed"] == "True" else "no"])
    cols = ["Planet", "Dataset", "Fac.", r"$n$", "amp", r"$\sigma$", "best", r"$\chi^2_\nu$", r"$\chi^2_{\nu,\rm flat}$",
            r"$b_{\rm clear}$", r"$b_{\rm cloudy}$", r"$b_{\rm hazy}$", r"$b_{\rm high\text{-}met}$", "brk."]
    t1 = table(cols, out["table"], "Committed POSEIDON tables. " + cap, "Table D1", align="lllrrr" + "lrr" + "rrrr" + "l", size=r"\scriptsize")
    t2 = table(cols, out["parametric"], "Parametric model with the same showcase presets. " + cap, "Table D2", align="lllrrr" + "lrr" + "rrrr" + "l", size=r"\scriptsize")
    return t1 + t2


def table_g():
    out = []
    for r in read_csv(RESULTS / "checkG_tlse.csv"):
        out.append([r["band_um"], r["n_wl"], f"{fnum(r['median_visit_sigma_ppm']):.0f}", f"{fnum(r['median_visit_to_visit_std_ppm']):.0f}",
                    f"{fnum(r['median_excess_scatter_ppm']):.0f}", f"{fnum(r['p84_excess_scatter_ppm']):.0f}", f"{fnum(r['visit_offset_spread_ppm']):.0f}",
                    f"{fnum(r['lazuli_1transit_sigma_ppm_median_channel']):.0f}", f"{fnum(r['param_clear_amp_ppm']):.0f}", f"{fnum(r['param_cloudy_amp_ppm']):.0f}"])
    return table([r"band ($\mu$m)", r"$n_\lambda$", r"$\sigma_{\rm visit}$", "std", "excess", "excess$_{84}$", "offsets",
                  r"Lazuli $\sigma_1$", r"clear", r"cloudy"], out,
                 r"Check G, TRAPPIST-1\,e (Espinoza et al.\ 2025, four JWST/NIRSpec PRISM visits, `Espinoza' reduction). All in ppm: median per-visit error; "
                 r"visit-to-visit standard deviation of the depth at each wavelength; the part of it in excess of the per-visit errors (median and 84th percentile); "
                 r"spread of the four visits' weighted-mean depths; lazuli-exo's single-transit error per native channel; and the parametric clear and cloudy H$_2$/He feature amplitudes "
                 r"over the same band. Derived here.", "Table G", size=r"\footnotesize")


if __name__ == "__main__":
    for name, fn in (("A", table_a), ("B1", table_b1), ("B2", table_b2), ("C", table_c), ("D", table_d), ("G", table_g)):
        (RESULTS / f"table_{name}.tex").write_text(fn())
        print(f"wrote memo/results/table_{name}.tex")
