"""Check D: scenario spectra vs published transmission spectra.

For every archive dataset (any facility) with >= 5 points and errors inside
0.40-1.70 um, plus the Espinoza et al. (2025) TRAPPIST-1 e JWST/PRISM visits
and HST/WFC3 visits, each of the five scenarios is binned to the dataset's
bins on both model paths (committed POSEIDON table; parametric model with the
showcase preset) and compared two ways:
  offset fit     data = c + model            -> chi2_offset / nu
  amplitude fit  data = c + b * model        -> b (data/model amplitude ratio)
'Bracketed' means the four atmospheric scenarios' b values straddle 1, i.e.
the scenario set spans the measured feature amplitude. These are toy
scenarios; the question is representativeness, not goodness of fit.
"""
import numpy as np

from common import BAND_UM, DATA, RESULTS, TDIR, fnum, read_csv, tables, write_csv
from lazuli_exo import presets
from lazuli_exo.atmosphere import transmission_depth
from checkB_photon import _bin_edges, datasets

SCENARIOS = ["clear", "cloudy", "hazy", "high_metallicity", "flat"]
ATMO = ["clear", "cloudy", "hazy", "high_metallicity"]
KEYS = ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "log_met", "c_to_o", "mu", "log_P_cloud_bar", "log_a_haze", "haze_slope")
_MODEL_CACHE = {}


def models(planet, scen):
    """(wl_um, depth_table, depth_parametric) at the table's native grid."""
    if (planet, scen) not in _MODEL_CACHE:
        wl, dt, _ = tables.load_model(TDIR, planet, scen)
        p = presets.showcase(planet, scen)
        dp, _ = transmission_depth(wl, **{k: p[k] for k in KEYS})
        _MODEL_CACHE[(planet, scen)] = (wl, dt, dp)
    return _MODEL_CACHE[(planet, scen)]


def bin_to(wl_m, d_m, lo, hi):
    out = np.empty(lo.size)
    for i, (a, b) in enumerate(zip(lo, hi)):
        m = (wl_m >= a) & (wl_m < b)
        out[i] = d_m[m].mean() if m.any() else np.interp(0.5 * (a + b), wl_m, d_m)
    return out


def fits(d, s, m):
    """Offset fit (data = c + model) and amplitude fit (data = c + b model),
    both weighted by 1/sigma^2; the amplitude fit is done on centred vectors
    so a nearly flat model stays well conditioned."""
    w = 1.0 / s ** 2
    c = np.sum(w * (d - m)) / np.sum(w)
    chi2_off = float(np.sum(w * (d - m - c) ** 2))
    if np.ptp(m) < 1e-7:                        # < 0.1 ppm of structure: nothing to fit
        return c, chi2_off, np.nan, np.nan, np.nan
    mc = m - np.sum(w * m) / np.sum(w)
    dc = d - np.sum(w * d) / np.sum(w)
    b = float(np.sum(w * mc * dc) / np.sum(w * mc ** 2))
    b_err = float(1.0 / np.sqrt(np.sum(w * mc ** 2)))
    chi2_amp = float(np.sum(w * (dc - b * mc) ** 2))
    return c, chi2_off, b, b_err, chi2_amp


def trappist1e_datasets():
    for f in sorted((DATA / "trappist1e").glob("*.txt")):
        rows = [l.split() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")]
        a = np.array([[fnum(x) for x in r] for r in rows])
        wl, lo2, hi2, d, s = a.T
        m = np.isfinite(d) & np.isfinite(s) & (wl >= BAND_UM[0]) & (wl <= BAND_UM[1])
        name = f.stem.replace("JWST_PRISM_Espinoza", "JWST/PRISM visit ").replace("HST_WFC3IR_visit", "HST/WFC3 visit ")
        kind = "JWST" if "PRISM" in f.stem else "HST"
        yield ("TRAPPIST-1 e", f"Espinoza et al. 2025, {name}", f.stem, kind), wl[m], (lo2 + hi2)[m], d[m] * 1e-6, s[m] * 1e-6


def all_datasets():
    yield from datasets(kinds=("JWST", "HST", "ground"))
    yield from trappist1e_datasets()


def main():
    rows, summary = [], []
    for (planet, ref, inst, kind), wl, bw, d, s in all_datasets():
        lo, hi = _bin_edges(wl, bw)
        n = len(wl)
        per_path = {}
        for path in ("table", "parametric"):
            res = {}
            for scen in SCENARIOS:
                wl_m, dt, dp = models(planet, scen)
                m = bin_to(wl_m, dt if path == "table" else dp, lo, hi)
                c, chi2_off, b, b_err, chi2_amp = fits(d, s, m)
                if np.ptp(m) < 1e-9:
                    b, b_err, chi2_amp = np.nan, np.nan, np.nan
                res[scen] = dict(c=c, chi2_offset_red=chi2_off / max(n - 1, 1), b=b, b_err=b_err,
                                 chi2_amp_red=chi2_amp / max(n - 2, 1), model_amp_ppm=float(np.ptp(m)) * 1e6)
                rows.append(dict(planet=planet, reference=ref, instrument=inst, facility=kind, path=path,
                                 scenario=scen, n_bins=n, wl_min_um=wl.min(), wl_max_um=wl.max(),
                                 data_amp_ppm=float(np.ptp(d)) * 1e6, median_sigma_ppm=float(np.median(s)) * 1e6, **res[scen]))
            bs = {k: res[k]["b"] for k in ATMO}
            best = min(SCENARIOS, key=lambda k: res[k]["chi2_offset_red"])
            per_path[path] = dict(best=best, chi2_best=res[best]["chi2_offset_red"], chi2_flat=res["flat"]["chi2_offset_red"],
                                  b_clear=bs["clear"], b_cloudy=bs["cloudy"], b_hazy=bs["hazy"], b_highmet=bs["high_metallicity"],
                                  bracketed=bool(min(bs.values()) <= 1.0 <= max(bs.values())))
        summary.append(dict(planet=planet, reference=ref, instrument=inst[:20], facility=kind, n_bins=n,
                            wl_min_um=wl.min(), wl_max_um=wl.max(), data_amp_ppm=float(np.ptp(d)) * 1e6,
                            median_sigma_ppm=float(np.median(s)) * 1e6,
                            **{f"{p}_{k}": v for p, dd in per_path.items() for k, v in dd.items()}))
    write_csv(RESULTS / "checkD_spectra_long.csv", rows)
    write_csv(RESULTS / "checkD_spectra_summary.csv", summary)
    print(f"\n{'planet':12s} {'dataset':30s} {'fac':6s} {'n':>3s} {'amp':>5s} {'sig':>5s} | table: best chi2r chi2r_flat b_clr b_cld brk | param: best chi2r b_clr b_cld brk")
    for r in summary:
        print(f"{r['planet']:12s} {(r['reference'][:18] + ' ' + r['instrument'][:10]):30s} {r['facility']:6s} {r['n_bins']:3d} {r['data_amp_ppm']:5.0f} {r['median_sigma_ppm']:5.0f} | "
              f"{r['table_best']:8s} {r['table_chi2_best']:5.1f} {r['table_chi2_flat']:5.1f} {r['table_b_clear']:5.2f} {r['table_b_cloudy']:5.2f} {'Y' if r['table_bracketed'] else 'n'} | "
              f"{r['parametric_best']:8s} {r['parametric_chi2_best']:5.1f} {r['parametric_b_clear']:5.2f} {r['parametric_b_cloudy']:5.2f} {'Y' if r['parametric_bracketed'] else 'n'}")


if __name__ == "__main__":
    main()
