"""Compare the parametric model against the 14 x 4 committed POSEIDON tables.

Run: ~/anaconda3/envs/py313/bin/python precompute/validate_zeta_grid.py
Writes notebooks/figs/06_validation.json and prints a markdown table.

For each planet x scenario both spectra are binned to R = 100 over
0.40-1.70 um, a constant offset (the P_ref convention, which the parametric
path defines away) is removed, and the feature amplitude (max - min) is
compared. The three T_eq < 300 K planets were computed by POSEIDON with the
isotherm clamped to 300 K for the scale height as well; for them a second,
like-for-like row with T_eq_K = 300 is reported and is the one gated.

Radius convention: the POSEIDON tables put the archive R_p at P_ref = 10 bar
(and evaluate gravity there), whereas the parametric model's R_p is the
band-median transit radius with gravity evaluated there. These are different
planets by ~10 scale heights, so the comparison passes the table's own
band-median transit radius, R_p = sqrt(median depth) R_s, to the model. The
reported "offset_ppm" is then a residual, not the P_ref convention.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terminator import binning, presets, tables  # noqa: E402
from terminator.atmosphere import transmission_depth  # noqa: E402

TOL = {"clear": 0.10, "high_metallicity": 0.10, "cloudy": 0.20, "hazy": 0.20}
R_CMP, BAND = 100, (0.40, 1.70)
R_J_M, R_SUN_M = 7.1492e7, 6.957e8
OUT = ROOT / "notebooks" / "figs" / "06_validation.json"
ATMO_KEYS = ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "log_met", "c_to_o",
             "mu", "log_P_cloud_bar", "log_a_haze", "haze_slope")


def compare(planet, scen, tdir, edges, T_override=None):
    wl, d_tab, meta_tab = tables.load_model(tdir, planet, scen)
    p = presets.showcase(planet, scen)
    kw = {k: p[k] for k in ATMO_KEYS}
    if T_override is not None:
        kw["T_eq_K"] = T_override
    band = (wl >= BAND[0]) & (wl <= BAND[1])
    kw["R_p_RJ"] = float(np.sqrt(np.median(d_tab[band])) * p["R_s_RSun"] * R_SUN_M / R_J_M)
    d_mod, info = transmission_depth(wl, **kw, tables_dir=tdir)
    bt = binning.bin_model(wl, d_tab, edges)
    bm = binning.bin_model(wl, d_mod, edges)
    offset = float(np.mean(bm - bt))
    dev = bm - bt - offset
    amp_t, amp_m = float(np.ptp(bt)), float(np.ptp(bm))
    ratio = amp_m / amp_t if amp_t > 0 else float("nan")
    return dict(planet=planet, scenario=scen,
                T_eq_used_K=float(kw["T_eq_K"]),
                R_p_archive_RJ=p["R_p_RJ"], R_p_median_RJ=kw["R_p_RJ"],
                H_over_Rp=info["H_over_Rp"],
                like_for_like=T_override is not None,
                max_abs_dev_ppm=float(np.max(np.abs(dev)) * 1e6),
                amp_table_ppm=amp_t * 1e6, amp_model_ppm=amp_m * 1e6,
                amp_ratio=float(ratio), offset_ppm=offset * 1e6,
                tol=TOL[scen], T_clamped=bool(info["T_clamped"]),
                T_iso_table_K=float(meta_tab.get("T_iso_K", kw["T_eq_K"])),
                passed=bool(abs(ratio - 1) <= TOL[scen]))


def validate(tables_dir=None):
    tdir = tables.resolve_tables_dir(tables_dir)
    edges = binning.constant_R_edges(*BAND, R_CMP)
    rows = []
    for planet in presets.PLANET_NAMES:
        for scen in TOL:
            row = compare(planet, scen, tdir, edges)
            rows.append(row)
            if row["T_clamped"]:
                rows.append(compare(planet, scen, tdir, edges,
                                    T_override=row["T_iso_table_K"]))
    return rows


def main():
    rows = validate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print("| planet | scenario | T_eq used | amp table [ppm] | amp model [ppm] "
          "| ratio | max dev [ppm] | pass |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        tag = " (like-for-like)" if r["like_for_like"] else ""
        print(f"| {r['planet']} | {r['scenario']} | {r['T_eq_used_K']:.0f}{tag} | "
              f"{r['amp_table_ppm']:.0f} | {r['amp_model_ppm']:.0f} | "
              f"{r['amp_ratio']:.2f} | {r['max_abs_dev_ppm']:.0f} | "
              f"{'yes' if r['passed'] else 'NO'} |")
    gated = [r for r in rows if r["like_for_like"] or not r["T_clamped"]]
    n_fail = sum(not r["passed"] for r in gated)
    print(f"\n{len(gated) - n_fail}/{len(gated)} gated comparisons within tolerance")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
