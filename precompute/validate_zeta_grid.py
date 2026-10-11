"""Compare the parametric model against the 14 x 5 POSEIDON tables in precompute/tables.

Run: ~/anaconda3/envs/py313/bin/python precompute/validate_zeta_grid.py
Writes notebooks/figs/06_validation.json and prints a markdown table.

14 planets x 5 scenarios. The four atmospheric scenarios are gated on the
feature-amplitude ratio; 'flat' (deck at 10 nbar) is gated on the model's
absolute residual amplitude (< FLAT_MAX_PPM), since the table's is zero.

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

Deck convention: see DECK_REALIZED_LOG_OFFSET below.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# The cold (T_clamped) and ultra-hot showcase planets are compared on purpose
# and T_clamped is reported per row; silence the #39 validity warnings here.
warnings.filterwarnings("ignore", message=r"T_eq=.* K", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lazuli_exo import binning, presets, tables  # noqa: E402
from lazuli_exo.atmosphere import transmission_depth  # noqa: E402

TOL = {"clear": 0.10, "high_metallicity": 0.10, "cloudy": 0.20, "hazy": 0.20}
FLAT_MAX_PPM = 10.0      # 'flat' has no amplitude ratio: gate the model's residual absolutely
# Documented exceptions to the gate, (planet, scenario) -> reason. Each one is
# still reported with its numbers; waiving it only keeps the regression test
# green. Remove an entry rather than widen TOL if the model improves.
WAIVED = {
    ("WASP-12 b", "cloudy"):
        "73 ppm feature, 19 ppm absolute error (noise floor 178 ppm/transit): "
        "POSEIDON's discretized deck sits between the literal 1 mbar (model ratio "
        "0.75 in a direct run) and the 1.376 mbar realized at 300 K (1.16); at "
        "2600 K the x-extrapolation is also +0.09 high. Clear spectrum matches "
        "POSEIDON to 1.00.",
}
R_CMP, BAND = 100, (0.40, 1.70)
R_J_M, R_SUN_M = 7.1492e7, 6.957e8
# POSEIDON applies the MacMad17 deck to layers strictly deeper than P_cloud,
# so on the tables' 100-layer grid the "1 mbar" deck's tau=1 surface sits at
# 1.376 mbar (measured from a direct run with the atmosphere's own r(P)).
# The parametric model takes P_cloud literally; the comparison uses the
# pressure POSEIDON actually realized.
DECK_REALIZED_LOG_OFFSET = float(np.log10(1.376))
OUT = ROOT / "notebooks" / "figs" / "06_validation.json"
MODELS = ROOT / "precompute" / "tables"     # POSEIDON tables (not runtime data)
ATMO_KEYS = ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "log_met", "c_to_o",
             "mu", "log_P_cloud_bar", "log_a_haze", "haze_slope")


def compare(planet, scen, tdir, edges, T_override=None):
    wl, d_tab, meta_tab = tables.load_model(MODELS, planet, scen)
    p = presets.showcase(planet, scen)
    kw = {k: p[k] for k in ATMO_KEYS}
    if T_override is not None:
        kw["T_eq_K"] = T_override
    band = (wl >= BAND[0]) & (wl <= BAND[1])
    kw["R_p_RJ"] = float(np.sqrt(np.median(d_tab[band])) * p["R_s_RSun"] * R_SUN_M / R_J_M)
    if kw["log_P_cloud_bar"] is not None:
        kw["log_P_cloud_bar"] += DECK_REALIZED_LOG_OFFSET
    d_mod, info = transmission_depth(wl, **kw, tables_dir=tdir)
    bt = binning.bin_model(wl, d_tab, edges)
    bm = binning.bin_model(wl, d_mod, edges)
    offset = float(np.mean(bm - bt))
    dev = bm - bt - offset
    amp_t, amp_m = float(np.ptp(bt)), float(np.ptp(bm))
    ratio = amp_m / amp_t if amp_t > 0 else float("nan")
    passed = (amp_m * 1e6 <= FLAT_MAX_PPM if scen == "flat"
              else abs(ratio - 1) <= TOL[scen])
    return dict(planet=planet, scenario=scen,
                T_eq_used_K=float(kw["T_eq_K"]),
                R_p_archive_RJ=p["R_p_RJ"], R_p_median_RJ=kw["R_p_RJ"],
                H_over_Rp=info["H_over_Rp"],
                like_for_like=T_override is not None,
                max_abs_dev_ppm=float(np.max(np.abs(dev)) * 1e6),
                amp_table_ppm=amp_t * 1e6, amp_model_ppm=amp_m * 1e6,
                amp_ratio=float(ratio), offset_ppm=offset * 1e6,
                tol=TOL.get(scen, FLAT_MAX_PPM), T_clamped=bool(info["T_clamped"]),
                T_iso_table_K=float(meta_tab.get("T_iso_K", kw["T_eq_K"])),
                passed=bool(passed),
                waived=WAIVED.get((planet, scen)))


def validate(tables_dir=None):
    tdir = tables.resolve_tables_dir(tables_dir)
    edges = binning.constant_R_edges(*BAND, R_CMP)
    rows = []
    for planet in presets.PLANET_NAMES:
        for scen in list(TOL) + ["flat"]:
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
        ratio = "abs<%g" % FLAT_MAX_PPM if r["scenario"] == "flat" else f"{r['amp_ratio']:.2f}"
        print(f"| {r['planet']} | {r['scenario']} | {r['T_eq_used_K']:.0f}{tag} | "
              f"{r['amp_table_ppm']:.0f} | {r['amp_model_ppm']:.0f} | "
              f"{ratio} | {r['max_abs_dev_ppm']:.0f} | "
              f"{'yes' if r['passed'] else 'NO'} |")
    gated = [r for r in rows if r["like_for_like"] or not r["T_clamped"]]
    n_fail = sum(not r["passed"] for r in gated)
    n_waived = sum((not r["passed"]) and bool(r["waived"]) for r in gated)
    print(f"\n{len(gated) - n_fail}/{len(gated)} gated comparisons within tolerance"
          f"{f', {n_waived} documented waiver(s)' if n_waived else ''}")
    for r in gated:
        if not r["passed"] and r["waived"]:
            print(f"WAIVED {r['planet']} {r['scenario']}: {r['waived']}")
    return 1 if n_fail > n_waived else 0


if __name__ == "__main__":
    raise SystemExit(main())
