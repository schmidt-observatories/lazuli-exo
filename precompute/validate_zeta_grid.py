"""Gate the parametric model against the POSEIDON reference set (issue #22).

Run: PYTHONPATH=src python precompute/validate_zeta_grid.py
Writes notebooks/figs/06_validation.json, prints the markdown tables and a
gate/waiver summary; exits 1 on any un-waived failure or stale waiver.
tests/test_validation_report.py runs the same two checks.

Two reference sets, both committed under precompute/tables:

* models/: 14 planets x 5 scenarios of full POSEIDON spectra. Both spectra
  are binned to R = 100 over 0.40-1.70 um, a constant offset is removed (the
  P_ref convention, which the parametric path defines away) and the residual
  is compared with the single-transit per-bin sigma of that planet's noise
  table. The three T_eq < 300 K planets were computed by POSEIDON with the
  isotherm clamped to 300 K for the scale height as well; for them a second,
  like-for-like row with T_eq_K = 300 is reported and is the one gated.
* holdouts/zeta_holdouts.npz: direct POSEIDON zeta at the two fiducial nodes
  and the transit depth at a planet-like node, at points off the grid nodes
  (T = 650/1100/2450 K midpoints, 567/736/1166 K showcase temperatures,
  [M/H] = 1.75, C/O = 0.9, x = +0.63). The grid interpolation is evaluated at
  the same points (``atmosphere._trilinear`` for the fiducial zeta,
  ``transmission_depth`` end to end for the planet-like depth).

Gates (review 2026-10-10, area B):
  G2    worst bin |residual| < 1 sigma (single transit)
  G5    err_snr = sqrt(sum (residual/sigma)^2) < 3  (the error spectrum would
        not be detected at 3 sigma in one transit)
  G3    template S/N ratio model/table within 10 % (5 % is below the model's
        own mu/x fidelity)
  flat  model residual amplitude < FLAT_MAX_PPM (the table's is zero)
  GZ    holdouts only: worst-bin fiducial |dzeta|, converted to ppm on the
        named planet, < 1 sigma

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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lazuli_exo import binning, detect, presets, tables  # noqa: E402
from lazuli_exo.atmosphere import _trilinear, transmission_depth  # noqa: E402

SCENARIOS = ("clear", "high_metallicity", "cloudy", "hazy", "flat")
FLAT_MAX_PPM = 10.0
GATES = {   # name -> (description, predicate on a metrics row)
    "G2": ("worst bin < 1 sigma (1 transit)", lambda r: abs(r["worst_sig"]) < 1.0),
    "G5": ("err_snr < 3", lambda r: r["err_snr"] < 3.0),
    "G3": ("template S/N ratio within 10 %", lambda r: abs(r["snr_ratio"] - 1) < 0.10),
    "flat": (f"flat: model amplitude < {FLAT_MAX_PPM:g} ppm",
             lambda r: r["amp_model_ppm"] < FLAT_MAX_PPM),
    "GZ": ("holdout fiducial |dzeta| worst bin < 1 sigma", lambda r: abs(r["dzeta_sig"]) < 1.0),
}
MODEL_GATES = {"flat": ("flat",), None: ("G2", "G5", "G3")}
HOLDOUT_GATES = ("GZ", "G2", "G5")

# Documented exceptions: (planet, scenario) or holdout label -> {gates, reason}.
# A waiver is an expected failure: the row is still reported with its numbers,
# and a waiver whose gates no longer fail is itself an error (remove it with
# the fix that made it pass). Never widen a threshold instead.
P35_T = "pending #35 (T nodes across the alkali transition)"
P35_X = "pending #35 (x extrapolation)"
P35_CO = "pending #35 (C/O 0.9 node)"
P35_MU = "pending #35 (log-mu interpolation at [M/H] 1.75)"
WAIVED = {
    ("WASP-107 b", "hazy"): dict(gates=("G2", "G5"), reason=P35_X),
    ("WASP-107 b", "clear"): dict(gates=("G2", "G5"), reason=P35_X),
    ("WASP-107 b", "cloudy"): dict(gates=("G2", "G5"), reason=P35_X),
    ("WASP-107 b", "high_metallicity"): dict(gates=("G2", "G5"), reason=P35_X),
    ("GJ 1214 b", "clear"): dict(gates=("G2", "G5"), reason=P35_T),
    ("GJ 1214 b", "high_metallicity"): dict(gates=("G5",), reason=P35_T),
    ("WASP-12 b", "cloudy"): dict(gates=("G3",), reason=(
        "73 ppm feature, 19 ppm absolute error (noise floor 178 ppm/transit): "
        "POSEIDON's discretized deck sits between the literal 1 mbar and the "
        "1.376 mbar realized at 300 K; at 2600 K the x-extrapolation is also "
        "+0.09 high. Clear spectrum matches POSEIDON to 1.00.")),
    ("WASP-121 b", "cloudy"): dict(gates=("G3",), reason=(
        "203 ppm feature, 15 ppm absolute error: discretized-deck convention "
        "(review B F8), same as WASP-12 b cloudy.")),
    ("TRAPPIST-1 e", "cloudy"): dict(gates=("G3",), reason=(
        "amplitude 0.88 at the 300 K node (0.15 sigma of 553 ppm/transit): "
        "slant-column x-scaling at H/R_p = 0.026 (review B F7).")),
    # holdouts (label -> gates): the alkali-transition, x-extrapolation and
    # C/O = 0.9 cases of review B, i.e. exactly the #35 regeneration items
    "gj1214_567_m0": dict(gates=("G2", "G5"), reason=P35_T),
    "gj1214_567_m2": dict(gates=("G5",), reason=P35_T),
    "gj3470_650_m2": dict(gates=("G5",), reason=P35_T),
    "gj3470_600_m1.75": dict(gates=("G5",), reason=P35_MU),
    "hd189_1100_m0": dict(gates=("GZ", "G5"), reason=(
        "pending #35 (T nodes: +0.19 H at the 1.62 um CH4/H2O band between the "
        "1000 and 1200 K nodes, 1.1 sigma on this 29 ppm host)")),
    "hd189_1200_co0.9": dict(gates=("GZ", "G2", "G5"), reason=P35_CO),
    "w12_2600_co0.9": dict(gates=("GZ", "G2", "G5"), reason=P35_CO),
    "w107_700_m0": dict(gates=("G2", "G5"), reason=P35_X),
    "w107_736_m0": dict(gates=("GZ", "G2", "G5"), reason=f"{P35_X}; {P35_T}"),
    "w107_800_m0": dict(gates=("G2", "G5"), reason=P35_X),
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
HOLDOUTS = MODELS / "holdouts" / "zeta_holdouts.npz"
ATMO_KEYS = ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "log_met", "c_to_o",
             "mu", "log_P_cloud_bar", "log_a_haze", "haze_slope")


def _require(path, what):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: the {what} are committed in the repo "
            "(precompute/tables); check out the full tree, do not skip.")


def _sigma(tdir, planet, edges):
    wl_n, err, _ = tables.load_noise(tdir, planet)
    return binning.bin_sigma(wl_n, err, edges)


def _metrics(bt, bm, cen, sig):
    """Residual metrics of binned model `bm` vs binned table `bt`."""
    res = bm - bt
    offset = float(np.mean(res))
    res = res - offset
    z = res / sig
    iw = int(np.argmax(np.abs(res)))
    amp_t, amp_m = float(np.ptp(bt)), float(np.ptp(bm))
    snr_t = detect.detect_features(cen, bt, sig)["template_snr_diagonal"]
    snr_m = detect.detect_features(cen, bm, sig)["template_snr_diagonal"]
    return dict(rms_ppm=float(np.sqrt(np.mean(res ** 2)) * 1e6),
                worst_ppm=float(res[iw] * 1e6), worst_um=float(cen[iw]),
                worst_sig=float(z[iw]), sigma_med_ppm=float(np.median(sig) * 1e6),
                err_snr=float(np.sqrt(np.sum(z ** 2))),
                snr_table=snr_t, snr_model=snr_m,
                snr_ratio=snr_m / snr_t if snr_t > 0 else float("nan"),
                amp_table_ppm=amp_t * 1e6, amp_model_ppm=amp_m * 1e6,
                amp_ratio=amp_m / amp_t if amp_t > 0 else float("nan"),
                offset_ppm=offset * 1e6)


def _apply_gates(row, key, gates):
    row["gates"] = list(gates)
    row["failed"] = [g for g in gates if not GATES[g][1](row)]
    w = WAIVED.get(key)
    row["waived"] = w["reason"] if w else None
    row["unwaived"] = [g for g in row["failed"] if not (w and g in w["gates"])]
    row["stale_waiver"] = [g for g in w["gates"] if g not in row["failed"]] if w else []
    return row


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
    row = dict(planet=planet, scenario=scen,
               T_eq_used_K=float(kw["T_eq_K"]),
               R_p_archive_RJ=p["R_p_RJ"], R_p_median_RJ=kw["R_p_RJ"],
               H_over_Rp=info["H_over_Rp"],
               like_for_like=T_override is not None,
               T_clamped=bool(info["T_clamped"]),
               T_iso_table_K=float(meta_tab.get("T_iso_K", kw["T_eq_K"])))
    row.update(_metrics(binning.bin_model(wl, d_tab, edges),
                        binning.bin_model(wl, d_mod, edges),
                        binning.bin_centers(edges), _sigma(tdir, planet, edges)))
    if scen == "flat":        # the table is featureless: ratios are meaningless
        row["snr_ratio"] = row["amp_ratio"] = float("nan")
    row["gated"] = row["like_for_like"] or not row["T_clamped"]
    return _apply_gates(row, (planet, scen), MODEL_GATES.get(scen, MODEL_GATES[None]))


def validate(tables_dir=None):
    """One row per (planet, scenario), plus a like-for-like row for the
    clamped planets. Only rows with row["gated"] count."""
    _require(MODELS / "models", "POSEIDON model tables")
    tdir = tables.resolve_tables_dir(tables_dir)
    edges = binning.constant_R_edges(*BAND, R_CMP)
    rows = []
    for planet in presets.PLANET_NAMES:
        for scen in SCENARIOS:
            row = compare(planet, scen, tdir, edges)
            rows.append(row)
            if row["T_clamped"]:
                rows.append(compare(planet, scen, tdir, edges,
                                    T_override=row["T_iso_table_K"]))
    return rows


def validate_holdouts(tables_dir=None):
    """Grid interpolation vs direct POSEIDON at the off-node holdout points.

    GZ: fiducial zeta (grid _trilinear - direct), binned to R=100, median
    removed, worst bin converted to ppm on the named planet (1 H = 2 depth
    H/R_p). G2/G5: transmission_depth at the holdout point vs the direct
    depth at the planet-like node, both scaled to the named planet's R_s so
    the ppm are on its noise table's scale (depth is exactly ~ R_s^-2)."""
    _require(HOLDOUTS, "POSEIDON holdouts")
    tdir = tables.resolve_tables_dir(tables_dir)
    g = tables.load_zeta_grid(tdir)
    axes = (g["T_K"], g["log_met"], g["c_to_o"])
    h = np.load(HOLDOUTS)
    meta = json.loads(str(h["meta"]))
    wl_h = h["wavelength_um"]                 # R=1000 bin centres
    edges_h = binning.constant_R_edges(0.35, 1.85, 1000)
    edges = binning.constant_R_edges(*BAND, R_CMP)
    cen = binning.bin_centers(edges)
    assert np.allclose(binning.bin_centers(edges_h), wl_h)

    def to_cmp(wl, y):   # native (or R=1000) -> R=1000 bins -> R=100 bins, same path for both sides
        if len(wl) != len(wl_h):
            y = binning.bin_model(wl, y, edges_h)
        return binning.bin_model(wl_h, y, edges)

    rows = []
    for c in meta["cases"]:
        lab, planet = c["label"], c["planet"]
        point = (c["T_K"], c["log_met"], c["c_to_o"])
        p = presets.showcase(planet, "clear")
        sig = _sigma(tdir, planet, edges)
        scale = 1.0 / p["R_s_RSun"] ** 2            # 1 R_sun -> the planet's star
        d_ref = h[f"{lab}__depth_p"].astype(float) * scale
        band = (wl_h >= BAND[0]) & (wl_h <= BAND[1])
        R_p_RJ = float(np.sqrt(np.median(d_ref[band])) * p["R_s_RSun"] * R_SUN_M / R_J_M)
        d_mod, info = transmission_depth(
            g["wavelength_um"], T_eq_K=c["T_K"], R_p_RJ=R_p_RJ, M_p_MJ=c["M_p_MJ"],
            R_s_RSun=p["R_s_RSun"], log_met=c["log_met"], c_to_o=c["c_to_o"],
            tables_dir=tdir)
        one_H = 2 * float(np.median(d_ref[band])) * info["H_over_Rp"]   # dzeta -> depth
        dz_worst, dz_sig = 0.0, 0.0
        for i in (0, 1):
            dz = to_cmp(g["wavelength_um"], _trilinear(g["zeta"][..., i, :], axes, point)) \
                - to_cmp(wl_h, h[f"{lab}__zeta_{i}"].astype(float))
            dz = dz - np.median(dz)
            k = int(np.argmax(np.abs(dz)))
            if abs(dz[k]) > abs(dz_worst):
                dz_worst, dz_sig = float(dz[k]), float(dz[k] * one_H / sig[k])
        row = dict(label=lab, planet=planet, T_K=c["T_K"], log_met=c["log_met"],
                   c_to_o=c["c_to_o"], x_p=c["x_p"], x_model=info["zeta_shift"],
                   mu_direct=c["mu_p"], mu_grid=info["mu_used"],
                   one_H_ppm=one_H * 1e6, dzeta_worst_H=dz_worst, dzeta_sig=dz_sig)
        row.update(_metrics(to_cmp(wl_h, d_ref), to_cmp(g["wavelength_um"], d_mod), cen, sig))
        row["gated"] = True
        rows.append(_apply_gates(row, lab, HOLDOUT_GATES))
    return rows


def summarize(rows):
    """(unwaived failures, stale waivers, failures per gate) over gated rows."""
    gated = [r for r in rows if r["gated"]]
    name = lambda r: r.get("label") or f"{r['planet']} {r['scenario']}"  # noqa: E731
    unwaived = [(name(r), g) for r in gated for g in r["unwaived"]]
    stale = [(name(r), g) for r in gated for g in r["stale_waiver"]]
    per_gate = {g: sum(g in r["failed"] for r in gated) for g in GATES}
    return unwaived, stale, per_gate


def _print_summary(title, rows):
    unwaived, stale, per_gate = summarize(rows)
    gated = [r for r in rows if r["gated"]]
    n_fail = sum(bool(r["failed"]) for r in gated)
    n_waived = sum(bool(r["failed"]) and not r["unwaived"] for r in gated)
    print(f"\n{title}: {len(gated) - n_fail}/{len(gated)} gated rows pass all gates, "
          f"{n_waived} waived, {len(unwaived)} un-waived failure(s), {len(stale)} stale waiver(s)")
    print("failures per gate: " + ", ".join(
        f"{g} {n} ({GATES[g][0]})" for g, n in per_gate.items() if n))
    for r in gated:
        key = r.get("label") or f"{r['planet']} {r['scenario']}"
        if r["failed"] and r["waived"]:
            print(f"WAIVED {key} [{' '.join(r['failed'])}]: {r['waived']}")
    for key, g in unwaived:
        print(f"FAIL {key}: {g} ({GATES[g][0]})")
    for key, g in stale:
        print(f"STALE WAIVER {key}: {g} passes now; remove it from WAIVED")
    return not unwaived and not stale


def _cell(x, fmt):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)


def _status(r):
    if not r["failed"]:
        return "ok"
    tag = "not gated" if not r["gated"] else ("waived" if not r["unwaived"] else "FAIL")
    return f"{tag} {' '.join(r['failed'])}"


def main():
    rows = validate()
    hold = validate_holdouts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"models": rows, "holdouts": hold}, indent=1))
    print("| planet | scenario | T_eq used | rms [ppm] | worst [ppm @ um] | worst/sigma "
          "| err_snr | S/N ratio | amp ratio | amp model [ppm] | gates |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        tag = " (like-for-like)" if r["like_for_like"] else ("" if r["gated"] else " (not gated)")
        print(f"| {r['planet']} | {r['scenario']} | {r['T_eq_used_K']:.0f}{tag} | "
              f"{r['rms_ppm']:.0f} | {r['worst_ppm']:+.0f} @ {r['worst_um']:.2f} | "
              f"{r['worst_sig']:+.2f} | {r['err_snr']:.1f} | {_cell(r['snr_ratio'], '.3f')} | "
              f"{_cell(r['amp_ratio'], '.3f')} | {r['amp_model_ppm']:.0f} | {_status(r)} |")
    ok = _print_summary("models", rows)
    print("\n| holdout | planet | T [K] | [M/H] | C/O | x_p | dzeta worst [H] | dzeta/sigma "
          "| worst [ppm @ um] | worst/sigma | err_snr | S/N ratio | gates |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in hold:
        print(f"| {r['label']} | {r['planet']} | {r['T_K']:.0f} | {r['log_met']:.2f} | {r['c_to_o']:.2f} | "
              f"{r['x_p']:+.2f} | {r['dzeta_worst_H']:+.3f} | {r['dzeta_sig']:+.2f} | "
              f"{r['worst_ppm']:+.0f} @ {r['worst_um']:.2f} | {r['worst_sig']:+.2f} | "
              f"{r['err_snr']:.1f} | {r['snr_ratio']:.3f} | {_status(r)} |")
    ok &= _print_summary("holdouts", hold)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
