"""Check C: white-light transit geometry vs per-reference archive solutions and batman.

Part 1 (registry vs references): every NASA Exoplanet Archive 'ps' row with
(k, a/R*, i) for the 14 targets is compared with the frozen registry; t14 is
recomputed from each solution's own geometry with lazuli_exo.orbit, and the
archive's own pl_trandur is listed where present.

Part 2 (light-curve model vs batman): the numpy transit solver is compared with
batman at identical separations, and lazuli_exo's local-arc z(t) with batman's
Keplerian z(t), per target (as in the 2026-09-26 audit).
"""
import math

import numpy as np

from common import DATA, PLANET_NAMES, RESULTS, TARGETS, TDIR, fnum, read_csv, strip_html, tables, write_csv
from lazuli_exo.orbit import transit_duration_s


def impact_parameter(a_rs, inc_deg, e, w_deg):
    w = math.radians(w_deg)
    return a_rs * math.cos(math.radians(inc_deg)) * (1 - e * e) / (1 + e * math.sin(w))


def part1():
    rows = []
    for r in read_csv(DATA / "ps_solutions.csv"):
        k, a_rs, inc = fnum(r["pl_ratror"]), fnum(r["pl_ratdor"]), fnum(r["pl_orbincl"])
        if not all(np.isfinite([k, a_rs, inc])):
            continue
        t = TARGETS[r["pl_name"]]
        e = fnum(r["pl_orbeccen"]); e = 0.0 if not np.isfinite(e) else e
        w = fnum(r["pl_orblper"]); w = 90.0 if not np.isfinite(w) else w
        per = fnum(r["pl_orbper"]); per = t["period_d"] if not np.isfinite(per) else per
        t14_geo = transit_duration_s(per, a_rs, k, inc, e, w)
        t14_arch = fnum(r["pl_trandur"]) * 3600.0
        k_reg = math.sqrt(t["depth"])
        b_reg = impact_parameter(t["a_over_rs"], t["inc_deg"], t["ecc"], t["w_deg"])
        rows.append(dict(
            planet=r["pl_name"], reference=strip_html(r["pl_refname"]), pubdate=r["pl_pubdate"],
            k=k, k_err=fnum(r["pl_ratrorerr1"]), a_rs=a_rs, inc_deg=inc, ecc=e, w_deg=w,
            b=impact_parameter(a_rs, inc, e, w), t14_geometry_s=t14_geo, t14_archive_s=t14_arch,
            k_over_registry=k / k_reg, a_rs_over_registry=a_rs / t["a_over_rs"],
            b_minus_registry=impact_parameter(a_rs, inc, e, w) - b_reg,
            t14_geometry_over_registry_t14=t14_geo / t["t14_s"],
            t14_archive_over_registry_t14=t14_arch / t["t14_s"] if np.isfinite(t14_arch) else np.nan,
            depth_over_registry=(k / k_reg) ** 2))
    write_csv(RESULTS / "checkC_solutions.csv", rows)
    reg = []
    for p in PLANET_NAMES:
        t = TARGETS[p]
        t14g = transit_duration_s(t["period_d"], t["a_over_rs"], math.sqrt(t["depth"]), t["inc_deg"], t["ecc"], t["w_deg"])
        ks = [r["k"] for r in rows if r["planet"] == p]
        reg.append(dict(planet=p, registry_t14_s=t["t14_s"], t14_from_registry_geometry_s=t14g, ratio=t14g / t["t14_s"],
                        b=impact_parameter(t["a_over_rs"], t["inc_deg"], t["ecc"], t["w_deg"]),
                        n_reference_solutions=len(ks),
                        k_spread_over_k=(max(ks) - min(ks)) / math.sqrt(t["depth"]) if ks else np.nan,
                        depth_spread_ppm=((max(ks) ** 2 - min(ks) ** 2) * 1e6) if ks else np.nan))
    write_csv(RESULTS / "checkC_registry.csv", reg)
    print(f"\n{'planet':13s} {'t14geo/t14':>10s} {'b':>6s} {'nref':>4s} {'dk/k':>6s} {'ddepth ppm':>10s}  latest ref: k/k_reg aRs/aRs_reg t14geo/t14_reg")
    for g in reg:
        mine = [r for r in rows if r["planet"] == g["planet"]]
        last = mine[-1] if mine else None
        tail = (f"{last['reference'][:22]:22s} {last['k_over_registry']:.3f} {last['a_rs_over_registry']:.3f} "
                f"{last['t14_geometry_over_registry_t14']:.3f}" if last else "-")
        print(f"{g['planet']:13s} {g['ratio']:10.3f} {g['b']:6.3f} {g['n_reference_solutions']:4d} {g['k_spread_over_k']:6.3f} {g['depth_spread_ppm']:10.0f}  {tail}")


def part2():
    import batman
    import lazuli_exo
    from lazuli_exo.lightcurve import transit_flux
    rows = []
    for n in PLANET_NAMES:
        t = TARGETS[n]
        wl, u1, u2, _ = tables.load_ld(TDIR, n)
        k = math.sqrt(t["depth"])
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc = 0.0, t["period_d"], k, t["a_over_rs"], t["inc_deg"]
        p.ecc, p.w, p.limb_dark = t["ecc"], t["w_deg"], "quadratic"
        tt = np.linspace(-t["t14_s"], t["t14_s"], 801) / 86400
        worst = 0.0
        for i in range(0, len(wl), 6):
            p.u = [u1[i], u2[i]]
            mb = batman.TransitModel(p, tt)
            fb = mb.light_curve(p)
            worst = max(worst, float(np.max(np.abs(transit_flux(mb.ds, k, u1[i], u2[i]) - fb))))
        o = lazuli_exo.observe(n, "flat", R=30)
        lc = o.light_curves(1, seed=0, max_points=801)
        p.u = [float(lc.u1[0]), float(lc.u2[0])]
        p.rp = float(np.sqrt(lc.depth[0]))
        mb = batman.TransitModel(p, lc.time_hr / 24)
        fb = mb.light_curve(p)
        intr = lc.time_hr[fb < 1 - 1e-9]
        dur = (intr.max() - intr.min()) * 3600 if intr.size else 0.0
        mu = np.linspace(0, 1, 201)
        I = 1 - u1[:, None] * (1 - mu) - u2[:, None] * (1 - mu) ** 2
        bad = (I < 0).any(1)
        sigma_white = float(1.0 / np.sqrt(np.sum(1.0 / o.depth_err ** 2)))   # one-transit white-light depth error
        rows.append(dict(planet=n, solver_vs_batman_max_ppm=worst * 1e6,
                         local_arc_vs_kepler_max_ppm=float(np.max(np.abs(lc.flux_model[0] - fb))) * 1e6,
                         white_depth_ppm=float(lc.depth[0]) * 1e6, sigma_point_ppm=float(lc.sigma_point[0]) * 1e6,
                         sigma_white_depth_1transit_ppm=sigma_white * 1e6,
                         arc_over_sigma_point=float(np.max(np.abs(lc.flux_model[0] - fb))) / float(lc.sigma_point[0]),
                         ld_negative_wl_range_um=(f"{wl[bad].min():.2f}-{wl[bad].max():.2f}" if bad.any() else ""),
                         batman_t14_over_registry=dur / t["t14_s"],
                         ld_chunks_negative_intensity=int(np.sum((I < 0).any(1))),
                         ld_u1_plus_u2_max=float(np.max(u1 + u2))))
    write_csv(RESULTS / "checkC_lightcurve.csv", rows)
    print(f"\n{'planet':13s} {'solver ppm':>10s} {'arc ppm':>8s} {'arc/sig_pt':>10s} {'sig_pt ppm':>10s} {'sig_white ppm':>13s} {'t14 bat/reg':>11s} {'LD I<0':>6s} range")
    for r in rows:
        print(f"{r['planet']:13s} {r['solver_vs_batman_max_ppm']:10.2f} {r['local_arc_vs_kepler_max_ppm']:8.1f} {r['arc_over_sigma_point']:10.2f} "
              f"{r['sigma_point_ppm']:10.0f} {r['sigma_white_depth_1transit_ppm']:13.1f} {r['batman_t14_over_registry']:11.3f} {r['ld_chunks_negative_intensity']:6d} {r['ld_negative_wl_range_um']}")


if __name__ == "__main__":
    part1()
    part2()
