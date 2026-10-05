"""Check B: independent photon budget and achieved-precision benchmark.

B(i)  Hand photon budget vs slicersim. For each host the star table is
      normalized here to the catalog J (Vega, sncosmo), and
      N_ADU(lambda) = F_lambda * lambda/(hc) * A * eta(lambda) * QE(lambda) * dlambda * t_int * gain
      is compared channel by channel with slicersim's extracted signal for the
      same target and readout (no LSF). The ratio tests the whole normalization
      chain (magnitude system, collecting area, throughput, QE, channel width,
      integration time) independently of slicersim's own bandmag().
B(ii) Achieved precision. Every HST and JWST dataset in the archive with
      per-bin depth errors inside 0.40-1.70 um is compared with lazuli-exo's
      predicted single-transit error for the same bins (from the committed noise
      table). The ratio is listed next to the collecting-area-only expectation
      sqrt(A_other / A_Lazuli).
"""
import numpy as np

from common import (A_HST_M2, A_JWST_M2, A_LAZULI_M2, BAND_UM, DATA, HC_ERG_A, PLANET_NAMES, RESULTS,
                    TARGETS, TDIR, fnum, normalize_to_J, read_csv, star_table_sed, strip_html, tables, write_csv)


def part1():
    from slicersim.lazuli import LazuliTarget
    from lazuli_exo.slicer import vega_to_ab_offset
    offset = vega_to_ab_offset("J")
    rows = []
    for planet in PLANET_NAMES:
        wl_A, f, _ = star_table_sed(planet)
        J = TARGETS[planet]["J_mag"]              # what the noise tables used
        target = LazuliTarget(wl_A, f, mag=J + offset, band="2massj", instrument="lazuli_cbe")
        sim = target.simulation
        _, _, meta_n = tables.load_noise(TDIR, planet)
        target.change_detector(nmd=tuple(meta_n["nmd_used"]))
        lbda = np.asarray(sim.spectrograph.lbda, float)
        edges = np.asarray(sim.spectrograph.lbda_edges, float)
        dl = np.diff(edges) if edges.size == lbda.size + 1 else np.gradient(lbda)
        eta = np.asarray(sim.spectrograph.get_throughput(lbda), float)
        qe = np.asarray(sim.detector.get_qe(lbda), float)
        t_int = float(sim.detector.integration_time) * int(sim.extraction["nramps"])
        area = float(sim.telescope.surface) * 1e4                      # cm^2
        F = np.interp(lbda, wl_A, normalize_to_J(wl_A, f, J))
        hand = F * lbda / HC_ERG_A * area * eta * qe * dl * t_int * float(sim.detector.gain)
        _, sig, var = sim.get_spectrum(apply_lsf=False)
        sig, var = np.asarray(sig, float), np.asarray(var, float)
        ratio = sig / hand
        um = lbda / 1e4
        band = (um >= BAND_UM[0]) & (um <= BAND_UM[1])
        rows.append(dict(
            planet=planet, J_vega=J, J_ab=J + offset, nmd=":".join(map(str, meta_n["nmd_used"])),
            t_int_s=t_int, area_m2=area / 1e4,
            ratio_median=float(np.nanmedian(ratio[band])),
            ratio_p16=float(np.nanpercentile(ratio[band], 16)), ratio_p84=float(np.nanpercentile(ratio[band], 84)),
            ratio_blue=float(np.nanmedian(ratio[(um >= 0.40) & (um < 0.70)])),
            ratio_mid=float(np.nanmedian(ratio[(um >= 0.70) & (um < 1.20)])),
            ratio_red=float(np.nanmedian(ratio[(um >= 1.20) & (um <= 1.70)])),
            var_over_signal_median=float(np.nanmedian((var / sig)[band])),
            eta_qe_0p5=float(np.interp(5000, lbda, eta * qe)), eta_qe_1p2=float(np.interp(12000, lbda, eta * qe)),
            signal_e_per_channel_median=float(np.nanmedian(sig[band]))))
        r = rows[-1]
        print(f"{planet:13s} nmd={r['nmd']:7s} slicersim/hand = {r['ratio_median']:.3f} [{r['ratio_p16']:.3f}, {r['ratio_p84']:.3f}]"
              f"  blue {r['ratio_blue']:.3f} mid {r['ratio_mid']:.3f} red {r['ratio_red']:.3f}   var/signal {r['var_over_signal_median']:.2f}", flush=True)
    write_csv(RESULTS / "checkB_photon_budget.csv", rows)


def _bin_edges(wl, bw):
    """Full bin width per point: the archive 'bandwidth' when it looks like a
    full width, else the spacing between neighbouring centers."""
    wl = np.asarray(wl, float)
    spacing = np.gradient(wl) if wl.size > 1 else np.array([0.05])
    bw = np.asarray(bw, float)
    use_bw = np.isfinite(bw) & (bw > 0.5 * spacing) & (bw < 3 * spacing)
    width = np.where(use_bw, bw, spacing)
    return wl - width / 2, wl + width / 2


def lazuli_sigma_for_bins(planet, lo, hi):
    wl_n, e_n, _ = tables.load_noise(TDIR, planet)
    dl_ch = np.gradient(wl_n)
    out = np.empty(lo.size)
    for i, (a, b) in enumerate(zip(lo, hi)):
        m = (wl_n >= a) & (wl_n < b)
        if m.any():
            out[i] = 1.0 / np.sqrt(np.sum(1.0 / e_n[m] ** 2))
        else:                      # bin narrower than a channel: scale the nearest channel
            j = int(np.argmin(np.abs(wl_n - 0.5 * (a + b))))
            out[i] = e_n[j] * np.sqrt(dl_ch[j] / (b - a))
    return out


def datasets(kinds=("JWST", "HST"), min_points=5):
    """Archive datasets with per-bin errors inside the Lazuli band, grouped by
    (planet, reference, instrument). Yields (key, wl, bw, depth, sigma)."""
    by = {}
    for r in read_csv(DATA / "transitspec.csv"):
        wl = fnum(r["centralwavelng"])
        if not (BAND_UM[0] <= wl <= BAND_UM[1]):
            continue
        dep, dep_err = fnum(r["plntransdep"]), fnum(r["plntransdeperr1"])
        k, k_err = fnum(r["plnratror"]), fnum(r["plnratrorerr1"])
        if np.isfinite(dep) and np.isfinite(dep_err) and dep_err > 0:
            d, s = dep / 100.0, dep_err / 100.0           # archive depths are in percent
        elif np.isfinite(k) and np.isfinite(k_err) and k_err > 0:
            d, s = k * k, 2 * k * k_err
        else:
            continue
        fac = r["facility"]
        kind = "JWST" if "Webb" in fac else "HST" if "Hubble" in fac else "ground"
        if kind not in kinds:
            continue
        key = (r["plntname"], strip_html(r["plntranreflink"]), r["instrument"], kind)
        by.setdefault(key, []).append((wl, fnum(r["bandwidth"]), d, s))
    for key, pts in sorted(by.items()):
        if len(pts) >= min_points:
            pts.sort()
            yield key, *map(np.array, zip(*pts))


def part2():
    rows = []
    for (planet, ref, inst, kind), wl, bw, d, s in datasets():
        lo, hi = _bin_edges(wl, bw)
        sig_laz = lazuli_sigma_for_bins(planet, lo, hi)
        a_other = A_JWST_M2 if kind == "JWST" else A_HST_M2
        ratio = float(np.median(sig_laz / s))
        rows.append(dict(planet=planet, reference=ref, instrument=inst, facility=kind, n_bins=len(wl),
                         wl_min_um=wl.min(), wl_max_um=wl.max(), median_bin_width_um=float(np.median(hi - lo)),
                         achieved_sigma_ppm=float(np.median(s)) * 1e6,
                         lazuli_1transit_sigma_ppm=float(np.median(sig_laz)) * 1e6,
                         ratio_lazuli_over_achieved=ratio,
                         area_only_expectation=float(np.sqrt(a_other / A_LAZULI_M2)),
                         lazuli_better_than_area_scaling=bool(ratio < np.sqrt(a_other / A_LAZULI_M2))))
    write_csv(RESULTS / "checkB_achieved_precision.csv", rows)
    print(f"\n{'planet':12s} {'ref':24s} {'fac':5s} {'n':>3s} {'bin um':>7s} {'achieved':>9s} {'Lazuli 1tr':>10s} {'ratio':>6s} {'area-only':>9s}")
    for r in rows:
        print(f"{r['planet']:12s} {r['reference'][:24]:24s} {r['facility']:5s} {r['n_bins']:3d} {r['median_bin_width_um']:7.3f} "
              f"{r['achieved_sigma_ppm']:9.0f} {r['lazuli_1transit_sigma_ppm']:10.0f} {r['ratio_lazuli_over_achieved']:6.2f} "
              f"{r['area_only_expectation']:9.2f}{'  <-- below area scaling' if r['lazuli_better_than_area_scaling'] else ''}")


if __name__ == "__main__":
    part1()
    part2()
