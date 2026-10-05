"""Check G: stellar-contamination bound from measured JWST epoch variability (TRAPPIST-1 e).

Uses the four JWST/NIRSpec PRISM transmission spectra of TRAPPIST-1 e from
Espinoza et al. (2025; Zenodo 16125662, 'Espinoza' reduction). The
visit-to-visit scatter of the depth at each wavelength, in excess of the
per-visit errors, bounds the transit-light-source term that lazuli-exo does not
model; it is compared with lazuli-exo's predicted TRAPPIST-1 e feature amplitudes
and single-transit errors over the same wavelengths.
"""
import numpy as np

from common import RESULTS, TDIR, tables, write_csv
from checkD_spectra import trappist1e_datasets, models

PLANET = "TRAPPIST-1 e"


def main():
    visits = [(k[1], wl, d, s) for k, wl, bw, d, s in trappist1e_datasets() if "PRISM" in k[2]]
    wl0 = visits[0][1]
    D = np.vstack([np.interp(wl0, wl, d) for _, wl, d, s in visits])
    S = np.vstack([np.interp(wl0, wl, s) for _, wl, d, s in visits])
    mean, std = D.mean(0), D.std(0, ddof=1)
    excess = np.sqrt(np.clip(std ** 2 - np.mean(S ** 2, 0), 0, None))
    offsets = np.array([np.average(D[i] - mean, weights=1 / S[i] ** 2) for i in range(len(visits))])
    wl_n, e_n, _ = tables.load_noise(TDIR, PLANET)
    rows = []
    for lo, hi in ((0.6, 1.0), (1.0, 1.7), (0.6, 1.7)):
        m = (wl0 >= lo) & (wl0 <= hi)
        amp = {}
        for scen in ("clear", "cloudy"):
            wl_m, dt, dp = models(PLANET, scen)
            mm = (wl_m >= lo) & (wl_m <= hi)
            amp[f"table_{scen}_amp_ppm"] = float(np.ptp(dt[mm])) * 1e6
            amp[f"param_{scen}_amp_ppm"] = float(np.ptp(dp[mm])) * 1e6
        mn = (wl_n >= lo) & (wl_n <= hi)
        rows.append(dict(band_um=f"{lo}-{hi}", n_wl=int(m.sum()),
                         median_visit_sigma_ppm=float(np.median(S[:, m])) * 1e6,
                         median_visit_to_visit_std_ppm=float(np.median(std[m])) * 1e6,
                         median_excess_scatter_ppm=float(np.median(excess[m])) * 1e6,
                         p84_excess_scatter_ppm=float(np.percentile(excess[m], 84)) * 1e6,
                         visit_offset_spread_ppm=float(np.ptp(offsets)) * 1e6,
                         lazuli_1transit_sigma_ppm_median_channel=float(np.median(e_n[mn])) * 1e6,
                         **amp))
    write_csv(RESULTS / "checkG_tlse.csv", rows)
    for r in rows:
        print(r)

if __name__ == "__main__":
    main()
