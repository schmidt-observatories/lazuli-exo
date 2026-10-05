"""Check A: stellar SEDs vs Gaia DR3 XP spectra and SIMBAD photometry.

Both model stars used by lazuli-exo are tested: the committed star table
(POSEIDON PHOENIX, or blackbody for TRAPPIST-1) that fed the noise tables, and
the PHOENIX mini-grid the parametric path uses. Each is normalized to the
catalog 2MASS J (Vega) exactly as slicersim normalizes it, then compared with
the absolute XP flux in 50 nm bins over 0.40-1.02 um. The ratio model/XP is the
photon-count error Lazuli's blue channels would inherit from the SED.
"""
import numpy as np

from common import (DATA, PLANET_NAMES, J_vega, fnum, host, normalize_to_J,
                    phoenix_grid_sed, photometry, read_csv, star_table_sed,
                    synth_mag, tables, write_csv)

BANDS = {"V": "bessellv", "J": "2massj", "H": "2massh", "K": "2massks"}
EDGES_NM = np.arange(400, 1051, 50)
W_PER_M2_NM_TO_CGS = 100.0          # 1 W m-2 nm-1 = 100 erg s-1 cm-2 A-1


def xp_spectrum(planet):
    p = DATA / "gaia_xp" / f"{tables.slugify(host(planet))}.csv"
    if not p.exists():
        return None
    r = read_csv(p)
    return (np.array([fnum(x["wavelength_nm"]) for x in r]) * 10.0,
            np.array([fnum(x["flux_W_m2_nm"]) for x in r]) * W_PER_M2_NM_TO_CGS,
            np.array([fnum(x["flux_err_W_m2_nm"]) for x in r]) * W_PER_M2_NM_TO_CGS)


def colours(wl_A, f, obs):
    out = {}
    for b, sb in BANDS.items():
        try:
            m = synth_mag(wl_A, f, sb)
        except Exception:
            m = np.nan
        out[f"model_{b}"] = m
    for a, b in (("V", "J"), ("J", "H"), ("H", "K")):
        mo = out[f"model_{a}"] - out[f"model_{b}"]
        ob = fnum(obs.get(a)) - fnum(obs.get(b))
        out[f"d({a}-{b})"] = mo - ob
    return out


def main():
    phot = photometry()
    rows = []
    for planet in PLANET_NAMES:
        J = J_vega(planet, phot)
        xp = xp_spectrum(planet)
        for label, (wl_A, f, meta) in (("table", star_table_sed(planet)),
                                       ("phoenix_grid", phoenix_grid_sed(planet))):
            f = normalize_to_J(wl_A, f, J)
            row = dict(planet=planet, host=host(planet), model=label,
                       grid=meta.get("stellar_grid", meta.get("grid", "?")),
                       Teff=float(meta.get("T_eff_K", meta.get("teff_K", np.nan))),
                       J_used=J, **colours(wl_A, f, phot.get(planet, {})))
            if xp is not None:
                wx, fx, ex = xp
                fm = np.interp(wx, wl_A, f)
                ratio = []
                for lo, hi in zip(EDGES_NM[:-1], EDGES_NM[1:]):
                    m = (wx >= lo * 10) & (wx < hi * 10)
                    ratio.append(fm[m].sum() / fx[m].sum() if m.any() else np.nan)
                ratio = np.array(ratio)
                row.update({f"r{lo}": r for lo, r in zip(EDGES_NM[:-1], ratio)})
                blue = (wx >= 4000) & (wx <= 7000)
                red = (wx >= 7000) & (wx <= 10200)
                row["photon_ratio_0.40-0.70"] = float(np.sum(fm[blue] * wx[blue]) / np.sum(fx[blue] * wx[blue]))
                row["photon_ratio_0.70-1.02"] = float(np.sum(fm[red] * wx[red]) / np.sum(fx[red] * wx[red]))
                row["max_abs_dev_50nm"] = float(np.nanmax(np.abs(ratio - 1)))
            rows.append(row)
    write_csv(DATA.parent / "results" / "checkA_sed.csv", rows)

    print(f"\n{'host':11s} {'model':13s} {'d(V-J)':>7s} {'d(J-H)':>7s} {'blue 0.40-0.70':>15s} {'red 0.70-1.02':>14s} {'max|dev| 50nm':>14s}")
    for r in rows:
        print(f"{r['host']:11s} {r['model']:13s} {r['d(V-J)']:7.2f} {r['d(J-H)']:7.2f} "
              f"{r.get('photon_ratio_0.40-0.70', np.nan):15.2f} {r.get('photon_ratio_0.70-1.02', np.nan):14.2f} "
              f"{r.get('max_abs_dev_50nm', np.nan):14.2f}")



if __name__ == "__main__":
    main()
