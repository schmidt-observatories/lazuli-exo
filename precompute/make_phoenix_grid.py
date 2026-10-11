"""Build <package>/data/tables/star/phoenix_grid.npz from the local pysynphot PHOENIX grid.

Run: PYSYN_CDBS=$HOME/inputs/stellar_grids PYTHONPATH=src python precompute/make_phoenix_grid.py

Solar metallicity only (issue #34): the non-solar pysynphot folders
(phoenixm10/m05/p03/p05) are a different, low-resolution (R~741) product with
randomly missing log g columns and cool-star spectra that are wrong by 1-3 mag
in V-J; Gaia XP prefers the solar node for every metal-rich FGK host. The
[Fe/H] axis therefore has the single node 0.0 and star.phoenix_sed clamps to it.
"""
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lazuli_exo import binning, tables  # noqa: E402

CDBS = Path(os.environ.get("PYSYN_CDBS", Path.home() / "inputs" / "stellar_grids"))
FOLDER = "phoenixm00"
GRID = CDBS / "grid" / "phoenix" / FOLDER
TABLES = tables.resolve_tables_dir(None)
LOGG = [3.5, 4.0, 4.5, 5.0, 5.5]
TEFF_MIN, TEFF_MAX = 2000, 12000    # catalog spacing: 100 K to 7000, 200 K to 12000
WL_MIN_UM, WL_MAX_UM, R_STORE = 0.30, 2.50, 1000
FLOOR = 1e-30    # erg s-1 cm-2 A-1; keeps log10 finite in deep absorption
# Native sampling is R~13,700; every spectrum is first interpolated onto this
# fine constant-R grid, then bin-averaged to R_STORE.
R_FINE = 20000


def teff_of(path):
    return int(re.findall(r"_(\d+)\.fits$", path.name)[0])


def main():
    edges = binning.constant_R_edges(WL_MIN_UM, WL_MAX_UM, R_STORE)
    wl_store = binning.bin_centers(edges)
    wl_fine = binning.bin_centers(
        binning.constant_R_edges(WL_MIN_UM * 0.99, WL_MAX_UM * 1.01, R_FINE))
    native_R = None
    filled = {}      # "teff_logg" -> logg actually used (missing or all-zero column)
    files = sorted(GRID.glob("*.fits"), key=teff_of)
    teffs = [teff_of(p) for p in files if TEFF_MIN <= teff_of(p) <= TEFF_MAX]
    cols = {g: f"g{int(round(g * 10)):02d}" for g in LOGG}
    log_flam = np.full((len(teffs), len(LOGG), 1, len(wl_store)), np.nan)
    for i, teff in enumerate(teffs):
        with fits.open(GRID / f"{FOLDER}_{teff}.fits") as h:
            d = h[1].data
            wl_um = np.asarray(d["WAVELENGTH"], float) / 1e4
            sel = (wl_um >= WL_MIN_UM * 0.98) & (wl_um <= WL_MAX_UM * 1.02)
            if native_R is None:
                native_R = float(np.median(wl_um[sel][1:] / np.diff(wl_um[sel])))
            # The hot files drop the high-log g columns (g55 above 5000 K, g50
            # above 10000 K); treat a missing column like an all-zero one.
            nonzero = [g for g in LOGG if cols[g] in d.columns.names
                       and np.any(np.asarray(d[cols[g]], float)[sel] > 0)]
            if not nonzero:
                raise SystemExit(f"{FOLDER}_{teff}.fits: every log g column is missing or all-zero")
            for j, g in enumerate(LOGG):
                g_use = g if g in nonzero else min(nonzero, key=lambda x: abs(x - g))
                if g_use != g:
                    filled[f"{teff}_{g}"] = g_use
                flux = np.asarray(d[cols[g_use]], float)[sel]
                fine = np.interp(wl_fine, wl_um[sel], flux)
                binned = binning.bin_model(wl_fine, fine, edges)
                log_flam[i, j, 0] = np.log10(np.maximum(binned, FLOOR))
    assert np.isfinite(log_flam).all()
    tables.write_phoenix_grid(
        TABLES, teff_K=teffs, logg=LOGG, feh=[0.0], wavelength_um=wl_store,
        log_flam=log_flam,
        meta={"generator": "make_phoenix_grid.py", "created": str(date.today()),
              "source": f"pysynphot PHOENIX (STScI CDBS grid/phoenix/{FOLDER})",
              "folder": FOLDER,
              "units": "log10 F_lambda [erg s-1 cm-2 A-1] at the stellar "
                       "surface, bin-averaged at constant R",
              "R_store": R_STORE, "R_fine": R_FINE, "native_R": round(native_R),
              "filled_from_nearest_logg": filled,
              "metallicity": "solar by construction: the non-solar pysynphot "
                             "folders are an unreliable low-resolution product "
                             "(issue #34); phoenix_sed clamps [Fe/H] to 0.0"})
    tables.rebuild_index(TABLES)
    print(f"{len(teffs)} T_eff x {len(LOGG)} log g nodes, {teffs[0]}-{teffs[-1]} K, native R~{native_R:.0f}")
    print(f"filled {len(filled)} missing/all-zero log g columns from the nearest log g: "
          + ", ".join(f"{k}<-{v}" for k, v in filled.items()))
    print("wrote", TABLES / tables.PHOENIX_GRID)


if __name__ == "__main__":
    main()
