"""Build <package>/data/tables/star/phoenix_grid.npz from the local pysynphot PHOENIX grid.

Run: PYSYN_CDBS=$HOME/inputs/stellar_grids \
     ~/anaconda3/envs/poseidon/bin/python precompute/make_phoenix_grid.py
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
from terminator import binning, tables  # noqa: E402

CDBS = Path(os.environ.get("PYSYN_CDBS", Path.home() / "inputs" / "stellar_grids"))
GRID = CDBS / "grid" / "phoenix"
TABLES = tables.resolve_tables_dir(None)
FEH = {-1.0: "phoenixm10", -0.5: "phoenixm05", 0.0: "phoenixm00",
       0.3: "phoenixp03", 0.5: "phoenixp05"}
LOGG = [3.5, 4.0, 4.5, 5.0]
TEFF_MIN, TEFF_MAX = 2000, 8000
WL_MIN_UM, WL_MAX_UM, R_STORE = 0.30, 2.50, 1000
FLOOR = 1e-30    # erg s-1 cm-2 A-1; keeps log10 finite in deep absorption
# Native sampling differs per folder: phoenixm00 is R~13,700 (103,000 points),
# the other metallicities are R~160 (5,000 points). Every spectrum is first
# interpolated onto this fine constant-R grid, then bin-averaged to R_STORE,
# so the stored grid is defined the same way everywhere. The per-folder
# native resolution is recorded in meta["native_R"].
R_FINE = 20000


def teff_of(path):
    return int(re.findall(r"_(\d+)\.fits$", path.name)[0])


def main():
    edges = binning.constant_R_edges(WL_MIN_UM, WL_MAX_UM, R_STORE)
    wl_store = binning.bin_centers(edges)
    wl_fine = binning.bin_centers(
        binning.constant_R_edges(WL_MIN_UM * 0.99, WL_MAX_UM * 1.01, R_FINE))
    native_R = {}
    filled = {}      # "folder_teff_logg" -> logg actually used (all-zero placeholder models)
    files0 = sorted(GRID.joinpath(FEH[0.0]).glob("*.fits"), key=teff_of)
    teffs = [teff_of(p) for p in files0 if TEFF_MIN <= teff_of(p) <= TEFF_MAX]
    fehs = sorted(FEH)
    log_flam = np.full((len(teffs), len(LOGG), len(fehs), len(wl_store)), np.nan)
    for k, feh in enumerate(fehs):
        for i, teff in enumerate(teffs):
            path = GRID / FEH[feh] / f"{FEH[feh]}_{teff}.fits"
            with fits.open(path) as h:
                d = h[1].data
                wl_um = np.asarray(d["WAVELENGTH"], float) / 1e4
                sel = (wl_um >= WL_MIN_UM * 0.98) & (wl_um <= WL_MAX_UM * 1.02)
                native_R.setdefault(FEH[feh], float(np.median(
                    wl_um[sel][1:] / np.diff(wl_um[sel]))))
                cols = {g: f"g{int(round(g * 10)):02d}" for g in LOGG}
                missing = [c for c in cols.values() if c not in d.columns.names]
                if missing:
                    raise SystemExit(f"{path.name}: no column(s) {missing}")
                nonzero = [g for g in LOGG if np.any(np.asarray(d[cols[g]], float)[sel] > 0)]
                if not nonzero:
                    raise SystemExit(f"{path.name}: every log g column is all-zero")
                for j, g in enumerate(LOGG):
                    g_use = g if g in nonzero else min(nonzero, key=lambda x: abs(x - g))
                    if g_use != g:
                        filled[f"{FEH[feh]}_{teff}_{g}"] = g_use
                    flux = np.asarray(d[cols[g_use]], float)[sel]
                    fine = np.interp(wl_fine, wl_um[sel], flux)
                    binned = binning.bin_model(wl_fine, fine, edges)
                    log_flam[i, j, k] = np.log10(np.maximum(binned, FLOOR))
        print(f"[Fe/H]={feh:+.1f}: {len(teffs)} T_eff files done", flush=True)
    assert np.isfinite(log_flam).all()
    tables.write_phoenix_grid(
        TABLES, teff_K=teffs, logg=LOGG, feh=fehs, wavelength_um=wl_store,
        log_flam=log_flam,
        meta={"generator": "make_phoenix_grid.py", "created": str(date.today()),
              "source": "pysynphot PHOENIX (STScI CDBS grid/phoenix)",
              "units": "log10 F_lambda [erg s-1 cm-2 A-1] at the stellar "
                       "surface, bin-averaged at constant R",
              "R_store": R_STORE, "R_fine": R_FINE,
              "native_R": {k: round(v) for k, v in native_R.items()},
              "filled_from_nearest_logg": filled,
              "folders": {str(k): v for k, v in FEH.items()}})
    tables.rebuild_index(TABLES)
    print(f"filled {len(filled)} all-zero placeholder models from the nearest log g")
    print("wrote", TABLES / tables.PHOENIX_GRID)


if __name__ == "__main__":
    main()
