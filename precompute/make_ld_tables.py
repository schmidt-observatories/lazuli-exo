"""Generate per-planet quadratic limb-darkening tables with exotic-ld.

Run (py313 env; `pip install exotic-ld`; network on first use per star —
stellar model files auto-download to $EXOTIC_LD_DATA, default
~/inputs/exotic_ld_data, and are cached):

    ~/anaconda3/envs/py313/bin/python precompute/make_ld_tables.py

Uses the PHOENIX grid — consistent with the noise tables'
`stellar_grid: phoenix`, and it covers the cool hosts (TRAPPIST-1,
2566 K) that MPS-ATLAS's 3500 K floor would clamp. Coefficients are
computed in N_CHUNK contiguous top-hat wavelength chunks across the IFS
bandpass; the runtime interpolates per band. Stellar inputs are clamped
to the grid's support where needed (e.g. GJ 1214's logg 5.03 > 5.0) and
any clamping is recorded in the table meta.
"""
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terminator import tables                    # noqa: E402
from terminator.targets import TARGETS           # noqa: E402

TABLES = Path(__file__).resolve().parents[1] / "data" / "tables"
LD_DATA = os.environ.get("EXOTIC_LD_DATA",
                         str(Path.home() / "inputs" / "exotic_ld_data"))

WL_MIN_UM, WL_MAX_UM = 0.40, 1.70
N_CHUNK = 48
LD_GRID = "phoenix"
# exotic-ld phoenix grid support (per its docs); inputs outside are clamped
GRID_BOUNDS = {"Teff": (2300.0, 12000.0), "logg": (0.0, 6.0),
               "M_H": (-4.0, 1.0)}


def clamp(name, value):
    lo, hi = GRID_BOUNDS[name]
    c = min(max(value, lo), hi)
    return c, (c != value)


def main():
    from exotic_ld import StellarLimbDarkening

    edges = np.geomspace(WL_MIN_UM, WL_MAX_UM, N_CHUNK + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    for name, t in sorted(TARGETS.items()):
        teff, c_t = clamp("Teff", t["T_eff_K"])
        logg, c_g = clamp("logg", t["log_g_star"])
        m_h, c_m = clamp("M_H", t["Met_star"] if t["Met_star"] is not None
                         else 0.0)
        sld = StellarLimbDarkening(
            M_H=m_h, Teff=teff, logg=logg, ld_model=LD_GRID,
            ld_data_path=LD_DATA, interpolate_type="trilinear", verbose=0)

        u1 = np.empty(N_CHUNK)
        u2 = np.empty(N_CHUNK)
        for i in range(N_CHUNK):
            lo_a, hi_a = edges[i] * 1e4, edges[i + 1] * 1e4
            wl_a = np.linspace(lo_a, hi_a, 32)
            (u1[i], u2[i]) = sld.compute_quadratic_ld_coeffs(
                wavelength_range=[lo_a, hi_a], mode="custom",
                custom_wavelengths=wl_a,
                custom_throughput=np.ones_like(wl_a))

        clamped = {k: v for k, v in
                   [("Teff", c_t), ("logg", c_g), ("M_H", c_m)] if v}
        tables.write_ld_table(
            TABLES, name, centers, u1, u2,
            meta={"generator": "make_ld_tables.py",
                  "created": str(date.today()),
                  "ld_grid": LD_GRID, "law": "quadratic",
                  "interpolate_type": "trilinear",
                  "exotic_ld_version": _version(),
                  "Teff_used": teff, "logg_used": logg, "M_H_used": m_h,
                  "inputs_clamped": sorted(clamped) if clamped else [],
                  "n_chunks": N_CHUNK,
                  "wl_range_um": [WL_MIN_UM, WL_MAX_UM]})
        note = f"  (clamped: {', '.join(sorted(clamped))})" if clamped else ""
        print(f"ld: {name}  Teff={teff:.0f}  logg={logg:.2f}  "
              f"u1(0.55um)={np.interp(0.55, centers, u1):.3f}  "
              f"u1(1.4um)={np.interp(1.4, centers, u1):.3f}{note}")

    tables.rebuild_index(TABLES)
    print("index rebuilt")


def _version():
    import importlib.metadata
    return importlib.metadata.version("exotic-ld")


if __name__ == "__main__":
    main()
