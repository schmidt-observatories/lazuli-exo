"""Stage 2: per-planet single-transit depth-uncertainty tables via slicersim.

Run: ~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py

Thin loop over terminator.slicer.depth_error, which owns the physics: the
Vega -> AB conversion of the archive J magnitude (slicersim normalizes in AB),
the readout desaturation loop for bright hosts (see its docstring), and the
equal-baseline depth-error formula. Residual saturation at the ramp floor is
recorded in the table meta, never hidden.
"""
import warnings
from datetime import date
from pathlib import Path

import numpy as np

# Targeted, not blanket: dependency deprecation chatter is muted, but any
# runtime/scientific warning from slicersim (saturation, extraction, numeric)
# must reach the operator.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"

from terminator import tables  # noqa: E402
from terminator.slicer import depth_error  # noqa: E402
from terminator.targets import PLANET_NAMES, get_target  # noqa: E402

INSTRUMENT = "lazuli_cbe"
BAND = "J"            # 2MASS J; archive J magnitudes are Vega


def run_planet(name):
    t = get_target(name)
    wl_star_um, f_star, star_meta = tables.load_star(TABLES, name)
    wl_um, err, info = depth_error(wl_star_um * 1.0e4, f_star, mag=t["J_mag"],
                                   band=BAND, mag_system="vega",
                                   t14_s=t["t14_s"], instrument=INSTRUMENT)
    tables.write_noise_table(
        TABLES, name, wl_um, err,
        meta={"generator": "make_noise_tables.py", "created": str(date.today()),
              "J_mag": t["J_mag"], "norm_band": info["sncosmo_band"],
              "stellar_grid": star_meta.get("stellar_grid", "phoenix"),
              **info})
    sat = ("" if info["fully_desaturated"]
           else f"  RESIDUAL SATURATION nsat={info['nsaturated_detpx']}")
    trim = (f"  (trimmed {info['trimmed_channels']} ch)"
            if info["trimmed_channels"] else "")
    print(f"noise: {name}  J={t['J_mag']:.2f} (AB {info['mag_ab']:.2f})  "
          f"t14={t['t14_s']/3600:.2f} h  nmd={info['nmd_used']}  "
          f"median err = {np.nanmedian(err)*1e6:.0f} ppm" + trim + sat)


def main():
    for name in PLANET_NAMES:
        run_planet(name)
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main()
