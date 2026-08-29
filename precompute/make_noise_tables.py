"""Stage 2: per-planet single-transit depth-uncertainty tables via slicersim.

Run: ~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py
"""
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"

from slicersim.lazuli import LazuliTarget

from slixer import tables
from slixer.targets import PLANET_NAMES, get_target

INSTRUMENT = "lazuli_cbe"
NORM_BAND = "2massj"      # sncosmo bandpass for J-mag normalization
ANGSTROM_PER_UM = 1.0e4


def run_planet(name):
    t = get_target(name)
    wl_star_um, f_star, star_meta = tables.load_star(TABLES, name)

    target = LazuliTarget(wl_star_um * ANGSTROM_PER_UM, f_star,
                          mag=t["J_mag"], band=NORM_BAND,
                          instrument=INSTRUMENT)
    sim = target.simulation
    snr_default = np.asarray(sim.get_snr(), dtype=float)
    wl_um = np.asarray(sim.get_lbda(), dtype=float) / ANGSTROM_PER_UM
    t_obs = float(sim.observing_time)

    snr_in = snr_default * np.sqrt(t["t14_s"] / t_obs)
    depth_err = np.sqrt(2.0) / snr_in

    meta_extra = {}
    good = np.isfinite(snr_in) & (snr_in > 0) & np.isfinite(depth_err)
    if not np.all(good):
        n_bad = int(np.size(good) - np.count_nonzero(good))
        bad_wl = wl_um[~good]
        meta_extra["trimmed_channels"] = n_bad
        meta_extra["trimmed_wavelength_range_um"] = [
            float(np.min(bad_wl)), float(np.max(bad_wl))]
        wl_um = wl_um[good]
        depth_err = depth_err[good]

    tables.write_noise_table(
        TABLES, name, wl_um, depth_err,
        meta={"generator": "make_noise_tables.py", "created": str(date.today()),
              "instrument": INSTRUMENT, "norm_band": NORM_BAND,
              "J_mag": t["J_mag"], "t14_s": t["t14_s"],
              "t_obs_default_s": t_obs,
              "stellar_grid": star_meta.get("stellar_grid", "phoenix"),
              "assumption": "equal out-of-transit baseline; SNR ~ sqrt(t)",
              **meta_extra})
    print(f"noise: {name}  J={t['J_mag']:.2f}  t14={t['t14_s']/3600:.2f} h  "
          f"median err = {np.nanmedian(depth_err)*1e6:.0f} ppm"
          + (f"  (trimmed {meta_extra['trimmed_channels']} ch)"
             if meta_extra else ""))


def main():
    for name in PLANET_NAMES:
        run_planet(name)
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main()
