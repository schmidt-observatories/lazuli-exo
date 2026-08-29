"""Stage 2: per-planet single-transit depth-uncertainty tables via slicersim.

Run: ~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py

For bright hosts, slicersim's *default* readout (a single long ramp, fixed
regardless of source brightness) partially saturates the point-source PSF
core, which silently degrades the extracted SNR instead of flagging bad
channels. Before reading off SNR(lambda) for a target, this script tunes the
detector's MACC readout mode (`nmd = (ngroups, nframes_per_group, ndrops)`)
down via `LazuliTarget.change_detector(nmd=...)` until
`sim.detector.nsaturated_detpx == 0`, first shrinking the number of groups,
then (if still saturated at the floor) the frames per group, floored at the
detector's `min_group` / 1 frame. `max_group` itself has no setter in
slicersim (read-only property; confirmed empirically), so `nmd` is the only
usable knob. If a star cannot be fully de-saturated even at the minimal ramp
(2 groups x 1 frame), the residual `nsaturated_detpx` is recorded in the
table's meta and the minimal-ramp config is used anyway (no silent fallback
to the saturated default).
"""
import warnings
from datetime import date
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"

from slicersim.lazuli import LazuliTarget

from terminator import tables
from terminator.targets import PLANET_NAMES, get_target

INSTRUMENT = "lazuli_cbe"
NORM_BAND = "2massj"      # sncosmo bandpass for J-mag normalization
ANGSTROM_PER_UM = 1.0e4


def desaturate(target):
    """Shrink the detector's MACC ramp until the point source stops
    saturating (or the ramp floor is reached).

    Returns
    -------
    snr : ndarray
        `sim.get_snr()` evaluated at the final (desaturated, or floor)
        readout config.
    nmd : tuple
        Final (ngroups, nframes_per_group, ndrops) used.
    nsaturated : int
        Residual `sim.detector.nsaturated_detpx` at the final config (0 if
        fully desaturated).
    """
    sim = target.simulation
    n, m, d = sim.detector.nmd
    min_group = sim.detector.min_group

    snr = np.asarray(sim.get_snr(), dtype=float)
    nsat = sim.detector.nsaturated_detpx or 0

    # Phase 1: shrink the number of groups (n) first.
    while nsat > 0 and n > min_group:
        n = max(min_group, n // 2)
        target.change_detector(nmd=(n, m, d))
        snr = np.asarray(sim.get_snr(), dtype=float)
        nsat = sim.detector.nsaturated_detpx or 0

    # Phase 2: at the group floor, shrink frames-per-group (m) too.
    while nsat > 0 and m > 1:
        m = max(1, m // 2)
        target.change_detector(nmd=(n, m, d))
        snr = np.asarray(sim.get_snr(), dtype=float)
        nsat = sim.detector.nsaturated_detpx or 0

    return snr, (n, m, d), int(nsat)


def run_planet(name):
    t = get_target(name)
    wl_star_um, f_star, star_meta = tables.load_star(TABLES, name)

    target = LazuliTarget(wl_star_um * ANGSTROM_PER_UM, f_star,
                          mag=t["J_mag"], band=NORM_BAND,
                          instrument=INSTRUMENT)
    sim = target.simulation
    default_nmd = tuple(sim.detector.nmd)

    snr_default, nmd_used, nsaturated = desaturate(target)
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
              "default_nmd": list(default_nmd),
              "nmd_used": list(nmd_used),
              "nramps": sim.extraction.get("nramps"),
              "t_obs_s": t_obs,
              "nsaturated_detpx": nsaturated,
              "fully_desaturated": nsaturated == 0,
              "stellar_grid": star_meta.get("stellar_grid", "phoenix"),
              "assumption": "equal out-of-transit baseline; SNR ~ sqrt(t); "
                            "readout tuned per-target to remove point-source "
                            "saturation before reading SNR",
              **meta_extra})
    sat_note = "" if nsaturated == 0 else f"  RESIDUAL SATURATION nsat={nsaturated}"
    print(f"noise: {name}  J={t['J_mag']:.2f}  t14={t['t14_s']/3600:.2f} h  "
          f"nmd={nmd_used}  t_obs={t_obs:.3f} s  "
          f"median err = {np.nanmedian(depth_err)*1e6:.0f} ppm"
          + (f"  (trimmed {meta_extra['trimmed_channels']} ch)"
             if meta_extra else "")
          + sat_note)


def main():
    for name in PLANET_NAMES:
        run_planet(name)
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main()
