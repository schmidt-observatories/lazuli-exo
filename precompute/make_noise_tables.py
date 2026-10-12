"""Stage 2: per-planet single-transit depth-uncertainty tables via slicersim.

Run: ~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py

Thin loop over lazuli_exo.slicer.depth_error, which owns the physics: the
Vega -> AB conversion of the archive J magnitude (slicersim normalizes in AB),
the readout desaturation loop for bright hosts (see its docstring), and the
equal-baseline depth-error formula. Residual saturation at the ramp floor is
recorded in the table meta, never hidden.

The SED comes from star.phoenix_sed (the same solar-metallicity PHOENIX grid
simulate() uses), so observe() and simulate() share one SED by construction
(issue #34). The POSEIDON star tables (star/<slug>.npz) are validation data.
"""
import subprocess
import tomllib
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

from lazuli_exo import tables  # noqa: E402
from lazuli_exo.slicer import depth_error  # noqa: E402
from lazuli_exo.star import phoenix_sed  # noqa: E402
from lazuli_exo.targets import PLANET_NAMES, get_target  # noqa: E402

TABLES = tables.resolve_tables_dir(None)

INSTRUMENT = "lazuli_cbe"
BAND = "J"            # 2MASS J; archive J magnitudes are Vega


def provenance():
    """Package version, git commit, and sha256 of the inputs (issue #4)."""
    import slicersim
    cfg = Path(slicersim.__file__).parent / "config" / f"{INSTRUMENT}.toml"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    # code only: the tables this script writes are not "dirt"
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                                 "--", "src/lazuli_exo/*.py", "precompute"],
                                capture_output=True, text=True).stdout.strip())
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    return {"package_version": version, "git_commit": commit + ("-dirty" if dirty else ""),
            "instrument_config": cfg.name,
            "instrument_config_sha256": tables.sha256_file(cfg),
            "phoenix_grid_sha256": tables.sha256_file(TABLES / tables.PHOENIX_GRID),
            "slicersim_version": slicersim.__version__}


def run_planet(name, prov):
    t = get_target(name)
    wl_star_A, f_star, star_info = phoenix_sed(t["T_eff_K"], t["log_g_star"], t["Met_star"])
    wl_um, err, info = depth_error(wl_star_A, f_star, mag=t["J_mag"],
                                   band=BAND, mag_system="vega",
                                   t14_s=t["t14_s"], instrument=INSTRUMENT)
    tables.write_noise_table(
        TABLES, name, wl_um, err,
        meta={"generator": "make_noise_tables.py", "created": str(date.today()),
              "J_mag": t["J_mag"], "norm_band": info["sncosmo_band"],
              "stellar_grid": "phoenix_grid.npz solar [Fe/H]", "star": star_info,
              **info, **prov})
    sat = ("" if info["fully_desaturated"]
           else f"  RESIDUAL SATURATION nsat={info['nsaturated_detpx']}")
    trim = (f"  (trimmed {info['trimmed_channels']} ch)"
            if info["trimmed_channels"] else "")
    print(f"noise: {name}  J={t['J_mag']:.2f} (AB {info['mag_ab']:.2f})  "
          f"t14={t['t14_s']/3600:.2f} h  nmd={info['nmd_used']}  "
          f"median err = {np.nanmedian(err)*1e6:.0f} ppm" + trim + sat)


def main():
    prov = provenance()
    print("provenance:", {k: v for k, v in prov.items() if "sha256" not in k})
    for name in PLANET_NAMES:
        run_planet(name, prov)
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main()
