"""Stage 1: POSEIDON model spectra + PHOENIX host spectra for all targets.

Mirrors the verified recipe in precompute/smoke_poseidon.py exactly (Task 8):
chemistry_grid loaded once and passed to every make_atmosphere call, P built
descending (np.logspace(hi, lo, n)), inputs at $HOME/inputs/.

Run:
  POSEIDON_input_data=$HOME/inputs/ \
  PYSYN_CDBS=$HOME/inputs/stellar_grids \
  ~/anaconda3/envs/poseidon/bin/python precompute/make_model_tables.py [planet ...]
"""
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

os.environ.setdefault("POSEIDON_input_data",
                      str(Path.home() / "inputs") + "/")
os.environ.setdefault("PYSYN_CDBS",
                      str(Path.home() / "inputs" / "stellar_grids"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))      # lazuli_exo not pip-installed here
sys.path.insert(0, str(Path(__file__).parent))

from scenarios import (P_GRID_ARGS, P_REF_BAR, PARAM_SPECIES, SCENARIOS,
                       make_wl_grid)
from lazuli_exo import tables
from lazuli_exo.targets import PLANET_NAMES, get_target

from POSEIDON.chemistry import load_chemistry_grid
from POSEIDON.constants import M_J, R_J, R_Sun
from POSEIDON.core import (compute_spectrum, create_planet, create_star,
                           define_model, make_atmosphere, read_opacities)

TABLES = tables.resolve_tables_dir(None)
WL = make_wl_grid()

# Loaded once at script start (chem_eq requires this be passed explicitly
# into every make_atmosphere call -- see task-8-report.md Bug A).
CHEMISTRY_GRID = load_chemistry_grid(PARAM_SPECIES, grid="fastchem")

# The fastchem grid's own temperature axis has a hard floor (300 K) below
# which POSEIDON.chemistry.interpolate_log_X_grid raises "Requested
# temperature is out of the grid bounds." Several cool targets (K2-18 b
# T_eq~284K, LHS 1140 b~226K, TRAPPIST-1 e~249.7K) have T_eq below this --
# discovered when the full run crashed at K2-18 b. Fix: clamp the isotherm
# temperature used for both PT_params and T_fine to this floor (plus a tiny
# epsilon, since the grid check is `T < grid[0]` -- exact equality is valid,
# but a small margin avoids relying on that). This is a deliberate, disclosed
# approximation for a showcase tool, not a claim that the true equilibrium
# chemistry at ~230-290K equals that at 300K.
T_GRID_MIN = float(CHEMISTRY_GRID["T_grid"].min())
T_CHEM_EPS = 1e-3


def star_spectrum(t):
    """PHOENIX stellar spectrum, falling back to blackbody for very cool
    stars outside the PHOENIX grid range (e.g. TRAPPIST-1, T_eff~2566K)."""
    try:
        star = create_star(t["R_s_RSun"] * R_Sun, t["T_eff_K"],
                           t["log_g_star"], t["Met_star"],
                           stellar_grid="phoenix", wl=WL,
                           interp_backend="pysynphot")
        grid_used = "phoenix"
    except Exception as exc:
        # Only the documented out-of-grid failure (observed for TRAPPIST-1,
        # T_eff ~ 2566 K below the PHOENIX grid) may fall back to a
        # blackbody. Anything else (I/O, missing inputs, install problems)
        # must fail loudly, not silently swap the stellar model.
        msg = str(exc).lower()
        if not any(s in msg for s in ("out of", "bounds", "grid", "exceed")):
            raise
        print(f"  PHOENIX failed for T_eff={t['T_eff_K']}K ({exc!r}); "
              "falling back to blackbody")
        star = create_star(t["R_s_RSun"] * R_Sun, t["T_eff_K"],
                           t["log_g_star"], t["Met_star"],
                           stellar_grid="blackbody", wl=WL)
        grid_used = "blackbody"
    return star, np.asarray(star["wl_star"]), np.asarray(star["F_star"]), grid_used


def run_planet(name):
    t = get_target(name)
    planet = create_planet(tables.slugify(name), t["R_p_RJ"] * R_J,
                           mass=t["M_p_MJ"] * M_J)
    star, wl_s, f_s, grid_used = star_spectrum(t)
    tables.write_star_table(TABLES, name, wl_s, f_s,
                            meta={"units": "POSEIDON F_star (W/m^2/m)",
                                  "stellar_grid": grid_used,
                                  "T_eff_K": t["T_eff_K"]})
    print(f"star: {name}  grid={grid_used}")

    # POSEIDON expects P ordered deep -> shallow (descending); see
    # task-8-report.md Bug B / smoke_poseidon.py for why ascending silently
    # zeroes ~99/100 layers.
    lo, hi, n = P_GRID_ARGS
    P = np.logspace(hi, lo, n)
    T_eq = t["T_eq_K"]
    T_clamped = T_eq < T_GRID_MIN
    T_iso = (T_GRID_MIN + T_CHEM_EPS) if T_clamped else T_eq
    if T_clamped:
        print(f"CLAMPED: {name} T_eq={T_eq:.1f} -> T_iso={T_iso:.3f}")
    T_fine = np.linspace(max(T_iso - 200, 100), T_iso + 200, 10)
    log_P_fine = np.linspace(-6.0, 2.0, 40)

    depth_lo, depth_hi = 0.5 * t["depth"], 1.5 * t["depth"]
    opac = None
    for scen, cfg in SCENARIOS.items():
        model = define_model(f"{tables.slugify(name)}_{scen}",
                             bulk_species=["H2", "He"],
                             param_species=PARAM_SPECIES,
                             PT_profile="isotherm", X_profile="chem_eq",
                             cloud_model=cfg["cloud_model"],
                             **({"cloud_type": cfg["cloud_type"]}
                                if cfg["cloud_type"] else {}))
        if opac is None:   # species identical across scenarios -> reuse
            opac = read_opacities(model, WL,
                                  opacity_treatment="opacity_sampling",
                                  T_fine=T_fine, log_P_fine=log_P_fine)
        atmosphere = make_atmosphere(
            planet, model, P, P_REF_BAR, t["R_p_RJ"] * R_J,
            PT_params=[T_iso],
            log_X_params=[cfg["C_to_O"], cfg["log_Met"]],
            cloud_params=cfg["cloud_params"],
            chemistry_grid=CHEMISTRY_GRID)
        spectrum = compute_spectrum(planet, star, model, atmosphere, opac, WL,
                                    spectrum_type="transmission")
        assert np.isfinite(spectrum).all(), (name, scen, "non-finite spectrum")

        spec_min, spec_max = float(spectrum.min()), float(spectrum.max())
        if scen == "clear" and (spec_max - spec_min) < 10e-6:
            raise RuntimeError(
                f"FLAT spectrum for {name}/clear "
                f"(max-min={spec_max - spec_min:.2e} < 10ppm) -- "
                "this is the ascending-P bug signature; STOP.")
        if not (depth_lo <= spec_min <= depth_hi or
                depth_lo <= spec_max <= depth_hi or
                (spec_min <= depth_lo and spec_max >= depth_hi)):
            print(f"  WARNING: {name}/{scen} depth range "
                  f"[{spec_min:.5f}, {spec_max:.5f}] outside "
                  f"±50% of white-light depth {t['depth']:.5f} "
                  f"([{depth_lo:.5f}, {depth_hi:.5f}])")

        meta = {"generator": "make_model_tables.py",
                "created": str(date.today()),
                "T_eq_K": T_eq, "T_iso_K": T_iso, "T_clamped": T_clamped,
                **{k: v for k, v in cfg.items()}}
        if T_clamped:
            meta["note"] = "isothermal T clamped to fastchem grid floor"
        tables.write_model_table(TABLES, name, scen, WL, spectrum, meta=meta)
        print(f"model: {name} / {scen}  "
              f"depth {spec_min:.5f}-{spec_max:.5f}")


def main(argv):
    names = argv or PLANET_NAMES
    for name in names:
        run_planet(name)
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main(sys.argv[1:])
