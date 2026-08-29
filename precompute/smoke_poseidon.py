"""Smoke test: one POSEIDON forward model (WASP-39 b, clear) + grid parity.

Run:
  POSEIDON_input_data=$HOME/inputs/ \
  PYSYN_CDBS=$HOME/inputs/stellar_grids \
  ~/anaconda3/envs/poseidon/bin/python precompute/smoke_poseidon.py
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("POSEIDON_input_data",
                      str(Path.home() / "inputs") + "/")
os.environ.setdefault("PYSYN_CDBS",
                      str(Path.home() / "inputs" / "stellar_grids"))

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import (P_GRID_ARGS, P_REF_BAR, PARAM_SPECIES, SCENARIOS,
                       make_wl_grid)

from POSEIDON.chemistry import load_chemistry_grid
from POSEIDON.constants import R_J, M_J, R_Sun
from POSEIDON.core import (compute_spectrum, create_planet, create_star,
                           define_model, make_atmosphere, read_opacities,
                           wl_grid_constant_R)

# WASP-39 b (approx.; smoke test only)
T_eq, R_p, M_p, R_s, T_eff, log_g_s = 1120.0, 1.27, 0.28, 0.93, 5400.0, 4.45

wl = wl_grid_constant_R(0.35, 1.85, 10_000)
ours = make_wl_grid()
assert len(wl) == len(ours) and np.allclose(wl, ours), \
    "make_wl_grid() disagrees with POSEIDON wl_grid_constant_R — fix scenarios.py"

star = create_star(R_s * R_Sun, T_eff, log_g_s, 0.0)
planet = create_planet("smoke", R_p * R_J, mass=M_p * M_J)

cfg = SCENARIOS["clear"]
model = define_model("smoke_clear", bulk_species=["H2", "He"],
                     param_species=PARAM_SPECIES,
                     PT_profile="isotherm", X_profile="chem_eq",
                     cloud_model=cfg["cloud_model"])

# POSEIDON expects P ordered deep -> shallow (descending), matching its own
# tutorials' `np.logspace(log10(P_max), log10(P_min), N_layers)`. Building it
# ascending (as an earlier version of this script did) silently breaks
# POSEIDON's internal `i_bot = argmin(|P - P_deep|)` layer-loop bound in
# absorption.extinction(), which then only computes real opacity for ~1 of
# 100 layers, leaving the rest of the atmosphere at kappa=0 (fully
# transparent) and producing a flat, saturated spectrum set by geometry
# alone (found via debug instrumentation of POSEIDON.core.compute_spectrum;
# see task-8-report.md).
lo, hi, n = P_GRID_ARGS
P = np.logspace(hi, lo, n)
T_fine = np.linspace(max(T_eq - 200, 100), T_eq + 200, 10)
log_P_fine = np.linspace(-6.0, 2.0, 40)
opac = read_opacities(model, wl, opacity_treatment="opacity_sampling",
                      T_fine=T_fine, log_P_fine=log_P_fine)

# X_profile="chem_eq" requires a pre-loaded fastchem grid, passed explicitly
# into make_atmosphere (not implicit in `model`) — POSEIDON raises
# "Error: no chemistry grid loaded for an equilibrium model" without it.
chemistry_grid = load_chemistry_grid(PARAM_SPECIES, grid="fastchem")

atmosphere = make_atmosphere(planet, model, P, P_REF_BAR, R_p * R_J,
                             PT_params=[T_eq],
                             log_X_params=[cfg["C_to_O"], cfg["log_Met"]],
                             cloud_params=cfg["cloud_params"],
                             chemistry_grid=chemistry_grid)
spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                            spectrum_type="transmission")

assert np.isfinite(spectrum).all()
print(f"OK: {len(spectrum)} points, depth range "
      f"{spectrum.min():.5f}-{spectrum.max():.5f} "
      f"(expect roughly 0.018-0.031 for WASP-39 b; the upper end is set by "
      f"the saturated Na D line core at ~0.589 um, not the H2O/continuum "
      f"bulk of the spectrum -- see task-8-report.md)")
