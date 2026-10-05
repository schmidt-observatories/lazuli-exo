"""Does zeta at fixed slant column x = 0.5 ln(R_p H) depend on the (R_p, g) split?"""
import os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("POSEIDON_input_data", str(Path.home() / "inputs") + "/")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "precompute"))
from scenarios import P_GRID_ARGS, P_REF_BAR, PARAM_SPECIES, make_wl_grid
from POSEIDON.chemistry import load_chemistry_grid
from POSEIDON.constants import R_J, R_Sun
from POSEIDON.core import compute_spectrum, create_planet, create_star, define_model, make_atmosphere, read_opacities
WL = make_wl_grid(); band = (WL >= 0.4) & (WL <= 1.7)
chem = load_chemistry_grid(PARAM_SPECIES, grid="fastchem")
model = define_model("split", bulk_species=["H2", "He"], param_species=PARAM_SPECIES,
                     PT_profile="isotherm", X_profile="chem_eq", cloud_model="cloud-free")
lo, hi, n = P_GRID_ARGS; P = np.logspace(hi, lo, n); R_s = R_Sun
star = create_star(R_s, 5500.0, 4.5, 0.0, stellar_grid="blackbody", wl=WL)

def zeta(T, R_p_RJ, g, opac):
    R_p = R_p_RJ * R_J; planet = create_planet("p", R_p, mass=g * R_p**2 / 6.674e-11)
    atmo = make_atmosphere(planet, model, P, P_REF_BAR, R_p, PT_params=[T], log_X_params=[0.55, 0.0], cloud_params=[], chemistry_grid=chem)
    spec = compute_spectrum(planet, star, model, atmo, opac, WL, spectrum_type="transmission")
    r = np.asarray(atmo["r"], float).reshape(len(P), -1)[:, 0]; o = np.argsort(r)
    return -np.interp(np.sqrt(spec) * R_s, r[o], np.log(P[o]))

for T in (300.0, 1000.0):
    opac = read_opacities(model, WL, opacity_treatment="opacity_sampling",
                          T_fine=np.linspace(max(T - 200, 100), T + 200, 10), log_P_fine=np.linspace(-6, 2, 40))
    ref = zeta(T, 1.0, 10.0, opac)
    for R_p_RJ, g in [(0.3, 30.0), (0.1, 10.0), (0.3, 10.0), (1.0, 30.0), (3.0, 10.0), (1.0, 3.0), (0.3, 3.0)]:
        x = 0.5 * np.log(R_p_RJ * 10.0 / g)
        d = zeta(T, R_p_RJ, g, opac) - (ref + x)
        print(f"T={T:.0f} R_p={R_p_RJ:4.1f} RJ g={g:4.1f}: x={x:+.3f} ln(g/10)={np.log(g/10):+.3f} | zeta - (ref + x): "
              f"median {np.median(d[band]):+.3f}  p10 {np.percentile(d[band],10):+.3f}  p90 {np.percentile(d[band],90):+.3f}", flush=True)
    del opac
