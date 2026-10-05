"""Direct POSEIDON K2-18 b-like run (clear + 1 mbar deck) vs the blended two-node model,
focusing on the wavelengths above the deck."""
import os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("POSEIDON_input_data", str(Path.home() / "inputs") + "/")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "precompute"))
from scenarios import P_GRID_ARGS, P_REF_BAR, PARAM_SPECIES, make_wl_grid
from lazuli_exo import binning, tables
from lazuli_exo.atmosphere import transmission_depth, K_B, M_H, G
from POSEIDON.chemistry import load_chemistry_grid
from POSEIDON.constants import R_J, R_Sun, M_J
from POSEIDON.core import compute_spectrum, create_planet, create_star, define_model, make_atmosphere, read_opacities

WL = make_wl_grid(); T = 2601.0; band = (WL >= 0.4) & (WL <= 1.7)
chem = load_chemistry_grid(PARAM_SPECIES, grid="fastchem")
lo, hi, n = P_GRID_ARGS; P = np.logspace(hi, lo, n)
R_p = 1.965 * R_J; R_s = 1.69 * R_Sun; M_p = 1.47 * M_J
planet = create_planet("k218", R_p, mass=M_p)
star = create_star(R_s, 6265.0, 4.1, 0.0, stellar_grid="blackbody", wl=WL)
edges = binning.constant_R_edges(0.40, 1.70, 100)
res = {}
for scen, kw, cp in (("clear", dict(cloud_model="cloud-free"), []),
                     ("cloudy", dict(cloud_model="MacMad17", cloud_type="deck"), [-3.0])):
    model = define_model("d_" + scen, bulk_species=["H2", "He"], param_species=PARAM_SPECIES,
                         PT_profile="isotherm", X_profile="chem_eq", **kw)
    if scen == "clear":
        opac = read_opacities(model, WL, opacity_treatment="opacity_sampling",
                              T_fine=np.linspace(T - 200, T + 200, 10), log_P_fine=np.linspace(-6, 2, 40))
    atmo = make_atmosphere(planet, model, P, P_REF_BAR, R_p, PT_params=[T], log_X_params=[0.55, 0.0],
                           cloud_params=cp, chemistry_grid=chem)
    spec = compute_spectrum(planet, star, model, atmo, opac, WL, spectrum_type="transmission")
    r = np.asarray(atmo["r"], float).reshape(len(P), -1)[:, 0]; o = np.argsort(r)
    z = -np.interp(np.sqrt(spec) * R_s, r[o], np.log(P[o]))
    res[scen] = (spec, z, atmo)
spec_c, z_c, atmo_c = res["clear"]; spec_d, z_d, _ = res["cloudy"]
R_med = np.sqrt(np.median(spec_c[band])) * R_s
g_med = G * M_p / R_med**2
for label, extra in (("clear", {}), ("cloudy@-3", {"log_P_cloud_bar": -3.0}), ("cloudy@realized", {"log_P_cloud_bar": -3.0 + np.log10(1.376)})):
    dm, info = transmission_depth(WL, T_eq_K=T, R_p_RJ=R_med / R_J, M_p_MJ=1.47, R_s_RSun=1.69,
                                  tables_dir=None, **extra)
    tgt = spec_c if label == "clear" else spec_d
    bt = binning.bin_model(WL, tgt, edges); bm = binning.bin_model(WL, dm, edges)
    print(f"{label:16s}: POSEIDON amp {np.ptp(bt)*1e6:6.1f} ppm | model amp {np.ptp(bm)*1e6:6.1f} ppm | ratio {np.ptp(bm)/np.ptp(bt):.2f} | shift {info['zeta_shift']:+.3f} H/Rp {info['H_over_Rp']:.4f}")
    if label == "clear":
        # model zeta on absolute scale: zeta_ref + (R_med^2/H)(1/R_med - 1/r)
        H = info["H_m"]; r_m = np.sqrt(dm) * R_s
        z_m = info["zeta_ref"] + (R_med**2 / H) * (1 / R_med - 1 / r_m)
        d = z_c - z_m
        above = band & (z_c > 6.9)
        print(f"   zeta POSEIDON - model: median {np.median(d[band]):+.3f}; at the {above.sum()} wavelengths above 1 mbar: "
              f"median {np.median(d[above]):+.3f}, mean POSEIDON excess over deck {np.mean(z_c[above]-6.9):.3f} vs model {np.mean(z_m[above]-6.9):.3f}")
        top = band & (z_c > np.percentile(z_c[band], 95))
        print(f"   top-5% zeta wavelengths: POSEIDON {np.mean(z_c[top]):.3f} model {np.mean(z_m[top]):.3f}; "
              f"model-only above 1 mbar: {np.sum(band & (z_m > 6.9))} wavelengths")
