"""Build <package>/data/tables/atmo/zeta_grid.npz: photospheric log-pressure zeta(lambda)
for a fiducial planet on a (T, [M/H], C/O) grid.

Run (poseidon env, ~1 h; resumable, memory-safe):
  POSEIDON_input_data=$HOME/inputs/ PYSYN_CDBS=$HOME/inputs/stellar_grids \
  nice ~/anaconda3/envs/poseidon/bin/python precompute/make_zeta_grid.py \
      [--smoke] [--resume]

Output: zeta[T, met, co, fid, wl] float16 at the native R=10,000 (~38 MB raw
for two fiducials along the slant-column axis) and mu[T, met, co] float32.

Memory: opacities are loaded per temperature node with a 10-point T_fine
window (the footprint that already worked for the per-planet tables) and
freed before the next node; results are flushed to a .partial.npz after every
node so a crash costs at most one node.
"""
import argparse
import gc
import json
import os
import resource
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

os.environ.setdefault("POSEIDON_input_data", str(Path.home() / "inputs") + "/")
os.environ.setdefault("PYSYN_CDBS", str(Path.home() / "inputs" / "stellar_grids"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import scenarios  # noqa: E402
from scenarios import (FIDUCIALS, P_GRID_ARGS, P_REF_BAR, PARAM_SPECIES,  # noqa: E402
                       ZETA_C_TO_O, ZETA_LOG_MET, ZETA_T_K)
from lazuli_exo import tables  # noqa: E402

TABLES = tables.resolve_tables_dir(None)
LOG = Path(__file__).with_suffix(".log")
G_SI = 6.674e-11
AMU_KG = 1.66053907e-27
MU_LAYERS_BAR = (1e-4, 1e-1)


def log(msg):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    line = f"[{time.strftime('%H:%M:%S')} rss={rss_mb:.0f}MB] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def poseidon_version():
    try:
        from importlib.metadata import version
        return version("POSEIDON")
    except Exception:
        return "unknown"


def zeta_from_spectrum(spectrum, atmosphere, R_s_m):
    """Map depth(lambda) -> R(lambda) -> P through the atmosphere's own r(P)."""
    P = np.asarray(atmosphere["P"], float)                    # descending
    r = np.asarray(atmosphere["r"], float).reshape(len(P), -1)[:, 0]
    R_lam = np.sqrt(np.asarray(spectrum, float)) * R_s_m
    order = np.argsort(r)                                     # r ascending
    lnP = np.interp(R_lam, r[order], np.log(P[order]))
    return -lnP, int((R_lam < r.min()).sum()), int((R_lam > r.max()).sum())


def mu_photosphere(atmosphere):
    """Mean molecular weight [amu], median over 1e-4..1e-1 bar (POSEIDON mu is kg)."""
    P = np.asarray(atmosphere["P"], float)
    mu = np.asarray(atmosphere["mu"], float).reshape(len(P), -1)[:, 0]
    m = (P >= MU_LAYERS_BAR[0]) & (P <= MU_LAYERS_BAR[1])
    return float(np.median(mu[m]) / AMU_KG)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 T x 1 met x 1 C/O at full R; writes zeta_grid.smoke.npz")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    from POSEIDON.chemistry import load_chemistry_grid
    from POSEIDON.constants import R_J, R_Sun
    from POSEIDON.core import (compute_spectrum, create_planet, create_star,
                               define_model, make_atmosphere, read_opacities)

    target = TABLES / tables.ZETA_GRID
    if args.smoke:
        target = target.with_name("zeta_grid.smoke.npz")
    partial = target.with_suffix(".partial.npz")

    T_K = ZETA_T_K[7:8] if args.smoke else ZETA_T_K        # 1000 K
    log_met = ZETA_LOG_MET[2:3] if args.smoke else ZETA_LOG_MET
    c_to_o = ZETA_C_TO_O[1:2] if args.smoke else ZETA_C_TO_O
    WL = scenarios.make_wl_grid()
    # zeta is stored at the native model resolution (float16): a gray deck or
    # haze acts per wavelength, and max(mean zeta, zeta_deck) over a coarser
    # bin would erase the line cores that poke above a deck (Jensen).
    wl_store = WL

    fids = []
    for k, f in enumerate(FIDUCIALS):
        R_p = f["R_p_RJ"] * R_J
        R_s = f["R_s_RSun"] * R_Sun
        M_p = f["g_ms2"] * R_p ** 2 / G_SI
        fids.append(dict(R_p_m=float(R_p), M_p_kg=float(M_p), g_ms2=f["g_ms2"],
                         R_s_m=float(R_s), P_ref_bar=P_REF_BAR,
                         x_column=0.5 * float(np.log(
                             (f["R_p_RJ"] / FIDUCIALS[0]["R_p_RJ"])
                             * (FIDUCIALS[0]["g_ms2"] / f["g_ms2"]))),
                         planet=create_planet(f"zeta_fiducial_{k}", R_p, mass=M_p)))
    star = create_star(fids[0]["R_s_m"], 5500.0, 4.5, 0.0, stellar_grid="blackbody", wl=WL)
    model = define_model("zeta_grid", bulk_species=["H2", "He"],
                         param_species=PARAM_SPECIES, PT_profile="isotherm",
                         X_profile="chem_eq", cloud_model="cloud-free")
    chem = load_chemistry_grid(PARAM_SPECIES, grid="fastchem")
    lo, hi, n = P_GRID_ARGS
    P = np.logspace(hi, lo, n)            # descending: POSEIDON requires it
    log_P_fine = np.linspace(-6.0, 2.0, 40)

    shape = (len(T_K), len(log_met), len(c_to_o), len(fids), len(wl_store))
    zeta = np.full(shape, np.nan, np.float16)
    mu = np.full(shape[:3], np.nan, np.float32)
    done = np.zeros(len(T_K), bool)
    out_of_column = {}
    if args.resume and partial.exists():
        with np.load(partial) as f:
            if f["zeta"].shape == shape:
                zeta, mu, done = f["zeta"], f["mu"], f["done"]
                out_of_column = json.loads(str(f["out_of_column"]))
                log(f"resuming; done T nodes: {np.array(T_K)[done].tolist()}")

    t_start = time.time()
    for it, T in enumerate(T_K):
        if done[it]:
            continue
        T_fine = np.linspace(max(T - 200.0, 100.0), T + 200.0, 10)
        opac = read_opacities(model, WL, opacity_treatment="opacity_sampling",
                              T_fine=T_fine, log_P_fine=log_P_fine)
        log(f"T={T} K: opacities loaded")
        for im, met in enumerate(log_met):
            for ic, co in enumerate(c_to_o):
                for k, f in enumerate(fids):
                    atmo = make_atmosphere(f["planet"], model, P, P_REF_BAR,
                                           f["R_p_m"], PT_params=[float(T)],
                                           log_X_params=[float(co), float(met)],
                                           cloud_params=[], chemistry_grid=chem)
                    spec = compute_spectrum(f["planet"], star, model, atmo, opac,
                                            WL, spectrum_type="transmission")
                    assert np.isfinite(spec).all(), (T, met, co, k)
                    z, n_below, n_above = zeta_from_spectrum(spec, atmo, f["R_s_m"])
                    zeta[it, im, ic, k] = z
                    if k == 0:
                        mu[it, im, ic] = mu_photosphere(atmo)
                    if args.smoke:
                        log(f"  smoke T={T} met={met} co={co} fid={k}: depth "
                            f"{spec.min():.5f}..{spec.max():.5f}, zeta "
                            f"{z.min():.2f}..{z.max():.2f}, mu {mu[it, im, ic]:.3f}, "
                            f"out_of_column {n_below}/{n_above}")
                    if n_below or n_above:
                        out_of_column[f"{T}_{met}_{co}_{k}"] = [n_below, n_above]
        done[it] = True
        del opac
        gc.collect()
        partial.parent.mkdir(parents=True, exist_ok=True)
        np.savez(partial, zeta=zeta, mu=mu, done=done,
                 out_of_column=json.dumps(out_of_column))
        elapsed = (time.time() - t_start) / 60
        log(f"T={T} K done ({done.sum()}/{len(T_K)}), {elapsed:.1f} min elapsed, "
            f"zeta {np.nanmin(zeta[it]):.2f}..{np.nanmax(zeta[it]):.2f}, "
            f"mu {np.nanmin(mu[it]):.2f}..{np.nanmax(mu[it]):.2f}")

    fid_meta = [{k: v for k, v in f.items() if k != "planet"} for f in fids]
    meta = {"generator": "make_zeta_grid.py", "created": str(date.today()),
            "fiducials": fid_meta, "fiducial": fid_meta[0],
            "column_axis": "x = 0.5 ln[(R_p H)/(R_p H)_fid0]; zeta axis 3",
            "P_grid_log10bar": list(P_GRID_ARGS), "species": PARAM_SPECIES,
            "chemistry": "fastchem", "PT_profile": "isotherm",
            "poseidon_version": poseidon_version(),
            "R_model": int(round(1.0 / np.log(WL[1] / WL[0]))),
            "R_store": "native", "zeta_dtype": "float16",
            "mu_layers_bar": list(MU_LAYERS_BAR),
            "zeta_definition": "-ln(P_tau=1 / 1 bar) at slant optical depth "
                               "unity, via depth -> R -> r(P) column",
            "out_of_column": out_of_column, "smoke": bool(args.smoke)}
    tables.write_zeta_grid(TABLES, T_K=T_K, log_met=log_met, c_to_o=c_to_o,
                           wavelength_um=wl_store, zeta=zeta, mu=mu, meta=meta,
                           smoke=args.smoke)
    partial.unlink(missing_ok=True)
    if not args.smoke:
        tables.rebuild_index(TABLES)
    log(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
