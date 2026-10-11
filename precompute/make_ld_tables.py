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
to the grid's support where needed and any clamping is recorded in the
table meta.

exotic-ld's unconstrained quadratic fit can leave the Kipping (2013)
triangle (u1 >= 0, u1 + u2 <= 1, u1 + 2 u2 >= 0; negative limb intensity)
for cool stars (TRAPPIST-1, issue #10), so each chunk is refitted to the
same I(mu) profile over mu >= MU_MIN under those constraints; chunks where
a constraint was active are listed in meta["constrained_chunks"].
"""
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lazuli_exo import tables                    # noqa: E402
from lazuli_exo.targets import TARGETS           # noqa: E402

TABLES = tables.resolve_tables_dir(None)
LD_DATA = os.environ.get("EXOTIC_LD_DATA",
                         str(Path.home() / "inputs" / "exotic_ld_data"))

WL_MIN_UM, WL_MAX_UM = 0.40, 1.70
N_CHUNK = 48
LD_GRID = "phoenix"
# exotic-ld phoenix grid support (per its docs); inputs outside are clamped
GRID_BOUNDS = {"Teff": (2300.0, 12000.0), "logg": (0.0, 6.0),
               "M_H": (-4.0, 1.0)}
MU_MIN = 0.10   # exotic-ld default; the 2566 K fit depends on it (issue #10)
# Kipping (2013): u1 >= 0, u1 + u2 <= 1, u1 + 2 u2 >= 0
KIPPING = [{"type": "ineq", "fun": lambda u: u[0]},
           {"type": "ineq", "fun": lambda u: 1.0 - u[0] - u[1]},
           {"type": "ineq", "fun": lambda u: u[0] + 2.0 * u[1]}]


def is_physical(u1, u2):
    return u1 >= 0 and u1 + u2 <= 1 and u1 + 2 * u2 >= 0


def fit_quadratic_constrained(mu, i_mu, mu_min=MU_MIN):
    """Least-squares (u1, u2) to I(mu)/I(1) over mu >= mu_min, refitted under
    the Kipping constraints if the unconstrained optimum (the exotic-ld
    answer) lies outside them. Returns (u1, u2, constrained)."""
    x = 1.0 - mu[mu >= mu_min]
    A = np.column_stack([x, x ** 2])
    y = 1.0 - i_mu[mu >= mu_min]
    u, *_ = np.linalg.lstsq(A, y, rcond=None)
    if is_physical(*u):
        return float(u[0]), float(u[1]), False
    res = minimize(lambda v: np.sum((A @ v - y) ** 2), u, method="SLSQP",
                   constraints=KIPPING)
    if not res.success:
        raise RuntimeError(f"constrained LD fit failed: {res.message}")
    return float(res.x[0]), float(res.x[1]), True


def constrain_quadratic(u1, u2, mu_min=MU_MIN):
    """Project (u1, u2) onto the Kipping triangle, least-squares in the
    quadratic-law I(mu) over [mu_min, 1]; physical pairs are returned as is."""
    if is_physical(u1, u2):
        return u1, u2
    mu = np.linspace(mu_min, 1.0, 256)
    return fit_quadratic_constrained(
        mu, 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2, mu_min)[:2]


def clamp(name, value):
    lo, hi = GRID_BOUNDS[name]
    c = min(max(value, lo), hi)
    return c, (c != value)


def realised_grid(sld):
    """exotic-ld silently falls back to 'nearest' when the trilinear cuboid is
    incomplete and only records that on an internal object; redo the lookup
    (cached files) to get the realised interpolate_type and cuboid."""
    from exotic_ld.ld_grids import StellarGrids
    sg = StellarGrids(sld.M_H_input, sld.Teff_input, sld.logg_input,
                      sld.ld_model, sld.ld_data_path, sld.remote_ld_data_path,
                      sld.ld_data_version, sld.interpolate_type, 0)
    sg.get_stellar_data()
    cuboid = None
    if sg.interpolate_type == "trilinear":
        scale = (sg._r_M_H, sg._r_M_H, sg._r_Teff, sg._r_Teff,
                 sg._r_logg, sg._r_logg)
        cuboid = [round(c * r, 4) + 0.0 for c, r in   # + 0.0 drops -0.0
                  zip(sg._get_surrounding_grid_cuboid(), scale)]
    return sg.interpolate_type, cuboid


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

        interp, cuboid = realised_grid(sld)

        u1 = np.empty(N_CHUNK)
        u2 = np.empty(N_CHUNK)
        constrained = []
        for i in range(N_CHUNK):
            lo_a, hi_a = edges[i] * 1e4, edges[i + 1] * 1e4
            wl_a = np.linspace(lo_a, hi_a, 32)
            sld.compute_quadratic_ld_coeffs(      # integrates sld.I_mu
                wavelength_range=[lo_a, hi_a], mode="custom",
                custom_wavelengths=wl_a,
                custom_throughput=np.ones_like(wl_a), mu_min=MU_MIN)
            u1[i], u2[i], active = fit_quadratic_constrained(
                sld.mus, sld.I_mu, MU_MIN)
            if active:
                constrained.append(i)

        clamped = {k: v for k, v in
                   [("Teff", c_t), ("logg", c_g), ("M_H", c_m)] if v}
        tables.write_ld_table(
            TABLES, name, centers, u1, u2,
            meta={"generator": "make_ld_tables.py",
                  "created": str(date.today()),
                  "ld_grid": LD_GRID, "law": "quadratic",
                  "interpolate_type": interp,       # realised, not requested
                  "grid_cuboid": cuboid,            # [M_H0, M_H1, T0, T1, g0, g1]
                  "mu_min": MU_MIN,
                  "constrained_chunks": constrained,
                  "exotic_ld_version": _version(),
                  "Teff_used": teff, "logg_used": logg, "M_H_used": m_h,
                  "inputs_clamped": sorted(clamped) if clamped else [],
                  "n_chunks": N_CHUNK,
                  "wl_range_um": [WL_MIN_UM, WL_MAX_UM]})
        note = f"  (clamped: {', '.join(sorted(clamped))})" if clamped else ""
        if constrained:
            note += f"  (constrained chunks: {constrained})"
        if interp != "trilinear":
            note += f"  (interpolate_type={interp})"
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
