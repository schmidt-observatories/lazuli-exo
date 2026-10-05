"""Shared helpers for the cross-check scripts (py313 env, run from the repo root)."""
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MEMO = ROOT / "memo"
DATA, RESULTS, FIGS = MEMO / "data", MEMO / "results", MEMO / "figures"
for d in (RESULTS, FIGS):
    d.mkdir(exist_ok=True)

from lazuli_exo import tables  # noqa: E402
from lazuli_exo.targets import PLANET_NAMES, TARGETS  # noqa: E402

TDIR = tables.resolve_tables_dir(None)
BAND_UM = (0.40, 1.70)
HC_ERG_A = 6.62607015e-27 * 2.99792458e10 * 1e8      # h c [erg Angstrom]
A_LAZULI_M2 = np.pi / 4 * 3.0 ** 2                      # slicersim lazuli_cbe: 3.0 m, unobstructed
A_HST_M2, A_JWST_M2 = 4.52, 25.4                        # HST: pi (1.2 m)^2 geometric; JWST: JDox 25.4 m^2


def host(planet):
    return planet.rsplit(" ", 1)[0]


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def strip_html(s):
    return re.sub(r"<.*?>", "", s or "").replace("&amp;", "&").strip()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def photometry():
    """SIMBAD photometry by planet name (fetch_data.py)."""
    return {r["planet"]: r for r in read_csv(DATA / "simbad_photometry.csv")}


def synth_mag(wl_A, flam, band, system="vega"):
    """Synthetic magnitude of F_lambda [erg s-1 cm-2 A-1] through an sncosmo band."""
    import sncosmo
    return float(sncosmo.Spectrum(np.asarray(wl_A, float), np.asarray(flam, float))
                 .bandmag(band, system))


def normalize_to_J(wl_A, flam, J_vega):
    """Rescale F_lambda so that the synthetic 2MASS J (Vega) equals J_vega."""
    return np.asarray(flam, float) * 10 ** (-0.4 * (J_vega - synth_mag(wl_A, flam, "2massj")))


def star_table_sed(planet):
    """The committed star table (POSEIDON PHOENIX or blackbody): (A, F_lambda arb., meta)."""
    wl_um, f, meta = tables.load_star(TDIR, planet)
    return wl_um * 1e4, f, meta


def phoenix_grid_sed(planet):
    """The parametric path's star: lazuli_exo.star.phoenix_sed at the registry Teff/logg/[Fe/H]."""
    from lazuli_exo.star import phoenix_sed
    t = TARGETS[planet]
    return phoenix_sed(t["T_eff_K"], t["log_g_star"], t["Met_star"])


def J_vega(planet, phot=None):
    """Catalog 2MASS J: SIMBAD if available, else the frozen registry."""
    phot = photometry() if phot is None else phot
    j = fnum(phot.get(planet, {}).get("J"))
    return j if np.isfinite(j) else TARGETS[planet]["J_mag"]


def gks_style():
    import matplotlib.pyplot as plt
    try:
        plt.style.use("gks")
    except OSError:
        pass
    return plt


def savefig(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    print(f"wrote memo/figures/{name}.pdf")
