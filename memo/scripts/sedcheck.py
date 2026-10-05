"""V-J and J-H colours of the committed star tables vs SIMBAD photometry (py313; network)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, sncosmo
from astroquery.simbad import Simbad
from lazuli_exo import tables
from lazuli_exo.targets import TARGETS
S = Simbad(); S.add_votable_fields("V", "J", "H")
vega = sncosmo.get_magsystem("vega")
for n, t in sorted(TARGETS.items()):
    r = S.query_object(n.rsplit(" ", 1)[0])
    cols = {c.lower(): c for c in r.colnames}
    get = lambda k: float(r[cols[k]][0]) if k in cols and not np.ma.is_masked(r[cols[k]][0]) else np.nan
    V, J, H = get("v"), get("j"), get("h")
    wl, F, _ = tables.load_star(str(tables.resolve_tables_dir(None)), n)
    w, f = wl * 1e4, F                     # colours only: normalization irrelevant
    def mag(b):
        bp = sncosmo.get_bandpass(b)
        flux = np.trapezoid(np.interp(bp.wave, w, f) * bp.trans * bp.wave, bp.wave)
        return -2.5 * np.log10(flux / vega.zpbandflux(b))
    VJ, JH = mag("bessellv") - mag("2massj"), mag("2massj") - mag("2massh")
    print(f"{n:14s} V-J model {VJ:5.2f} obs {V-J:5.2f} (Δ {VJ-(V-J):+.2f})  "
          f"J-H model {JH:5.2f} obs {J-H:5.2f} (Δ {JH-(J-H):+.2f})")
