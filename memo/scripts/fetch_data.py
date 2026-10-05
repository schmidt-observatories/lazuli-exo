"""Fetch the as-measured reference data for the cross-check memo (py313; network).

Writes to memo/data/:
  simbad_photometry.csv     host, Gaia DR3 id, V J H K G (SIMBAD)
  gaia_xp/<host-slug>.csv   Gaia DR3 XP sampled spectrum (nm, W m-2 nm-1)
  transitspec.csv           NASA Exoplanet Archive 'transitspec' rows, all 14 planets
  ps_solutions.csv          NASA Exoplanet Archive 'ps' rows (per-reference solutions)
  trappist1e/*.txt          Espinoza et al. 2025 TRAPPIST-1 e JWST/PRISM + HST/WFC3 spectra
                            (Zenodo record 16125662, appendix-figure1/spectra, Espinoza reduction)
Everything downstream (checkA..checkD) reads these files, so the checks are
reproducible offline and the data snapshot is versioned with the memo.
"""
import csv
import re
import sys
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lazuli_exo.targets import PLANET_NAMES  # noqa: E402
from lazuli_exo.tables import slugify  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
HOSTS = {p: p.rsplit(" ", 1)[0] for p in PLANET_NAMES}


def tap(query, out):
    url = TAP + urllib.parse.quote(query)
    body = urllib.request.urlopen(url, timeout=300).read().decode()
    if body.lstrip().startswith("<?xml"):
        raise RuntimeError(body[:500])
    out.write_text(body)
    print(f"{out.name}: {body.count(chr(10)) - 1} rows")


def fetch_archive():
    names = ",".join(f"'{p}'" for p in PLANET_NAMES)
    tap("select plntname,centralwavelng,bandwidth,plntransdep,plntransdeperr1,"
        "plntransdeperr2,plnratror,plnratrorerr1,plnratrorerr2,facility,instrument,"
        f"plntranreflink from transitspec where plntname in ({names}) "
        "order by plntname,centralwavelng", DATA / "transitspec.csv")
    tap("select pl_name,pl_refname,pl_ratror,pl_ratrorerr1,pl_ratdor,pl_ratdorerr1,"
        "pl_orbincl,pl_orbinclerr1,pl_trandur,pl_trandurerr1,pl_imppar,pl_orbeccen,"
        "pl_orblper,pl_orbper,pl_radj,pl_bmassj,st_rad,st_teff,st_logg,st_met,pl_eqt,"
        f"sy_jmag,sy_vmag,sy_gaiamag,default_flag,pl_pubdate,disc_facility from ps "
        f"where pl_name in ({names}) order by pl_name,pl_pubdate", DATA / "ps_solutions.csv")


def fetch_simbad_and_xp():
    from astroquery.gaia import Gaia
    from astroquery.simbad import Simbad
    S = Simbad()
    S.add_votable_fields("ids", "V", "J", "H", "K", "G")
    (DATA / "gaia_xp").mkdir(exist_ok=True)
    rows = []
    for planet, host in sorted(HOSTS.items()):
        r = S.query_object(host)
        ids = str(r["ids"][0])
        m = re.search(r"Gaia DR3 (\d+)", ids)
        gaia = int(m.group(1)) if m else None

        def val(col):
            v = r[col][0]
            return "" if v is None or getattr(v, "mask", False) or str(v) in ("--", "") else float(v)
        rows.append(dict(planet=planet, host=host, gaia_dr3=gaia or "",
                         **{b: val(b) for b in ("V", "J", "H", "K", "G")}))
        if gaia:
            res = Gaia.load_data(ids=[gaia], data_release="Gaia DR3",
                                 retrieval_type="XP_SAMPLED", format="votable",
                                 verbose=False)
            if not res:
                print(f"XP {host}: no XP_SAMPLED product (Gaia DR3 {gaia})")
                continue
            key = next(iter(res))
            tab = res[key][0].to_table()
            with (DATA / "gaia_xp" / f"{slugify(host)}.csv").open("w") as f:
                f.write("wavelength_nm,flux_W_m2_nm,flux_err_W_m2_nm\n")
                for w, fl, fe in zip(tab["wavelength"], tab["flux"], tab["flux_error"]):
                    f.write(f"{float(w):.1f},{float(fl):.6e},{float(fe):.6e}\n")
            print(f"XP {host}: {len(tab)} samples")
        else:
            print(f"XP {host}: no Gaia DR3 id in SIMBAD")
    with (DATA / "simbad_photometry.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


ZENODO_T1E = ("https://zenodo.org/api/records/16125662/files/"
              "TRAPPIST-1e-GTO-2025-main%20(2).zip/content")


def fetch_trappist1e():
    """Four JWST/NIRSpec PRISM visits + two HST/WFC3 visits of TRAPPIST-1 e
    (Espinoza et al. 2025, their appendix figure 1; 'Espinoza' reduction)."""
    import io
    import zipfile
    out = DATA / "trappist1e"
    out.mkdir(exist_ok=True)
    blob = urllib.request.urlopen(ZENODO_T1E, timeout=600).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            if "appendix-figure1/spectra/" in name and (
                    "JWST_PRISM_Espinoza" in name or "HST_WFC3IR" in name):
                (out / Path(name).name).write_bytes(z.read(name))
                print("trappist1e:", Path(name).name)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    fetch_archive()
    fetch_simbad_and_xp()
    fetch_trappist1e()
