"""Lookup-table storage for terminator.

Layout under a tables directory:
    index.json
    models/<planet-slug>__<scenario>.npz   (wavelength_um, transit_depth, meta)
    star/<planet-slug>.npz                 (wavelength_um, f_lambda, meta)
    noise/<planet-slug>.npz                (wavelength_um, depth_err_1transit, meta)
    ld/<planet-slug>.npz                   (wavelength_um, u1, u2, meta)
    atmo/zeta_grid.npz      (T_K, log_met, c_to_o, wavelength_um, zeta[T,met,co,fid,wl], mu[T,met,co], meta)
    star/phoenix_grid.npz   (teff_K, logg, feh, wavelength_um, log_flam, meta)

All wavelengths are in micron. `meta` is a JSON dict stored as a string array.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
ZETA_GRID = Path("atmo") / "zeta_grid.npz"
PHOENIX_GRID = Path("star") / "phoenix_grid.npz"
_GRID_CACHE = {}


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def resolve_tables_dir(tables_dir=None):
    if tables_dir is not None:
        return Path(tables_dir)
    env = os.environ.get("TERMINATOR_TABLES")
    if env:
        return Path(env)
    default = Path(__file__).resolve().parent / "data" / "tables"
    if default.exists():
        return default
    raise FileNotFoundError(
        "No tables directory: pass tables_dir=, set $TERMINATOR_TABLES, "
        f"or create {default}"
    )


def _write(path, meta, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta or {}), **arrays)
    return path


def _read(path, *keys):
    with np.load(path) as f:
        out = [np.asarray(f[k], dtype=float) for k in keys]
        meta = json.loads(str(f["meta"]))
    return (*out, meta)


def _model_path(tdir, planet, scenario):
    return Path(tdir) / "models" / f"{slugify(planet)}__{scenario}.npz"


def write_model_table(tables_dir, planet, scenario, wavelength_um,
                      transit_depth, meta):
    meta = dict(meta or {}, planet=planet, scenario=scenario)
    return _write(_model_path(tables_dir, planet, scenario), meta,
                  wavelength_um=wavelength_um, transit_depth=transit_depth)


def write_star_table(tables_dir, planet, wavelength_um, f_lambda, meta):
    meta = dict(meta or {}, planet=planet)
    return _write(Path(tables_dir) / "star" / f"{slugify(planet)}.npz", meta,
                  wavelength_um=wavelength_um, f_lambda=f_lambda)


def write_noise_table(tables_dir, planet, wavelength_um,
                      depth_err_1transit, meta):
    meta = dict(meta or {}, planet=planet)
    return _write(Path(tables_dir) / "noise" / f"{slugify(planet)}.npz", meta,
                  wavelength_um=wavelength_um,
                  depth_err_1transit=depth_err_1transit)


def write_ld_table(tables_dir, planet, wavelength_um, u1, u2, meta):
    """Quadratic limb-darkening coefficients u1(lambda), u2(lambda)."""
    meta = dict(meta or {}, planet=planet)
    return _write(Path(tables_dir) / "ld" / f"{slugify(planet)}.npz", meta,
                  wavelength_um=wavelength_um, u1=u1, u2=u2)


def _write_grid(path, meta, dtypes, **arrays):
    arrays = {k: np.asarray(v, dtype=dtypes.get(k, float))
              for k, v in arrays.items()}
    return _write(path, meta, **arrays)


def _load_grid(path, script):
    path = Path(path)
    key = str(path.resolve())
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run precompute/{script}")
    with np.load(path) as f:
        g = {k: np.asarray(f[k]) for k in f.files if k != "meta"}
        g["meta"] = json.loads(str(f["meta"]))
    _GRID_CACHE[key] = g
    return g


def write_zeta_grid(tables_dir, *, T_K, log_met, c_to_o, wavelength_um,
                    zeta, mu, meta):
    """Planet-independent photospheric log-pressure grid.

    zeta is stored float16 (abs. precision <= 0.016 at zeta <= 16, i.e. < 2 %
    of a scale height) at the native model resolution; mu float32.
    """
    _GRID_CACHE.pop(str((Path(tables_dir) / ZETA_GRID).resolve()), None)
    return _write_grid(Path(tables_dir) / ZETA_GRID, meta,
                       {"zeta": np.float16, "mu": np.float32},
                       T_K=T_K, log_met=log_met, c_to_o=c_to_o,
                       wavelength_um=wavelength_um, zeta=zeta, mu=mu)


def load_zeta_grid(tables_dir=None):
    return _load_grid(resolve_tables_dir(tables_dir) / ZETA_GRID,
                      "make_zeta_grid.py")


def write_phoenix_grid(tables_dir, *, teff_K, logg, feh, wavelength_um,
                       log_flam, meta):
    """PHOENIX SED mini-grid (float32 log10 F_lambda)."""
    _GRID_CACHE.pop(str((Path(tables_dir) / PHOENIX_GRID).resolve()), None)
    return _write_grid(Path(tables_dir) / PHOENIX_GRID, meta,
                       {"log_flam": np.float32},
                       teff_K=teff_K, logg=logg, feh=feh,
                       wavelength_um=wavelength_um, log_flam=log_flam)


def load_phoenix_grid(tables_dir=None):
    return _load_grid(resolve_tables_dir(tables_dir) / PHOENIX_GRID,
                      "make_phoenix_grid.py")


def _planet_entry(index, planet):
    for name, entry in index["planets"].items():
        if name == planet or entry["slug"] == slugify(planet):
            return name, entry
    raise KeyError(
        f"Unknown planet {planet!r}. Available: {sorted(index['planets'])}")


def load_model(tables_dir, planet, scenario):
    index = load_index(tables_dir)
    name, entry = _planet_entry(index, planet)
    if scenario not in entry["scenarios"]:
        raise KeyError(
            f"Unknown scenario {scenario!r} for {name!r}. "
            f"Available: {entry['scenarios']}")
    path = _model_path(tables_dir, name, scenario)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is indexed but missing; re-run "
            "precompute/make_model_tables.py and tables.rebuild_index().")
    return _read(path, "wavelength_um", "transit_depth")


def load_star(tables_dir, planet):
    path = Path(tables_dir) / "star" / f"{slugify(planet)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; re-run precompute/make_model_tables.py")
    return _read(path, "wavelength_um", "f_lambda")


def load_noise(tables_dir, planet):
    index = load_index(tables_dir)
    name, _ = _planet_entry(index, planet)
    path = Path(tables_dir) / "noise" / f"{slugify(name)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; re-run precompute/make_noise_tables.py "
            "and tables.rebuild_index().")
    return _read(path, "wavelength_um", "depth_err_1transit")


def load_ld(tables_dir, planet):
    index = load_index(tables_dir)
    name, _ = _planet_entry(index, planet)
    path = Path(tables_dir) / "ld" / f"{slugify(name)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; re-run precompute/make_ld_tables.py "
            "and tables.rebuild_index().")
    return _read(path, "wavelength_um", "u1", "u2")


def rebuild_index(tables_dir):
    tdir = Path(tables_dir)
    planets = {}
    for path in sorted((tdir / "models").glob("*.npz")):
        *_, meta = _read(path, "wavelength_um")
        name = meta["planet"]
        entry = planets.setdefault(name, {"slug": slugify(name),
                                          "scenarios": [], "has_noise": False})
        entry["scenarios"].append(meta["scenario"])
    for name, entry in planets.items():
        entry["scenarios"] = sorted(entry["scenarios"])
        entry["has_noise"] = (tdir / "noise" / f"{entry['slug']}.npz").exists()
        entry["has_ld"] = (tdir / "ld" / f"{entry['slug']}.npz").exists()
    index = {"version": SCHEMA_VERSION, "planets": planets,
             "atmo": {"zeta_grid": (tdir / ZETA_GRID).exists()},
             "star_grid": {"phoenix_grid": (tdir / PHOENIX_GRID).exists()}}
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    return index


def load_index(tables_dir):
    path = Path(resolve_tables_dir(tables_dir)) / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run tables.rebuild_index()")
    index = json.loads(path.read_text())
    if index.get("version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Table schema version {index.get('version')} != supported "
            f"{SCHEMA_VERSION}; regenerate tables (precompute/) or update terminator.")
    return index


def list_planets(tables_dir=None):
    return sorted(load_index(tables_dir)["planets"])


def list_scenarios(planet, tables_dir=None):
    _, entry = _planet_entry(load_index(tables_dir), planet)
    return entry["scenarios"]
