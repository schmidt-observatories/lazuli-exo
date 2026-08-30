"""Lookup-table storage for terminator.

Layout under a tables directory:
    index.json
    models/<planet-slug>__<scenario>.npz   (wavelength_um, transit_depth, meta)
    star/<planet-slug>.npz                 (wavelength_um, f_lambda, meta)
    noise/<planet-slug>.npz                (wavelength_um, depth_err_1transit, meta)
    ld/<planet-slug>.npz                   (wavelength_um, u1, u2, meta)

All wavelengths are in micron. `meta` is a JSON dict stored as a string array.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def resolve_tables_dir(tables_dir=None):
    if tables_dir is not None:
        return Path(tables_dir)
    env = os.environ.get("TERMINATOR_TABLES")
    if env:
        return Path(env)
    default = Path(__file__).resolve().parents[2] / "data" / "tables"
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
    index = {"version": SCHEMA_VERSION, "planets": planets}
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
