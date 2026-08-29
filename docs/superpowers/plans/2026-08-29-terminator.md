# terminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the terminator package: a numpy-only runtime that simulates Lazuli IFS transmission spectroscopy from precomputed POSEIDON model tables and slicersim noise tables, for 14 showcase planets × 5 scenarios.

**Architecture:** Two-stage precompute. Stage 1 (dedicated `poseidon` conda env) writes high-resolution transit-depth spectra and PHOENIX host-star spectra per planet×scenario. Stage 2 (py313 env: slicersim + lazuli_transit) writes single-transit depth-uncertainty tables. The runtime package under `src/terminator/` imports only numpy (matplotlib as `[plot]` extra) and combines the tables.

**Tech Stack:** Python, numpy, pytest; precompute only: POSEIDON (own conda env), slicersim, lazuli_transit, jaxoplanet, astroquery, sncosmo.

**Spec:** `docs/superpowers/specs/2026-08-29-terminator-design.md`

## Global Constraints

- Runtime package (`src/terminator/`) imports **numpy only**; `plotting.py` may import matplotlib lazily. NO scipy, POSEIDON, slicersim, jax, jaxoplanet, pandas, astropy anywhere under `src/terminator/`.
- **No GPL dependencies** anywhere (jaxoplanet not batman; POSEIDON is BSD; slicersim is checked Apache-2.0).
- Dev/test environment: conda `py313` → `/Users/gudmundurstefansson/anaconda3/envs/py313/bin/python`. All pytest runs use this env.
- Stage-1 precompute runs ONLY in the new `poseidon` conda env (`~/anaconda3/envs/poseidon/bin/python`) with env vars `POSEIDON_input_data=$HOME/Downloads/inputs/` and `PYSYN_CDBS=$HOME/Downloads/inputs/stellar_grids`.
- Table schema version integer: `1` (in `index.json` and checked by the reader).
- IFS instrument config: slicersim `instrument="lazuli_cbe"`. Native grid ≈ 4000–16993 Å, 525 channels (⇒ native sampling resolution R≈360; requested output R must be lower).
- Model wavelength grid: constant-R, `WL_MIN_UM=0.35`, `WL_MAX_UM=1.85`, `R_MODEL=10000`.
- Wavelength unit everywhere in tables and runtime: **µm**. slicersim works in Å — convert at the boundary in `make_noise_tables.py` only.
- Working directory for all commands: `/Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/terminator`.
- Commit after every task (repo already initialized, `main` branch).

---

### Task 1: Package scaffold + table I/O (`tables.py`)

**Files:**
- Create: `pyproject.toml`
- Create: `src/terminator/__init__.py`
- Create: `src/terminator/tables.py`
- Create: `tests/test_tables.py`
- Create: `.gitignore`

**Interfaces:**
- Produces (used by every later task):
  - `terminator.tables.slugify(name: str) -> str`
  - `terminator.tables.resolve_tables_dir(tables_dir=None) -> pathlib.Path`
  - `terminator.tables.write_model_table(tables_dir, planet, scenario, wavelength_um, transit_depth, meta: dict) -> Path`
  - `terminator.tables.write_star_table(tables_dir, planet, wavelength_um, f_lambda, meta: dict) -> Path`
  - `terminator.tables.write_noise_table(tables_dir, planet, wavelength_um, depth_err_1transit, meta: dict) -> Path`
  - `terminator.tables.load_model(tables_dir, planet, scenario) -> (np.ndarray, np.ndarray, dict)`
  - `terminator.tables.load_star(tables_dir, planet) -> (np.ndarray, np.ndarray, dict)`
  - `terminator.tables.load_noise(tables_dir, planet) -> (np.ndarray, np.ndarray, dict)`
  - `terminator.tables.rebuild_index(tables_dir) -> dict` (writes `index.json`)
  - `terminator.tables.load_index(tables_dir) -> dict` (raises on version mismatch)
  - `terminator.tables.list_planets(tables_dir=None) -> list[str]`
  - `terminator.tables.list_scenarios(planet, tables_dir=None) -> list[str]`
  - `terminator.tables.SCHEMA_VERSION = 1`

- [ ] **Step 1: Write scaffold files**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "terminator"
version = "0.1.0"
description = "Lazuli IFS exoplanet transmission spectroscopy simulator (lookup-table runtime)"
requires-python = ">=3.10"
dependencies = ["numpy"]

[project.optional-dependencies]
plot = ["matplotlib"]
test = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]
```

`.gitignore`:

```
__pycache__/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
```

`src/terminator/__init__.py` (temporary; Task 4 extends it):

```python
"""terminator: Lazuli IFS exoplanet transmission spectroscopy simulator."""
from . import tables
from .tables import list_planets, list_scenarios

__all__ = ["tables", "list_planets", "list_scenarios"]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_tables.py`:

```python
import json
import numpy as np
import pytest

from terminator import tables


@pytest.fixture
def tdir(tmp_path):
    return tmp_path / "tables"


def _write_minimal(tdir):
    wl = np.linspace(0.4, 1.7, 50)
    tables.write_model_table(tdir, "Testar-1 b", "clear", wl, np.full(50, 0.01),
                             meta={"note": "fixture"})
    tables.write_model_table(tdir, "Testar-1 b", "flat", wl, np.full(50, 0.01),
                             meta={})
    tables.write_noise_table(tdir, "Testar-1 b", wl, np.full(50, 1e-4),
                             meta={"t14_s": 7200.0})
    tables.rebuild_index(tdir)


def test_slugify():
    assert tables.slugify("WASP-39 b") == "wasp-39-b"
    assert tables.slugify("TRAPPIST-1 e") == "trappist-1-e"


def test_model_roundtrip(tdir):
    wl = np.linspace(0.4, 1.7, 50)
    depth = np.full(50, 0.0146)
    path = tables.write_model_table(tdir, "Testar-1 b", "clear", wl, depth,
                                    meta={"scenario": "clear"})
    assert path.exists()
    wl2, d2, meta = tables.load_model(tdir, "Testar-1 b", "clear")
    np.testing.assert_allclose(wl2, wl)
    np.testing.assert_allclose(d2, depth)
    assert meta["scenario"] == "clear"


def test_noise_roundtrip(tdir):
    wl = np.linspace(0.4, 1.7, 20)
    err = np.full(20, 2e-4)
    tables.write_noise_table(tdir, "Testar-1 b", wl, err, meta={})
    wl2, e2, _ = tables.load_noise(tdir, "Testar-1 b")
    np.testing.assert_allclose(e2, err)


def test_index_and_listings(tdir):
    _write_minimal(tdir)
    idx = tables.load_index(tdir)
    assert idx["version"] == tables.SCHEMA_VERSION
    assert tables.list_planets(tdir) == ["Testar-1 b"]
    assert tables.list_scenarios("Testar-1 b", tdir) == ["clear", "flat"]


def test_version_mismatch_rejected(tdir):
    _write_minimal(tdir)
    idx = json.loads((tdir / "index.json").read_text())
    idx["version"] = 999
    (tdir / "index.json").write_text(json.dumps(idx))
    with pytest.raises(RuntimeError, match="version"):
        tables.load_index(tdir)


def test_unknown_planet_lists_options(tdir):
    _write_minimal(tdir)
    with pytest.raises(KeyError, match="Testar-1 b"):
        tables.load_model(tdir, "Nope-1 b", "clear")


def test_unknown_scenario_lists_options(tdir):
    _write_minimal(tdir)
    with pytest.raises(KeyError, match="clear"):
        tables.load_model(tdir, "Testar-1 b", "nope")


def test_resolve_tables_dir_env(tdir, monkeypatch):
    _write_minimal(tdir)
    monkeypatch.setenv("TERMINATOR_TABLES", str(tdir))
    assert tables.resolve_tables_dir(None) == tdir
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_tables.py -v`
Expected: FAIL / collection errors ("No module named terminator" until install; after `pip install -e .`, AttributeError on missing functions).

First install editable: `~/anaconda3/envs/py313/bin/python -m pip install -e ".[test,plot]"`

- [ ] **Step 4: Implement `src/terminator/tables.py`**

```python
"""Lookup-table storage for terminator.

Layout under a tables directory:
    index.json
    models/<planet-slug>__<scenario>.npz   (wavelength_um, transit_depth, meta)
    star/<planet-slug>.npz                 (wavelength_um, f_lambda, meta)
    noise/<planet-slug>.npz                (wavelength_um, depth_err_1transit, meta)

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_tables.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/terminator tests/test_tables.py
git commit -m "feat: package scaffold and lookup-table I/O"
```

---

### Task 2: Frozen target registry (`targets.py` via `precompute/freeze_targets.py`)

**Files:**
- Create: `precompute/freeze_targets.py`
- Create: `src/terminator/targets.py` (generated by the script, then committed)
- Test: `tests/test_targets.py`

**Interfaces:**
- Produces:
  - `terminator.targets.TARGETS: dict[str, dict]` — keys are display names (e.g. `"WASP-39 b"`); each value has float fields `R_p_RJ, M_p_MJ, R_s_RSun, T_eff_K, log_g_star, Met_star, T_eq_K, period_d, t14_s, depth, a_over_rs, inc_deg, ecc, w_deg, J_mag` and str fields `name, slug, provenance`.
  - `terminator.targets.get_target(name) -> dict` (KeyError lists options)
  - `terminator.targets.PLANET_NAMES: list[str]` (the 14 names, sorted)

**Details:** The 14 planets: HD 209458 b, HD 189733 b, WASP-39 b, WASP-96 b, WASP-107 b, WASP-121 b, WASP-12 b, HAT-P-11 b, GJ 1214 b, GJ 3470 b, K2-18 b, TOI-270 d, TRAPPIST-1 e, LHS 1140 b.

`freeze_targets.py` queries the NASA Exoplanet Archive TAP service directly via astroquery (the lazuli_transit CSV cache lacks `pl_bmassj`, `pl_trandur`, `pl_eqt`, `st_logg`, `st_met`), because we need columns beyond `lazuli_transit.ARCHIVE_COLUMNS`. Derived fallbacks when a column is null: `T_eq = st_teff * sqrt(0.5 / a_over_rs)`; `t14 = (period/π) * asin( sqrt((1+rp)² − b²) / (a_over_rs · sin i) )` with `b = a_over_rs·cos i`; `depth = rp²`. A hand-maintained `OVERRIDES` dict at the top of the script patches any remaining gaps (fill during execution from the archive's planet pages; every override carries a comment with its source). `Met_star` defaults to 0.0 and `ecc`/`w_deg` to 0.0/90.0 when null.

- [ ] **Step 1: Write the failing test**

`tests/test_targets.py`:

```python
import math
import pytest

from terminator import targets

REQUIRED = ["R_p_RJ", "M_p_MJ", "R_s_RSun", "T_eff_K", "log_g_star",
            "Met_star", "T_eq_K", "period_d", "t14_s", "depth",
            "a_over_rs", "inc_deg", "ecc", "w_deg", "J_mag"]


def test_fourteen_planets():
    assert len(targets.PLANET_NAMES) == 14
    assert "WASP-39 b" in targets.PLANET_NAMES
    assert "TRAPPIST-1 e" in targets.PLANET_NAMES


def test_all_fields_finite():
    for name in targets.PLANET_NAMES:
        t = targets.get_target(name)
        for key in REQUIRED:
            assert math.isfinite(t[key]), f"{name}.{key} = {t[key]}"


def test_physical_sanity():
    for name in targets.PLANET_NAMES:
        t = targets.get_target(name)
        assert 0.001 < t["depth"] < 0.1, name        # 0.1%–10% transits
        assert 60 * 10 < t["t14_s"] < 60 * 60 * 12, name
        assert 1 < t["a_over_rs"] < 300, name
        assert 2000 < t["T_eff_K"] < 7000, name
        assert 4 < t["J_mag"] < 13, name


def test_get_target_unknown():
    with pytest.raises(KeyError, match="WASP-39 b"):
        targets.get_target("Kepler-999x b")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_targets.py -v`
Expected: FAIL (`terminator.targets` missing).

- [ ] **Step 3: Write `precompute/freeze_targets.py`**

```python
"""Generate src/terminator/targets.py from the NASA Exoplanet Archive (PSCompPars).

Run (needs astroquery, network):
    ~/anaconda3/envs/py313/bin/python precompute/freeze_targets.py
"""
import math
from pathlib import Path

PLANETS = [
    "HD 209458 b", "HD 189733 b", "WASP-39 b", "WASP-96 b", "WASP-107 b",
    "WASP-121 b", "WASP-12 b", "HAT-P-11 b", "GJ 1214 b", "GJ 3470 b",
    "K2-18 b", "TOI-270 d", "TRAPPIST-1 e", "LHS 1140 b",
]

# Literature patches for archive gaps: {"planet": {"field": value}}.
# Add entries as the run below reports missing fields; comment each source.
OVERRIDES = {}

COLS = ("pl_name, pl_radj, pl_bmassj, pl_orbper, pl_trandur, pl_eqt, "
        "pl_ratror, pl_ratdor, pl_orbincl, pl_orbeccen, pl_orblper, "
        "st_rad, st_teff, st_logg, st_met, sy_jmag")

OUT = Path(__file__).resolve().parents[1] / "src" / "terminator" / "targets.py"

HEADER = '''"""Frozen showcase-target registry (GENERATED by precompute/freeze_targets.py).

Source: NASA Exoplanet Archive PSCompPars + OVERRIDES in freeze_targets.py.
Do not edit by hand; re-run the script to refresh.
"""

TARGETS = '''

FOOTER = '''

PLANET_NAMES = sorted(TARGETS)


def get_target(name):
    try:
        return TARGETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown target {name!r}. Available: {PLANET_NAMES}") from None
'''


def fetch():
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    names = ",".join(f"'{p}'" for p in PLANETS)
    return NasaExoplanetArchive.query_criteria(
        table="pscomppars", select=COLS, where=f"pl_name in ({names})")


def val(row, col):
    v = row[col]
    try:
        v = float(v.value if hasattr(v, "value") else v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def build_entry(row, name):
    ov = OVERRIDES.get(name, {})
    g = lambda field, raw: ov.get(field, raw)

    rp_ratio = val(row, "pl_ratror")
    a_rs = val(row, "pl_ratdor")
    inc = g("inc_deg", val(row, "pl_orbincl") or 90.0)
    teff = g("T_eff_K", val(row, "st_teff"))
    per = g("period_d", val(row, "pl_orbper"))

    t_eq = val(row, "pl_eqt")
    if t_eq is None and teff and a_rs:
        t_eq = teff * math.sqrt(0.5 / a_rs)

    t14 = val(row, "pl_trandur")          # hours in the archive
    t14_s = t14 * 3600.0 if t14 is not None else None
    if t14_s is None and per and a_rs and rp_ratio:
        b = a_rs * math.cos(math.radians(inc))
        arg = math.sqrt(max((1 + rp_ratio) ** 2 - b ** 2, 0.0)) / (
            a_rs * math.sin(math.radians(inc)))
        t14_s = per * 86400.0 / math.pi * math.asin(min(arg, 1.0))

    entry = {
        "name": name,
        "slug": name.lower().replace(" ", "-"),
        "R_p_RJ": g("R_p_RJ", val(row, "pl_radj")),
        "M_p_MJ": g("M_p_MJ", val(row, "pl_bmassj")),
        "R_s_RSun": g("R_s_RSun", val(row, "st_rad")),
        "T_eff_K": teff,
        "log_g_star": g("log_g_star", val(row, "st_logg")),
        "Met_star": g("Met_star", val(row, "st_met") or 0.0),
        "T_eq_K": g("T_eq_K", t_eq),
        "period_d": per,
        "t14_s": g("t14_s", t14_s),
        "depth": g("depth", rp_ratio ** 2 if rp_ratio else None),
        "a_over_rs": g("a_over_rs", a_rs),
        "inc_deg": inc,
        "ecc": g("ecc", val(row, "pl_orbeccen") or 0.0),
        "w_deg": g("w_deg", val(row, "pl_orblper") or 90.0),
        "J_mag": g("J_mag", val(row, "sy_jmag")),
        "provenance": "NEA PSCompPars"
                      + (" + overrides" if name in OVERRIDES else ""),
    }
    missing = [k for k, v in entry.items() if v is None]
    if missing:
        raise SystemExit(f"{name}: missing {missing} — add to OVERRIDES")
    return entry


def main():
    table = fetch()
    rows = {str(r["pl_name"]): r for r in table}
    targets = {}
    for name in PLANETS:
        if name not in rows:
            raise SystemExit(f"{name} not returned by the archive query")
        targets[name] = build_entry(rows[name], name)
    import pprint
    body = pprint.pformat(targets, width=79, sort_dicts=True)
    OUT.write_text(HEADER + body + FOOTER)
    print(f"Wrote {OUT} with {len(targets)} targets")


if __name__ == "__main__":
    main()
```

Note on `slugify` duplication: the generated file stores the slug literal (same rule as `tables.slugify` for these names) so `targets.py` stays import-free.

- [ ] **Step 4: Run the freezer**

Run: `~/anaconda3/envs/py313/bin/python precompute/freeze_targets.py`
(If astroquery is missing: `~/anaconda3/envs/py313/bin/python -m pip install astroquery`.)
If it exits with "missing [...] — add to OVERRIDES": look the values up (NASA Exoplanet Archive planet page for that planet), add them to `OVERRIDES` with a source comment, re-run. Repeat until it writes `targets.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_targets.py -v`
Expected: all PASS. If a sanity bound fails, inspect the archive value; fix via `OVERRIDES` if it is an archive quirk, or relax the bound only if the true value genuinely lies outside it (comment why).

- [ ] **Step 6: Commit**

```bash
git add precompute/freeze_targets.py src/terminator/targets.py tests/test_targets.py
git commit -m "feat: frozen showcase target registry (14 planets)"
```

---

### Task 3: Wavelength binning (`binning.py`)

**Files:**
- Create: `src/terminator/binning.py`
- Test: `tests/test_binning.py`

**Interfaces:**
- Produces:
  - `terminator.binning.constant_R_edges(wl_min, wl_max, R) -> np.ndarray` — log-spaced bin edges with Δλ/λ = 1/R; last edge ≤ wl_max is kept, so coverage never exceeds [wl_min, wl_max].
  - `terminator.binning.bin_centers(edges) -> np.ndarray` — geometric means.
  - `terminator.binning.bin_model(wl, depth, edges) -> np.ndarray` — unweighted mean of model samples inside each bin; raises `ValueError` naming the empty bin if any bin catches no samples.
  - `terminator.binning.bin_sigma(wl, sigma, edges) -> np.ndarray` — inverse-variance combination `1/sqrt(sum(1/σ_i²))` of noise channels inside each bin; same empty-bin `ValueError` (message says "requested R exceeds the native noise sampling").

- [ ] **Step 1: Write the failing tests**

`tests/test_binning.py`:

```python
import numpy as np
import pytest

from terminator import binning


def test_constant_R_edges_spacing():
    edges = binning.constant_R_edges(0.4, 1.7, 50)
    ratios = edges[1:] / edges[:-1]
    np.testing.assert_allclose(ratios, ratios[0])
    assert edges[0] == pytest.approx(0.4)
    assert edges[-1] <= 1.7 + 1e-12
    # Δλ/λ ≈ 1/R at each bin
    assert ratios[0] == pytest.approx(1 + 1.0 / 50, rel=1e-6)


def test_bin_model_conserves_constant():
    wl = np.geomspace(0.4, 1.7, 5000)
    depth = np.full_like(wl, 0.0146)
    edges = binning.constant_R_edges(0.4, 1.7, 100)
    binned = binning.bin_model(wl, depth, edges)
    np.testing.assert_allclose(binned, 0.0146)


def test_bin_model_mean_of_linear():
    # binning a linear function gives the mean of the samples in each bin
    wl = np.linspace(1.0, 2.0, 10001)
    edges = np.array([1.0, 1.5, 2.0])
    binned = binning.bin_model(wl, wl.copy(), edges)
    np.testing.assert_allclose(binned, [1.25, 1.75], rtol=1e-3)


def test_bin_sigma_inverse_variance():
    wl = np.linspace(1.0, 2.0, 4)
    sigma = np.array([2.0, 2.0, 2.0, 2.0])
    edges = np.array([0.9, 2.1])  # one bin catching all 4 channels
    out = binning.bin_sigma(wl, sigma, edges)
    np.testing.assert_allclose(out, [1.0])  # 2/sqrt(4)


def test_empty_bin_raises():
    wl = np.array([1.0, 2.0])
    edges = np.array([1.0, 1.2, 1.4, 2.1])
    with pytest.raises(ValueError, match="native noise sampling"):
        binning.bin_sigma(wl, np.ones(2), edges)
    with pytest.raises(ValueError):
        binning.bin_model(wl, np.ones(2), edges)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_binning.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/terminator/binning.py`**

```python
"""Constant-R wavelength binning for model spectra and noise channels."""
import numpy as np


def constant_R_edges(wl_min, wl_max, R):
    step = 1.0 + 1.0 / float(R)
    n = int(np.floor(np.log(wl_max / wl_min) / np.log(step)))
    if n < 1:
        raise ValueError(f"R={R} gives no complete bin in [{wl_min}, {wl_max}]")
    return wl_min * step ** np.arange(n + 1)


def bin_centers(edges):
    edges = np.asarray(edges, dtype=float)
    return np.sqrt(edges[:-1] * edges[1:])


def _bin_indices(wl, edges, what):
    idx = np.digitize(wl, edges) - 1
    inside = (idx >= 0) & (idx < len(edges) - 1)
    counts = np.bincount(idx[inside], minlength=len(edges) - 1)
    if (counts == 0).any():
        empty = int(np.argmax(counts == 0))
        raise ValueError(
            f"Bin {empty} ([{edges[empty]:.4f}, {edges[empty + 1]:.4f}] um) "
            f"contains no {what} samples; requested R exceeds the native "
            "noise sampling" if what == "noise" else
            f"Bin {empty} ([{edges[empty]:.4f}, {edges[empty + 1]:.4f}] um) "
            "contains no model samples; use a coarser R")
    return idx, inside, counts


def bin_model(wl, depth, edges):
    wl, depth = np.asarray(wl, float), np.asarray(depth, float)
    idx, inside, counts = _bin_indices(wl, edges, "model")
    sums = np.bincount(idx[inside], weights=depth[inside],
                       minlength=len(edges) - 1)
    return sums / counts


def bin_sigma(wl, sigma, edges):
    wl, sigma = np.asarray(wl, float), np.asarray(sigma, float)
    idx, inside, _ = _bin_indices(wl, edges, "noise")
    inv_var = np.bincount(idx[inside], weights=1.0 / sigma[inside] ** 2,
                          minlength=len(edges) - 1)
    return 1.0 / np.sqrt(inv_var)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_binning.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/terminator/binning.py tests/test_binning.py
git commit -m "feat: constant-R binning with inverse-variance noise combination"
```

---

### Task 4: Observation assembly (`observe.py`)

**Files:**
- Create: `src/terminator/observe.py`
- Create: `tests/conftest.py` (shared synthetic-table fixture)
- Modify: `src/terminator/__init__.py`
- Test: `tests/test_observe.py`

**Interfaces:**
- Consumes: `tables.load_model/load_noise/list_*`, `binning.*` (Tasks 1, 3).
- Produces:
  - `terminator.observe(planet, scenario="clear", n_transits=1, R=100, tables_dir=None) -> SimulatedSpectrum`
  - `SimulatedSpectrum` attributes: `planet, scenario, n_transits, R, wavelength, depth, depth_err, model_wavelength, model_depth, flat_depth (float)`; methods: `detect() -> dict` (Task 5 wires it), `simulate_points(seed=None) -> np.ndarray`, `plot(ax=None, **kw)` (Task 6 wires it).
  - `terminator.__init__` re-exports: `observe`, `SimulatedSpectrum`, `list_planets`, `list_scenarios`.

- [ ] **Step 1: Write the shared fixture**

`tests/conftest.py`:

```python
import numpy as np
import pytest

from terminator import tables

PLANET = "Testar-1 b"
BASE_DEPTH = 0.0100
BUMP = 0.0004        # 400 ppm gaussian feature at 1.4 um
SIGMA_CH = 1.0e-4    # noise per native channel, one transit


@pytest.fixture
def fixture_tables(tmp_path):
    """Synthetic table set with analytically known content."""
    tdir = tmp_path / "tables"
    wl_m = np.geomspace(0.4, 1.7, 8000)
    depth = BASE_DEPTH + BUMP * np.exp(-0.5 * ((wl_m - 1.4) / 0.05) ** 2)
    tables.write_model_table(tdir, PLANET, "clear", wl_m, depth,
                             meta={"scenario": "clear"})
    tables.write_model_table(tdir, PLANET, "flat", wl_m,
                             np.full_like(wl_m, BASE_DEPTH),
                             meta={"scenario": "flat"})
    wl_n = np.geomspace(0.4, 1.7, 500)
    tables.write_noise_table(tdir, PLANET, wl_n,
                             np.full_like(wl_n, SIGMA_CH),
                             meta={"t14_s": 7200.0})
    tables.rebuild_index(tdir)
    return tdir
```

- [ ] **Step 2: Write the failing tests**

`tests/test_observe.py`:

```python
import numpy as np
import pytest

import terminator
from conftest import PLANET, BASE_DEPTH, SIGMA_CH  # pytest puts tests/ on sys.path


def test_observe_shapes(fixture_tables):
    obs = terminator.observe(PLANET, scenario="clear", n_transits=1, R=50,
                         tables_dir=fixture_tables)
    assert obs.wavelength.shape == obs.depth.shape == obs.depth_err.shape
    assert np.isfinite(obs.depth).all() and np.isfinite(obs.depth_err).all()
    assert obs.planet == PLANET and obs.scenario == "clear"


def test_flat_scenario_is_flat(fixture_tables):
    obs = terminator.observe(PLANET, scenario="flat", R=50,
                         tables_dir=fixture_tables)
    np.testing.assert_allclose(obs.depth, BASE_DEPTH, rtol=1e-12)


def test_error_scaling_with_transits(fixture_tables):
    o1 = terminator.observe(PLANET, n_transits=1, R=50, tables_dir=fixture_tables)
    o9 = terminator.observe(PLANET, n_transits=9, R=50, tables_dir=fixture_tables)
    np.testing.assert_allclose(o1.depth_err / o9.depth_err, 3.0)


def test_error_magnitude(fixture_tables):
    # 500 uniform channels at R such that ~10 channels/bin
    obs = terminator.observe(PLANET, n_transits=1, R=50, tables_dir=fixture_tables)
    # every bin: sigma = SIGMA_CH / sqrt(n_channels_in_bin); just check bounds
    assert (obs.depth_err < SIGMA_CH).all()
    assert (obs.depth_err > SIGMA_CH / np.sqrt(30)).all()


def test_r_too_high_raises(fixture_tables):
    with pytest.raises(ValueError, match="native noise sampling"):
        terminator.observe(PLANET, R=2000, tables_dir=fixture_tables)


def test_simulate_points_reproducible(fixture_tables):
    obs = terminator.observe(PLANET, R=50, tables_dir=fixture_tables)
    a = obs.simulate_points(seed=42)
    b = obs.simulate_points(seed=42)
    np.testing.assert_array_equal(a, b)
    scatter = np.std((a - obs.depth) / obs.depth_err)
    assert 0.7 < scatter < 1.3


def test_unknown_planet_raises(fixture_tables):
    with pytest.raises(KeyError, match="Testar-1 b"):
        terminator.observe("Nope-1 b", tables_dir=fixture_tables)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_observe.py -v`
Expected: FAIL (`terminator.observe` missing).

- [ ] **Step 4: Implement `src/terminator/observe.py`**

```python
"""Assemble a simulated Lazuli IFS transmission-spectroscopy observation."""
from dataclasses import dataclass, field

import numpy as np

from . import binning, tables


@dataclass
class SimulatedSpectrum:
    planet: str
    scenario: str
    n_transits: int
    R: int
    wavelength: np.ndarray       # bin centers [um]
    depth: np.ndarray            # binned model transit depth
    depth_err: np.ndarray        # 1-sigma per bin, n_transits included
    model_wavelength: np.ndarray  # native-resolution model [um]
    model_depth: np.ndarray
    flat_depth: float            # (Rp/Rs)^2 baseline from the 'flat' table
    meta: dict = field(default_factory=dict)

    def simulate_points(self, seed=None):
        """One noisy realization of the binned spectrum."""
        rng = np.random.default_rng(seed)
        return self.depth + rng.normal(0.0, self.depth_err)

    def detect(self):
        from .detect import detect_features
        return detect_features(self.wavelength, self.depth, self.depth_err)

    def plot(self, ax=None, **kwargs):
        from .plotting import plot_spectrum
        return plot_spectrum(self, ax=ax, **kwargs)


def observe(planet, scenario="clear", n_transits=1, R=100, tables_dir=None):
    """Simulate observing `planet` under `scenario` for `n_transits` transits.

    Returns a SimulatedSpectrum with the noiseless binned model in `.depth`
    and per-bin 1-sigma uncertainties in `.depth_err` (analytic single-transit
    depth precision from the noise tables, scaled by 1/sqrt(n_transits)).
    """
    if n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    tdir = tables.resolve_tables_dir(tables_dir)

    wl_m, d_m, meta_m = tables.load_model(tdir, planet, scenario)
    wl_n, e_n, meta_n = tables.load_noise(tdir, planet)
    _, d_flat, _ = tables.load_model(tdir, planet, "flat")

    edges = binning.constant_R_edges(wl_n.min(), wl_n.max(), R)
    depth = binning.bin_model(wl_m, d_m, edges)
    err = binning.bin_sigma(wl_n, e_n, edges) / np.sqrt(n_transits)

    keep = (wl_m >= edges[0]) & (wl_m <= edges[-1])
    return SimulatedSpectrum(
        planet=meta_m["planet"], scenario=scenario, n_transits=int(n_transits),
        R=int(R), wavelength=binning.bin_centers(edges), depth=depth,
        depth_err=err, model_wavelength=wl_m[keep], model_depth=d_m[keep],
        flat_depth=float(np.median(d_flat)),
        meta={"model": meta_m, "noise": meta_n})
```

Update `src/terminator/__init__.py`:

```python
"""terminator: Lazuli IFS exoplanet transmission spectroscopy simulator."""
from . import binning, tables
from .observe import SimulatedSpectrum, observe
from .tables import list_planets, list_scenarios

__all__ = ["observe", "SimulatedSpectrum", "list_planets", "list_scenarios",
           "tables", "binning"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_observe.py tests/test_tables.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/terminator/observe.py src/terminator/__init__.py tests/conftest.py tests/test_observe.py
git commit -m "feat: observe() assembling binned spectra with noise scaling"
```

---

### Task 5: Detection significances (`detect.py`)

**Files:**
- Create: `src/terminator/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `SimulatedSpectrum.detect()` calls `detect_features` (already wired in Task 4).
- Produces:
  - `terminator.detect.BAND_MASKS: dict[str, list[tuple[float, float]]]`
  - `terminator.detect.detect_features(wavelength, depth, depth_err) -> dict[str, float]` — keys `"atmosphere"` plus each molecule whose mask overlaps the wavelength range.

**Statistic (documented in the module docstring):** the predicted significance of rejecting the best-fit flat line: with weights `w = 1/σ²` and weighted-mean level `d̄`, `σ_detect = sqrt(Σ w (d − d̄)²)`. `depth` is the noiseless binned model, so this is the expected Δχ² of a flat fit — the standard predicted detection significance. Per-molecule values reuse the same global `d̄` restricted to that molecule's band mask, which double-counts overlapping bands (e.g. H2O/CH4 near 1.15 µm); the docstring states this and names leave-one-out tables as the rigorous upgrade.

Band masks (µm), chosen inside the 0.4–1.7 µm IFS range:

```python
BAND_MASKS = {
    "Na":  [(0.575, 0.605)],
    "K":   [(0.755, 0.785)],
    "H2O": [(0.90, 1.00), (1.10, 1.20), (1.30, 1.55)],
    "CH4": [(0.86, 0.91), (1.10, 1.20), (1.60, 1.70)],
    "CO2": [(1.42, 1.46), (1.56, 1.62)],
}
```

- [ ] **Step 1: Write the failing tests**

`tests/test_detect.py`:

```python
import numpy as np
import pytest

from terminator.detect import BAND_MASKS, detect_features


def test_flat_spectrum_zero_sigma():
    wl = np.linspace(0.4, 1.7, 100)
    out = detect_features(wl, np.full(100, 0.01), np.full(100, 1e-4))
    assert out["atmosphere"] == pytest.approx(0.0, abs=1e-9)


def test_single_bump_analytic():
    # one bin offset by k*sigma among N equal bins:
    # sigma_detect = k * sqrt(1 - 1/N)
    N, k = 100, 5.0
    wl = np.linspace(0.4, 1.7, N)
    sigma = np.full(N, 1e-4)
    depth = np.full(N, 0.01)
    depth[50] += k * sigma[50]
    out = detect_features(wl, depth, sigma)
    assert out["atmosphere"] == pytest.approx(k * np.sqrt(1 - 1 / N), rel=1e-9)


def test_molecule_band_isolation():
    # bump inside the Na band only -> Na sigma ~ atmosphere sigma, K ~ small
    wl = np.linspace(0.4, 1.7, 1000)
    sigma = np.full_like(wl, 1e-4)
    depth = np.full_like(wl, 0.01)
    in_na = (wl > 0.575) & (wl < 0.605)
    depth[in_na] += 5e-4
    out = detect_features(wl, depth, sigma)
    assert out["Na"] > 0.9 * out["atmosphere"]
    assert out["K"] < 0.2 * out["Na"]


def test_masks_within_ifs_range():
    for mol, spans in BAND_MASKS.items():
        for lo, hi in spans:
            assert lo < hi, mol
            assert lo >= 0.4 or hi > 0.4, mol  # at least partially usable
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_detect.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/terminator/detect.py`**

```python
"""Predicted detection significances from a noiseless binned model spectrum.

The statistic is the expected significance of rejecting the best-fit flat
line: sigma = sqrt(sum w (d - dbar)^2), w = 1/sigma_d^2, dbar the weighted
mean depth. Per-molecule values evaluate the same sum restricted to that
molecule's wavelength band mask (relative to the same global dbar). Bands of
different molecules overlap (H2O and CH4 near 1.15 um), so per-molecule
values are approximate; per-molecule leave-one-out model tables are the
rigorous upgrade path.
"""
import numpy as np

BAND_MASKS = {
    "Na":  [(0.575, 0.605)],
    "K":   [(0.755, 0.785)],
    "H2O": [(0.90, 1.00), (1.10, 1.20), (1.30, 1.55)],
    "CH4": [(0.86, 0.91), (1.10, 1.20), (1.60, 1.70)],
    "CO2": [(1.42, 1.46), (1.56, 1.62)],
}


def _mask_array(wavelength, spans):
    m = np.zeros(wavelength.shape, dtype=bool)
    for lo, hi in spans:
        m |= (wavelength >= lo) & (wavelength <= hi)
    return m


def detect_features(wavelength, depth, depth_err):
    wavelength = np.asarray(wavelength, float)
    depth = np.asarray(depth, float)
    w = 1.0 / np.asarray(depth_err, float) ** 2
    dbar = np.sum(w * depth) / np.sum(w)
    chi2 = w * (depth - dbar) ** 2

    out = {"atmosphere": float(np.sqrt(np.sum(chi2)))}
    for mol, spans in BAND_MASKS.items():
        m = _mask_array(wavelength, spans)
        if m.any():
            out[mol] = float(np.sqrt(np.sum(chi2[m])))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_detect.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/terminator/detect.py tests/test_detect.py
git commit -m "feat: flat-line-rejection detection significances with band masks"
```

---

### Task 6: Plotting (`plotting.py`)

**Files:**
- Create: `src/terminator/plotting.py`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: `SimulatedSpectrum` (Task 4).
- Produces: `terminator.plotting.plot_spectrum(obs, ax=None, seed=0, show_model=True) -> matplotlib Axes` (already wired as `obs.plot()`).

**Style:** follow the gks-plotting skill (`plt.style.use('gks')` if that style is registered; wrap in try/except so the package works without it). Depth axis in percent, x log-scaled 0.4–1.7 µm with plain-number ticks, hi-res model as a line, simulated points with error bars, title `"<planet> — <scenario>, N transits"`.

- [ ] **Step 1: Write the failing test**

`tests/test_plotting.py`:

```python
import matplotlib
matplotlib.use("Agg")

import terminator
from conftest import PLANET  # pytest puts tests/ on sys.path


def test_plot_returns_axes(fixture_tables):
    obs = terminator.observe(PLANET, R=50, tables_dir=fixture_tables)
    ax = obs.plot(seed=1)
    assert ax.get_xlabel()
    assert len(ax.lines) + len(ax.containers) >= 2  # model line + errorbars
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_plotting.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/terminator/plotting.py`**

```python
"""Matplotlib plotting for SimulatedSpectrum (requires the [plot] extra)."""


def plot_spectrum(obs, ax=None, seed=0, show_model=True):
    import matplotlib.pyplot as plt
    try:
        plt.style.use("gks")
    except OSError:
        pass

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    if show_model:
        ax.plot(obs.model_wavelength, obs.model_depth * 100.0,
                lw=0.8, alpha=0.7, label=f"model ({obs.scenario})")
    points = obs.simulate_points(seed=seed)
    ax.errorbar(obs.wavelength, points * 100.0, yerr=obs.depth_err * 100.0,
                fmt="o", ms=3.5, lw=1, capsize=2,
                label=f"simulated (R={obs.R})")
    ax.set_xscale("log")
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Transit depth (%)")
    ax.set_title(f"{obs.planet} — {obs.scenario}, "
                 f"{obs.n_transits} transit{'s' if obs.n_transits > 1 else ''}")
    ax.legend(frameon=False)
    return ax
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_plotting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/terminator/plotting.py tests/test_plotting.py
git commit -m "feat: spectrum plotting with simulated error bars"
```

---

### Task 7: Scenario definitions + flat tables (`precompute/scenarios.py`, `make_flat_tables.py`)

**Files:**
- Create: `precompute/scenarios.py`
- Create: `precompute/make_flat_tables.py`
- Test: `tests/test_scenarios.py`
- Generated: `data/tables/models/<slug>__flat.npz` × 14 (committed)

**Interfaces:**
- Consumes: `terminator.targets`, `terminator.tables` (precompute scripts may import terminator; the constraint is one-directional).
- Produces:
  - `precompute/scenarios.py` constants: `WL_MIN_UM = 0.35`, `WL_MAX_UM = 1.85`, `R_MODEL = 10_000`, `PARAM_SPECIES = ["H2O", "CO", "CO2", "CH4", "NH3", "Na", "K"]`, `P_GRID_ARGS = (-7.0, 2.0, 100)` (log10 bar), `P_REF_BAR = 10.0`,
    `SCENARIOS: dict[str, dict]` with keys `clear, cloudy, high_metallicity, hazy`, each value `{"cloud_model": ..., "cloud_type": ..., "cloud_params": [...], "C_to_O": float, "log_Met": float}`.
  - `scenarios.make_wl_grid() -> np.ndarray` — numpy reimplementation of POSEIDON's `wl_grid_constant_R(WL_MIN_UM, WL_MAX_UM, R_MODEL)`: `wl = WL_MIN * exp(arange(n)/R)` up to WL_MAX (verify parity against POSEIDON in Task 8's smoke test).
  - Flat model tables on disk for all 14 planets.

- [ ] **Step 1: Write the failing test**

`tests/test_scenarios.py`:

```python
import importlib.util
import sys
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "scenarios", Path(__file__).resolve().parents[1] / "precompute" / "scenarios.py")
scenarios = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scenarios)


def test_scenario_keys():
    assert set(scenarios.SCENARIOS) == {"clear", "cloudy",
                                        "high_metallicity", "hazy"}
    for cfg in scenarios.SCENARIOS.values():
        assert {"cloud_model", "cloud_type", "cloud_params",
                "C_to_O", "log_Met"} <= set(cfg)


def test_wl_grid():
    wl = scenarios.make_wl_grid()
    assert wl[0] == scenarios.WL_MIN_UM
    assert wl[-1] <= scenarios.WL_MAX_UM
    np.testing.assert_allclose(np.diff(np.log(wl)), 1.0 / scenarios.R_MODEL)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_scenarios.py -v`
Expected: FAIL (file missing).

- [ ] **Step 3: Implement `precompute/scenarios.py`**

```python
"""Single source of truth for the precompute grids and atmosphere scenarios."""
import numpy as np

WL_MIN_UM = 0.35    # margin around the 0.40-1.70 um IFS bandpass
WL_MAX_UM = 1.85
R_MODEL = 10_000

PARAM_SPECIES = ["H2O", "CO", "CO2", "CH4", "NH3", "Na", "K"]
P_GRID_ARGS = (-7.0, 2.0, 100)   # log10(P/bar): min, max, n
P_REF_BAR = 10.0

# POSEIDON MacMad17 cloud_params ordering:
#   deck: [log_P_cloud (log10 bar)]; haze: [log_a, gamma]
SCENARIOS = {
    "clear": dict(cloud_model="cloud-free", cloud_type=None, cloud_params=[],
                  C_to_O=0.55, log_Met=0.0),
    "cloudy": dict(cloud_model="MacMad17", cloud_type="deck",
                   cloud_params=[-3.0], C_to_O=0.55, log_Met=0.0),
    "high_metallicity": dict(cloud_model="cloud-free", cloud_type=None,
                             cloud_params=[], C_to_O=0.55, log_Met=2.0),
    "hazy": dict(cloud_model="MacMad17", cloud_type="haze",
                 cloud_params=[2.0, -6.0], C_to_O=0.55, log_Met=0.0),
}


def make_wl_grid():
    """Constant-R grid matching POSEIDON's wl_grid_constant_R convention."""
    n = int(np.floor(np.log(WL_MAX_UM / WL_MIN_UM) * R_MODEL))
    return WL_MIN_UM * np.exp(np.arange(n + 1) / R_MODEL)
```

- [ ] **Step 4: Implement `precompute/make_flat_tables.py`**

```python
"""Write the analytic 'flat' scenario tables: constant depth = (Rp/Rs)^2.

Run: ~/anaconda3/envs/py313/bin/python precompute/make_flat_tables.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import make_wl_grid

from terminator import tables, targets

TABLES = Path(__file__).resolve().parents[1] / "data" / "tables"


def main():
    wl = make_wl_grid()
    for name in targets.PLANET_NAMES:
        t = targets.get_target(name)
        tables.write_model_table(
            TABLES, name, "flat", wl, np.full_like(wl, t["depth"]),
            meta={"generator": "make_flat_tables.py",
                  "depth": t["depth"], "provenance": t["provenance"]})
        print(f"flat: {name}  depth={t['depth']:.5f}")
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests and the script**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_scenarios.py -v` → PASS
Run: `~/anaconda3/envs/py313/bin/python precompute/make_flat_tables.py`
Expected: 14 lines printed; `data/tables/models/` holds 14 `__flat.npz` files; `data/tables/index.json` lists all 14 planets with `["flat"]`.

Sanity: `~/anaconda3/envs/py313/bin/python -c "import terminator; print(terminator.list_planets())"` → 14 names.

- [ ] **Step 6: Commit**

```bash
git add precompute/scenarios.py precompute/make_flat_tables.py tests/test_scenarios.py data/tables
git commit -m "feat: scenario definitions and analytic flat tables"
```

---

### Task 8: POSEIDON environment + smoke test

**Files:**
- Create: `precompute/poseidon_env.yml`
- Create: `precompute/smoke_poseidon.py`

**Interfaces:**
- Produces: working conda env `poseidon`; verified forward-model recipe (the exact call sequence Task 9 reuses); parity check of `scenarios.make_wl_grid()` against POSEIDON's `wl_grid_constant_R`.

- [ ] **Step 1: Write `precompute/poseidon_env.yml`**

```yaml
name: poseidon
channels: [conda-forge]
dependencies:
  - python=3.11
  - "numpy<2"
  - scipy
  - matplotlib
  - astropy
  - h5py
  - numba
  - pandas
  - tqdm
  - pip
  - pip:
      - pysynphot
      - spectres
```

- [ ] **Step 2: Create the env and install POSEIDON**

```bash
conda env create -f precompute/poseidon_env.yml
~/anaconda3/envs/poseidon/bin/python -m pip install --no-deps -e \
    /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/POSEIDON
~/anaconda3/envs/poseidon/bin/python -c "import POSEIDON.core; print('POSEIDON import OK')"
```

If the import fails on a missing package (POSEIDON's setup.py lists more than it needs for forward models — e.g. pymultinest/mpi4py are retrieval-only), install just the missing package with pip and retry. Do NOT install mpi4py/pymultinest unless the import actually demands them.

- [ ] **Step 3: Write `precompute/smoke_poseidon.py`**

```python
"""Smoke test: one POSEIDON forward model (WASP-39 b, clear) + grid parity.

Run:
  POSEIDON_input_data=$HOME/Downloads/inputs/ \
  PYSYN_CDBS=$HOME/Downloads/inputs/stellar_grids \
  ~/anaconda3/envs/poseidon/bin/python precompute/smoke_poseidon.py
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("POSEIDON_input_data",
                      str(Path.home() / "Downloads" / "inputs") + "/")
os.environ.setdefault("PYSYN_CDBS",
                      str(Path.home() / "Downloads" / "inputs" / "stellar_grids"))

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import (P_GRID_ARGS, P_REF_BAR, PARAM_SPECIES, SCENARIOS,
                       make_wl_grid)

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

lo, hi, n = P_GRID_ARGS
P = np.logspace(lo, hi, n)
T_fine = np.linspace(max(T_eq - 200, 100), T_eq + 200, 10)
log_P_fine = np.linspace(-6.0, 2.0, 40)
opac = read_opacities(model, wl, opacity_treatment="opacity_sampling",
                      T_fine=T_fine, log_P_fine=log_P_fine)

atmosphere = make_atmosphere(planet, model, P, P_REF_BAR, R_p * R_J,
                             PT_params=[T_eq],
                             log_X_params=[cfg["C_to_O"], cfg["log_Met"]],
                             cloud_params=cfg["cloud_params"])
spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                            spectrum_type="transmission")

assert np.isfinite(spectrum).all()
print(f"OK: {len(spectrum)} points, depth range "
      f"{spectrum.min():.5f}-{spectrum.max():.5f} "
      f"(expect roughly 0.018-0.024 for WASP-39 b)")
```

- [ ] **Step 4: Run the smoke test and iterate on API details**

Run (from repo root):

```bash
POSEIDON_input_data=$HOME/Downloads/inputs/ \
PYSYN_CDBS=$HOME/Downloads/inputs/stellar_grids \
~/anaconda3/envs/poseidon/bin/python precompute/smoke_poseidon.py
```

Expected: `OK: ... depth range ...` with values near 2%. The exact `make_atmosphere` argument shapes (e.g. `log_X_params` nesting, `cloud_params` for MacMad17) may differ in this POSEIDON version — if it raises, read the relevant docstring in `GIT/POSEIDON/POSEIDON/core.py` / `parameters.py`, adjust `smoke_poseidon.py` (and mirror any ordering fix into `scenarios.py` comments), and re-run until it prints OK. This smoke script is the authoritative recipe for Task 9.

- [ ] **Step 5: Commit**

```bash
git add precompute/poseidon_env.yml precompute/smoke_poseidon.py
git commit -m "feat: POSEIDON env spec and forward-model smoke test"
```

---

### Task 9: POSEIDON model tables (`make_model_tables.py`) + star spectra

**Files:**
- Create: `precompute/make_model_tables.py`
- Generated: `data/tables/models/<slug>__{clear,cloudy,high_metallicity,hazy}.npz` × 14, `data/tables/star/<slug>.npz` × 14 (committed)

**Interfaces:**
- Consumes: the verified recipe from `smoke_poseidon.py`; `terminator.targets` (import via `sys.path` — terminator is not installed in the poseidon env, so add `src/` to `sys.path`); `terminator.tables` writers (numpy-only, safe in the poseidon env).
- Produces: model tables for all 4 POSEIDON scenarios; star tables `(wavelength_um, f_lambda)` with PHOENIX spectra (`create_star(..., stellar_grid='phoenix', wl=wl, interp_backend='pysynphot')`, reading `star['wl_star'], star['F_star']`; units noted in meta as POSEIDON-native W/m²/m — absolute scale is irrelevant because Stage 2 renormalizes to J mag).

- [ ] **Step 1: Write `precompute/make_model_tables.py`**

```python
"""Stage 1: POSEIDON model spectra + PHOENIX host spectra for all targets.

Run:
  POSEIDON_input_data=$HOME/Downloads/inputs/ \
  PYSYN_CDBS=$HOME/Downloads/inputs/stellar_grids \
  ~/anaconda3/envs/poseidon/bin/python precompute/make_model_tables.py [planet ...]
"""
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

os.environ.setdefault("POSEIDON_input_data",
                      str(Path.home() / "Downloads" / "inputs") + "/")
os.environ.setdefault("PYSYN_CDBS",
                      str(Path.home() / "Downloads" / "inputs" / "stellar_grids"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))      # terminator not pip-installed here
sys.path.insert(0, str(Path(__file__).parent))

from scenarios import (P_GRID_ARGS, P_REF_BAR, PARAM_SPECIES, SCENARIOS,
                       make_wl_grid)
from terminator import tables
from terminator.targets import PLANET_NAMES, get_target

from POSEIDON.constants import M_J, R_J, R_Sun
from POSEIDON.core import (compute_spectrum, create_planet, create_star,
                           define_model, make_atmosphere, read_opacities)

TABLES = ROOT / "data" / "tables"
WL = make_wl_grid()


def star_spectrum(t):
    star = create_star(t["R_s_RSun"] * R_Sun, t["T_eff_K"], t["log_g_star"],
                       t["Met_star"], stellar_grid="phoenix", wl=WL,
                       interp_backend="pysynphot")
    return star, np.asarray(star["wl_star"]), np.asarray(star["F_star"])


def run_planet(name):
    t = get_target(name)
    planet = create_planet(tables.slugify(name), t["R_p_RJ"] * R_J,
                           mass=t["M_p_MJ"] * M_J)
    star, wl_s, f_s = star_spectrum(t)
    tables.write_star_table(TABLES, name, wl_s, f_s,
                            meta={"units": "POSEIDON F_star (W/m^2/m)",
                                  "stellar_grid": "phoenix",
                                  "T_eff_K": t["T_eff_K"]})

    lo, hi, n = P_GRID_ARGS
    P = np.logspace(lo, hi, n)
    T_eq = t["T_eq_K"]
    T_fine = np.linspace(max(T_eq - 200, 100), T_eq + 200, 10)
    log_P_fine = np.linspace(-6.0, 2.0, 40)

    opac = None
    for scen, cfg in SCENARIOS.items():
        model = define_model(f"{tables.slugify(name)}_{scen}",
                             bulk_species=["H2", "He"],
                             param_species=PARAM_SPECIES,
                             PT_profile="isotherm", X_profile="chem_eq",
                             cloud_model=cfg["cloud_model"],
                             **({"cloud_type": cfg["cloud_type"]}
                                if cfg["cloud_type"] else {}))
        if opac is None:   # species identical across scenarios -> reuse
            opac = read_opacities(model, WL,
                                  opacity_treatment="opacity_sampling",
                                  T_fine=T_fine, log_P_fine=log_P_fine)
        atmosphere = make_atmosphere(
            planet, model, P, P_REF_BAR, t["R_p_RJ"] * R_J,
            PT_params=[T_eq],
            log_X_params=[cfg["C_to_O"], cfg["log_Met"]],
            cloud_params=cfg["cloud_params"])  # see NOTE below
        spectrum = compute_spectrum(planet, star, model, atmosphere, opac, WL,
                                    spectrum_type="transmission")
        assert np.isfinite(spectrum).all(), (name, scen)
        tables.write_model_table(
            TABLES, name, scen, WL, spectrum,
            meta={"generator": "make_model_tables.py",
                  "created": str(date.today()), "T_eq_K": T_eq,
                  **{k: v for k, v in cfg.items()}})
        print(f"model: {name} / {scen}  "
              f"depth {spectrum.min():.5f}-{spectrum.max():.5f}")


def main(argv):
    names = argv or PLANET_NAMES
    for name in names:
        run_planet(name)
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main(sys.argv[1:])
```

NOTE: the `make_atmosphere(...)` call mirrors `smoke_poseidon.py`. If Task 8 Step 4 had to adjust the argument shapes (e.g. `log_X_params` nesting or cloud-param ordering) to make the smoke test pass, apply the identical adjustment here before running.

- [ ] **Step 2: Trial run on one planet**

```bash
POSEIDON_input_data=$HOME/Downloads/inputs/ \
PYSYN_CDBS=$HOME/Downloads/inputs/stellar_grids \
~/anaconda3/envs/poseidon/bin/python precompute/make_model_tables.py "WASP-39 b"
```

Expected: 1 star line + 4 model lines, depths ~0.018–0.026, no NaNs. If the PHOENIX call fails for very cool stars (TRAPPIST-1: T_eff≈2566 K may be outside the PHOENIX grid range), catch the exception in `star_spectrum` and fall back to `stellar_grid='blackbody'`, recording `"stellar_grid": "blackbody"` in the meta.

- [ ] **Step 3: Full run**

Same command without a planet argument. Expected ~14 star + 56 model tables (plus the 14 flat from Task 7 → 70 model files total). Check size: `du -sh data/tables` — should be ≲ 30 MB. Verify from py313:

```bash
~/anaconda3/envs/py313/bin/python -c "
import terminator
for p in terminator.list_planets():
    print(p, terminator.list_scenarios(p))"
```

Expected: every planet lists `['clear', 'cloudy', 'flat', 'hazy', 'high_metallicity']`.

- [ ] **Step 4: Commit**

```bash
git add precompute/make_model_tables.py data/tables
git commit -m "feat: POSEIDON model tables and PHOENIX star spectra for all targets"
```

---

### Task 10: Noise tables (`make_noise_tables.py`)

**Files:**
- Create: `precompute/make_noise_tables.py`
- Generated: `data/tables/noise/<slug>.npz` × 14 (committed); refreshed `index.json`

**Interfaces:**
- Consumes: star tables (Task 9), `terminator.targets` (t14_s, J_mag), slicersim `LazuliTarget` (py313).
- Produces: noise tables `(wavelength_um, depth_err_1transit)` on the native IFS grid.

**Physics:** slicersim's default readout gives SNR(λ) for its default `observing_time` t_obs. Ramps are independent exposures, so SNR scales as √t. In-transit SNR for T₁₄: `snr_in = snr_default * sqrt(t14 / t_obs)`; with an equal out-of-transit baseline, per-channel depth error `σ = sqrt(1/snr_in² + 1/snr_out²) = sqrt(2)/snr_in`.

- [ ] **Step 1: Write `precompute/make_noise_tables.py`**

```python
"""Stage 2: per-planet single-transit depth-uncertainty tables via slicersim.

Run: ~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py
"""
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"

from slicersim.lazuli import LazuliTarget

from terminator import tables
from terminator.targets import PLANET_NAMES, get_target

INSTRUMENT = "lazuli_cbe"
NORM_BAND = "2massj"      # sncosmo bandpass for J-mag normalization
ANGSTROM_PER_UM = 1.0e4


def run_planet(name):
    t = get_target(name)
    wl_star_um, f_star, star_meta = tables.load_star(TABLES, name)

    target = LazuliTarget(wl_star_um * ANGSTROM_PER_UM, f_star,
                          mag=t["J_mag"], band=NORM_BAND,
                          instrument=INSTRUMENT)
    sim = target.simulation
    snr_default = np.asarray(sim.get_snr(), dtype=float)
    wl_um = np.asarray(sim.get_lbda(), dtype=float) / ANGSTROM_PER_UM
    t_obs = float(sim.observing_time)

    snr_in = snr_default * np.sqrt(t["t14_s"] / t_obs)
    depth_err = np.sqrt(2.0) / snr_in

    tables.write_noise_table(
        TABLES, name, wl_um, depth_err,
        meta={"generator": "make_noise_tables.py", "created": str(date.today()),
              "instrument": INSTRUMENT, "norm_band": NORM_BAND,
              "J_mag": t["J_mag"], "t14_s": t["t14_s"],
              "t_obs_default_s": t_obs,
              "stellar_grid": star_meta.get("stellar_grid", "phoenix"),
              "assumption": "equal out-of-transit baseline; SNR ~ sqrt(t)"})
    print(f"noise: {name}  J={t['J_mag']:.2f}  t14={t['t14_s']/3600:.2f} h  "
          f"median err = {np.nanmedian(depth_err)*1e6:.0f} ppm")


def main():
    for name in PLANET_NAMES:
        run_planet(name)
    tables.rebuild_index(TABLES)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Trial run on one planet, then full run**

Trial: temporarily `main()` over `["WASP-39 b"]` via a quick edit or run the module in `python -c`. Then full run:

`~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py`

Expected: 14 lines; median single-transit errors span roughly 30 ppm (bright hosts like HD 209458) to a few×10³ ppm (faint TRAPPIST-1); NaN-free except possibly channels outside the throughput range — if `get_snr()` returns NaN/zero channels at the band edges, drop those channels (mask `snr > 0` and finite) before writing, and note the trimmed range in the meta.

If `LazuliTarget(...)` raises on the `2massj` band (sncosmo needs a one-time bandpass download — already cached during planning), or on the star spectrum input format, debug against `slicersim.lazuli.LazuliTarget.__init__` (`GIT/slicersim_repo2/slicersim/src/slicersim/lazuli.py:653`).

- [ ] **Step 3: Commit**

```bash
git add precompute/make_noise_tables.py data/tables
git commit -m "feat: slicersim single-transit noise tables for all targets"
```

---

### Task 11: End-to-end tests on the real tables + README

**Files:**
- Create: `tests/test_end_to_end.py`
- Create: `README.md`

**Interfaces:**
- Consumes: the full committed table set, the whole runtime API.

- [ ] **Step 1: Write the failing tests**

`tests/test_end_to_end.py`:

```python
"""End-to-end checks against the real committed tables in data/tables."""
import numpy as np
import pytest

import terminator

PLANETS = terminator.list_planets()


def test_fourteen_planets_all_scenarios():
    assert len(PLANETS) == 14
    for p in PLANETS:
        assert terminator.list_scenarios(p) == [
            "clear", "cloudy", "flat", "hazy", "high_metallicity"]


@pytest.mark.parametrize("planet", PLANETS)
def test_every_planet_observes(planet):
    for scen in terminator.list_scenarios(planet):
        obs = terminator.observe(planet, scenario=scen, n_transits=3, R=80)
        assert np.isfinite(obs.depth).all()
        assert np.isfinite(obs.depth_err).all()
        assert (obs.depth_err > 0).all()
        assert 0.0005 < np.median(obs.depth) < 0.1
        sig = obs.detect()
        assert np.isfinite(sig["atmosphere"])


def test_flat_scenario_detects_nothing():
    for planet in PLANETS:
        obs = terminator.observe(planet, scenario="flat", R=80)
        assert obs.detect()["atmosphere"] < 1e-6


def test_clear_beats_cloudy():
    # a grey deck mutes features -> lower detection significance
    for planet in ["WASP-39 b", "HD 209458 b"]:
        clear = terminator.observe(planet, scenario="clear", R=80).detect()
        cloudy = terminator.observe(planet, scenario="cloudy", R=80).detect()
        assert clear["atmosphere"] > cloudy["atmosphere"]


def test_bright_hot_jupiter_beats_faint_temperate():
    hj = terminator.observe("HD 209458 b", scenario="clear", R=80).detect()
    tp = terminator.observe("TRAPPIST-1 e", scenario="clear", R=80).detect()
    assert hj["atmosphere"] > tp["atmosphere"]


def test_sqrt_n_scaling_on_significance():
    s1 = terminator.observe("WASP-39 b", n_transits=1, R=80).detect()["atmosphere"]
    s4 = terminator.observe("WASP-39 b", n_transits=4, R=80).detect()["atmosphere"]
    assert s4 == pytest.approx(2 * s1, rel=1e-6)


def test_runtime_has_no_heavy_imports():
    import sys
    for mod in ["POSEIDON", "slicersim", "jax", "jaxoplanet", "scipy",
                "pandas", "astropy"]:
        assert not any(m == mod or m.startswith(mod + ".") for m in sys.modules), mod
```

- [ ] **Step 2: Run tests**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/ -v`
Expected: all PASS. Failures here are findings, not test bugs — e.g. if `clear_beats_cloudy` fails, inspect the actual spectra (quick matplotlib dump) before touching anything; a cloudy table with stronger features than clear means a stage-1 scenario bug.

- [ ] **Step 3: Write `README.md`**

Contents (concise, Gummi-style): what terminator is (one paragraph), install (`pip install -e ".[plot,test]"`), quickstart (the `observe`/`detect`/`plot` snippet from the spec), the two-stage precompute diagram (a short fenced text block: inputs → poseidon env → models/star tables; star tables + slicersim → noise tables; runtime = numpy + tables), how to regenerate tables (the three commands with their env vars), table schema note + `TERMINATOR_TABLES`, known approximations (isothermal eq-chem, band-mask molecule significances, √t SNR scaling, equal out-of-transit baseline), and a "Django/Heroku (planned)" section stating the runtime ships with tables and no heavy deps.

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py README.md
git commit -m "test: end-to-end suite on real tables; add README"
```

---

### Task 12: Verification notebook

**Files:**
- Create: `notebooks/verification.ipynb` (via the notebook-demo skill's builder-script workflow)

**Interfaces:**
- Consumes: the full public API on real tables.

- [ ] **Step 1: Invoke the notebook-demo skill** and build a notebook with these sections:

1. **Setup** — imports, `terminator.list_planets()` table.
2. **Gallery** — `obs.plot()` for WASP-39 b (clear, 1 transit) and K2-18 b (clear, 20 transits).
3. **Scenario comparison** — WASP-39 b: all four non-flat scenarios over-plotted (hi-res models, %).
4. **Detectability curves** — σ(atmosphere) and σ(H2O) vs n_transits ∈ {1,2,5,10,20,50} for WASP-39 b, K2-18 b, TRAPPIST-1 e; horizontal 5σ line.
5. **Sanity checks** (asserted in cells, not just plotted): WASP-39 b clear feature amplitude between 500 and 5000 ppm; HD 209458 b single-transit median error < 100 ppm; TRAPPIST-1 e single-transit median error > 500 ppm; `flat` detects at 0σ.

Plots follow the gks-plotting style.

- [ ] **Step 2: Execute the notebook end-to-end and verify every assert passes**

Run per the notebook-demo skill (execute in py313 with the builder script; confirm no cell errors).

- [ ] **Step 3: Commit**

```bash
git add notebooks/
git commit -m "docs: verification notebook with sanity checks"
```

---

## Self-Review Notes

- Spec coverage: layout (T1), targets (T2), binning/observe/detect/plot (T3–T6), scenarios+flat (T7), poseidon env (T8), model+star tables (T9), noise tables (T10), error handling & index versioning (T1, exercised T11), tests (throughout + T11), notebook (T12), README (T11). Django explicitly out of scope per spec.
- Known stitch point: the exact `make_atmosphere` call is finalized by Task 8's smoke test and copied into Task 9 — flagged inline in both tasks.
- Type consistency: table function names and `SimulatedSpectrum` fields are used identically across Tasks 1, 4, 5, 6, 9, 10, 11.
