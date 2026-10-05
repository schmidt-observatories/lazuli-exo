# Parametric Transit Model (terminator side) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give terminator a planet-independent parametric transmission model (ζ grid + analytic scale-height mapping), a PHOENIX SED mini-grid, a live slicersim noise function with Vega→AB handled, and a `simulate(params)` assembler, so any transiting planet can be simulated at runtime with numpy only.

**Architecture:** POSEIDON runs once offline on a (T, [M/H], C/O) grid for a fiducial planet and stores the photospheric log-pressure ζ(λ) = −ln P_τ=1(λ)/bar. At runtime `atmosphere.transmission_depth` shifts ζ onto the user's planet (scale-height and slant-column terms), applies cloud deck and haze analytically in pressure space, and maps to depth with R(λ) = R_p + H (ζ − ζ_ref). `star.phoenix_sed` interpolates a PHOENIX mini-grid; `slicer.depth_error` wraps slicersim (lazy import) with the desaturation loop and the Vega→AB conversion; `simulate` glues them into the existing `SimulatedSpectrum`.

**Tech Stack:** numpy at runtime. Precompute: POSEIDON (conda env `poseidon`, py3.11, numpy<2), astropy for PHOENIX FITS. Noise: slicersim 1.0.0 + sncosmo in conda env `py313`. Tests: pytest in `py313` (`~/anaconda3/envs/py313/bin/python -m pytest`).

**Spec:** `docs/superpowers/specs/2026-10-04-parametric-transit-simulator-design.md` (this plan covers the `terminator` half; the `lazuli-etc-web2` page gets its own plan once this half is validated, because the page's interfaces are the functions built here).

## Spec amendments made by this plan (recorded here, mirrored into the spec in Task 9)

1. **Haze parametrization** follows POSEIDON's MacMad17 form exactly instead of a (P_h, γ) pair: κ_haze = n · 10^{log_a} · 5.31e-31 m² · (λ / 0.35 µm)^γ. The runtime converts it to a pressure analytically: P_haze(λ) = k T_eq / (κ_haze/n · √(2π R_p H)). This removes the "fit P_h during validation" step; the `hazy` preset is just log_a = 2.0, γ = −6.0, which is what the committed POSEIDON tables used.
2. **ζ is put on the planet's absolute pressure scale** before clouds/haze are applied: ζ_planet = ζ_fid + ½ ln[(R_p H)/(R_p,fid H_fid)] − ln(T_eq / T_chem). This follows from τ_slant = (P/kT) σ √(2π R_p H) and is what makes a user's cloud-top pressure in bar mean the same thing it means in POSEIDON. R_p remains the band-median transit radius.
3. **PHOENIX grid floor is 2000 K**, not 2300 K: the local pysynphot grid has 2000 K files in all five metallicity folders, verified on 2026-10-04 (log g 3.5–5.0 present everywhere from 2000 to 8000 K).
4. The shared transit-duration formula lives in a new non-generated module `terminator/orbit.py` (not in the generated `targets.py`), so `freeze_targets.py` cannot overwrite it.

## Global Constraints

- `import terminator` must stay numpy-only; slicersim and sncosmo are imported lazily inside `slicer.depth_error`.
- Tables live under `data/tables/` and are found by `tables.resolve_tables_dir`; new files are `atmo/zeta_grid.npz` and `star/phoenix_grid.npz`; `index.json` gains `atmo` and `star_grid` keys; `SCHEMA_VERSION` bumps 1 → 2.
- ζ grid axes: T_K = 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2300, 2600, 3000; log_met = −1.0 … 3.0 step 0.5; c_to_o = 0.3, 0.55, 0.8, 1.0; wavelength constant R = 2000 over 0.35–1.85 µm; `zeta` and `mu` stored float32.
- Fiducial planet for the grid: R_p = 1 R_J, g = 10 m s⁻², R_s = 1 R_☉, P_ref = 10 bar, pressure grid logspace(+2, −7, 100) bar **descending**, species H2O, CO, CO2, CH4, NH3, Na, K on H2/He, fastchem equilibrium chemistry.
- PHOENIX grid axes: every T_eff file from 2000 to 8000 K, log g = 3.5, 4.0, 4.5, 5.0, [Fe/H] = −1.0, −0.5, 0.0, +0.3, +0.5 (folders m10, m05, m00, p03, p05); wavelength constant R = 1000 over 0.30–2.50 µm; `log_flam` float32.
- Magnitude default is Vega; conversion to AB per band uses sncosmo (`m_AB − m_Vega` of any spectrum through the same bandpass), never a hard-coded number. Bands: J → `2massj`, H → `2massh`, Ks → `2massks`, V → `bessellv`.
- Validation acceptance: feature amplitude (max − min at R = 100 over 0.40–1.70 µm) within 10 % for clear and high_metallicity, within 20 % for cloudy and hazy, after removing a constant offset.
- POSEIDON precompute runs must be memory-safe on a 24 GB laptop: opacities loaded per temperature node with a 10-point T_fine window (the footprint that already worked for the per-planet tables), partial results flushed to disk after every T node, RSS logged, resumable.
- All errors a user can trigger (ranges, units, missing tables) raise `ValueError`/`FileNotFoundError` with the valid range or fix in the message; nothing silently clamps without saying so in `info`.
- Commit after every task; do not commit the pre-existing uncommitted edits on `main` (README, webapp, tests from earlier sessions) unless a task says so.

## Review Focus

Inputs the spec implies but no test would otherwise exercise; each gets a test in the owning task:

1. **T_eq below 300 K with a real H** (K2-18 b-like): chemistry is clamped to 300 K but H must use the true T_eq and `info["T_clamped"]` must be True. → Task 3.
2. **Cloud deck deeper than the gas photosphere everywhere** (log_P_cloud = +1): the spectrum must equal the clear one, not go flat. → Task 3.
3. **Requested wavelength outside the grid** (0.3 µm or 2.0 µm): `transmission_depth` must raise rather than extrapolate. → Task 3.
4. **A star above the PHOENIX ceiling** (T_eff = 9000 K) raises; a star below (1800 K) falls back to a blackbody and says so; log g = 5.3 clamps to 5.0 and says so. → Task 5.
5. **A magnitude in AB from the user** must not be converted again (offset 0 reported). → Task 6.

---

### Task 1: `terminator.orbit.transit_duration_s` shared by freeze_targets

**Files:**
- Create: `src/terminator/orbit.py`
- Modify: `precompute/freeze_targets.py` (replace the inline eccentric-duration block with a call)
- Test: `tests/test_orbit.py`

**Interfaces:**
- Produces: `transit_duration_s(period_d, a_over_rs, rp_rs, inc_deg=90.0, ecc=0.0, w_deg=90.0) -> float` (seconds, Winn 2010 eq. 14/16 with the conjunction-distance factors). Task 7 and the web plan use it when the archive lacks `pl_trandur`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orbit.py
import math
import pytest

from terminator.orbit import transit_duration_s


def test_hd209458b_circular_matches_archive():
    # PSCompPars: P=3.52474859 d, a/Rs=8.81, Rp/Rs=0.1209, i=86.71 deg; t14=3.072 h
    t14 = transit_duration_s(3.52474859, 8.81, 0.1209, inc_deg=86.71)
    assert abs(t14 / 3600.0 - 3.072) < 0.05


def test_eccentric_factor_shrinks_duration_at_periastron_transit():
    circ = transit_duration_s(3.0, 10.0, 0.1, inc_deg=89.0)
    # omega=90 deg: transit at periastron -> faster -> shorter
    ecc = transit_duration_s(3.0, 10.0, 0.1, inc_deg=89.0, ecc=0.3, w_deg=90.0)
    assert ecc < circ


def test_grazing_returns_zero_not_nan():
    assert transit_duration_s(3.0, 10.0, 0.1, inc_deg=80.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_orbit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'terminator.orbit'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/terminator/orbit.py
"""Orbital geometry helpers shared by the precompute scripts and the runtime."""
import math


def transit_duration_s(period_d, a_over_rs, rp_rs, inc_deg=90.0, ecc=0.0,
                       w_deg=90.0):
    """Total transit duration T14 in seconds (Winn 2010, eqs. 14 and 16).

    Impact parameter and duration both carry the eccentric conjunction-
    distance / velocity factors. Returns 0.0 for a non-transiting geometry.
    """
    w = math.radians(w_deg)
    i = math.radians(inc_deg)
    fac = math.sqrt(1.0 - ecc ** 2) / (1.0 + ecc * math.sin(w))
    b = a_over_rs * math.cos(i) * (1.0 - ecc ** 2) / (1.0 + ecc * math.sin(w))
    arg = math.sqrt(max((1.0 + rp_rs) ** 2 - b ** 2, 0.0)) / (
        a_over_rs * math.sin(i))
    return period_d * 86400.0 / math.pi * math.asin(min(arg, 1.0)) * fac
```

In `precompute/freeze_targets.py`, replace the block

```python
    if t14_s is None and per and a_rs and rp_ratio:
        # eccentric duration (Winn 2010 eq. 14/16): ...
        w = math.radians(w_deg)
        ...
        t14_s = per * 86400.0 / math.pi * math.asin(min(arg, 1.0)) * fac
```

with

```python
    if t14_s is None and per and a_rs and rp_ratio:
        t14_s = transit_duration_s(per, a_rs, rp_ratio, inc_deg=inc,
                                   ecc=ecc, w_deg=w_deg)
```

and add near the top (after `from pathlib import Path`):

```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terminator.orbit import transit_duration_s  # noqa: E402
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_orbit.py tests/test_freeze_targets.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/terminator/orbit.py precompute/freeze_targets.py tests/test_orbit.py
git commit -m "refactor: shared transit_duration_s in terminator.orbit"
```

---

### Task 2: Grid tables in `terminator.tables` (schema v2)

**Files:**
- Modify: `src/terminator/tables.py`
- Modify: `data/tables/index.json` (regenerated by `rebuild_index`)
- Test: `tests/test_tables.py` (append)

**Interfaces:**
- Produces:
  - `write_zeta_grid(tables_dir, *, T_K, log_met, c_to_o, wavelength_um, zeta, mu, meta) -> Path`
  - `load_zeta_grid(tables_dir=None) -> dict` with keys `T_K, log_met, c_to_o, wavelength_um` (float64 1-D), `zeta` (float32, shape [T, met, co, wl]), `mu` (float32, [T, met, co]), `meta` (dict). Cached per path in a module dict.
  - `write_phoenix_grid(tables_dir, *, teff_K, logg, feh, wavelength_um, log_flam, meta) -> Path`
  - `load_phoenix_grid(tables_dir=None) -> dict` with keys `teff_K, logg, feh, wavelength_um`, `log_flam` (float32, [teff, logg, feh, wl]), `meta`.
  - `SCHEMA_VERSION = 2`; `rebuild_index` adds `"atmo": {"zeta_grid": bool}` and `"star_grid": {"phoenix_grid": bool}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tables.py`:

```python
import numpy as np
import pytest

from terminator import tables


def _tiny_zeta(tdir):
    T = np.array([300.0, 1000.0]); met = np.array([0.0, 1.0]); co = np.array([0.55])
    wl = np.geomspace(0.35, 1.85, 50)
    zeta = np.random.default_rng(0).normal(8.0, 1.0, (2, 2, 1, 50))
    mu = np.full((2, 2, 1), 2.3)
    return tables.write_zeta_grid(tdir, T_K=T, log_met=met, c_to_o=co,
                                  wavelength_um=wl, zeta=zeta, mu=mu,
                                  meta={"fiducial": {"g_ms2": 10.0}})


def test_zeta_grid_roundtrip_and_dtypes(tmp_path):
    _tiny_zeta(tmp_path)
    g = tables.load_zeta_grid(tmp_path)
    assert g["zeta"].dtype == np.float32 and g["zeta"].shape == (2, 2, 1, 50)
    assert g["mu"].shape == (2, 2, 1)
    assert g["meta"]["fiducial"]["g_ms2"] == 10.0
    assert tables.load_zeta_grid(tmp_path) is g        # cached


def test_phoenix_grid_roundtrip(tmp_path):
    teff = np.array([3000.0, 4000.0]); logg = np.array([4.5, 5.0]); feh = np.array([0.0])
    wl = np.geomspace(0.3, 2.5, 40)
    lf = np.zeros((2, 2, 1, 40)) - 10.0
    tables.write_phoenix_grid(tmp_path, teff_K=teff, logg=logg, feh=feh,
                              wavelength_um=wl, log_flam=lf, meta={"src": "x"})
    g = tables.load_phoenix_grid(tmp_path)
    assert g["log_flam"].dtype == np.float32
    np.testing.assert_allclose(g["teff_K"], teff)


def test_index_reports_grids(tmp_path):
    _tiny_zeta(tmp_path)
    idx = tables.rebuild_index(tmp_path)
    assert idx["version"] == 2
    assert idx["atmo"] == {"zeta_grid": True}
    assert idx["star_grid"] == {"phoenix_grid": False}


def test_missing_grid_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_zeta_grid.py"):
        tables.load_zeta_grid(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_tables.py -q`
Expected: FAIL with `AttributeError: module 'terminator.tables' has no attribute 'write_zeta_grid'`

- [ ] **Step 3: Implement**

In `src/terminator/tables.py`: set `SCHEMA_VERSION = 2`, extend the module docstring layout with

```
    atmo/zeta_grid.npz      (T_K, log_met, c_to_o, wavelength_um, zeta, mu, meta)
    star/phoenix_grid.npz   (teff_K, logg, feh, wavelength_um, log_flam, meta)
```

and add:

```python
ZETA_GRID = Path("atmo") / "zeta_grid.npz"
PHOENIX_GRID = Path("star") / "phoenix_grid.npz"
_GRID_CACHE = {}


def _write_grid(path, meta, float32_keys, **arrays):
    arrays = {k: (np.asarray(v, dtype=np.float32) if k in float32_keys
                  else np.asarray(v, dtype=float)) for k, v in arrays.items()}
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
    return _write_grid(Path(tables_dir) / ZETA_GRID, meta, {"zeta", "mu"},
                       T_K=T_K, log_met=log_met, c_to_o=c_to_o,
                       wavelength_um=wavelength_um, zeta=zeta, mu=mu)


def load_zeta_grid(tables_dir=None):
    return _load_grid(resolve_tables_dir(tables_dir) / ZETA_GRID,
                      "make_zeta_grid.py")


def write_phoenix_grid(tables_dir, *, teff_K, logg, feh, wavelength_um,
                       log_flam, meta):
    return _write_grid(Path(tables_dir) / PHOENIX_GRID, meta, {"log_flam"},
                       teff_K=teff_K, logg=logg, feh=feh,
                       wavelength_um=wavelength_um, log_flam=log_flam)


def load_phoenix_grid(tables_dir=None):
    return _load_grid(resolve_tables_dir(tables_dir) / PHOENIX_GRID,
                      "make_phoenix_grid.py")
```

In `rebuild_index`, before `index = {...}`:

```python
    index = {"version": SCHEMA_VERSION, "planets": planets,
             "atmo": {"zeta_grid": (tdir / ZETA_GRID).exists()},
             "star_grid": {"phoenix_grid": (tdir / PHOENIX_GRID).exists()}}
```

Note `_write` already calls `path.parent.mkdir(parents=True, exist_ok=True)`. `_load_grid` must not fail when `tables_dir` has no `index.json` (it does not call `load_index`).

Then regenerate the committed index so the version check passes:

```bash
~/anaconda3/envs/py313/bin/python -c "from terminator import tables; tables.rebuild_index('data/tables')"
```

- [ ] **Step 4: Run the whole suite**

Run: `~/anaconda3/envs/py313/bin/python -m pytest -q`
Expected: all PASS (the fixture in `tests/conftest.py` calls `rebuild_index`, so it picks up v2 automatically).

- [ ] **Step 5: Commit**

```bash
git add src/terminator/tables.py data/tables/index.json tests/test_tables.py
git commit -m "feat(tables): zeta and PHOENIX grid files, schema v2"
```

---

### Task 3: `terminator.atmosphere.transmission_depth`

**Files:**
- Create: `src/terminator/atmosphere.py`
- Test: `tests/test_atmosphere.py`
- Modify: `tests/conftest.py` (add `zeta_fixture` with analytic content)

**Interfaces:**
- Consumes: `tables.load_zeta_grid`.
- Produces:

```python
transmission_depth(wl_um, *, T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun,
                   log_met=0.0, c_to_o=0.55, mu=None,
                   log_P_cloud_bar=None, log_a_haze=None, haze_slope=-4.0,
                   tables_dir=None) -> (depth: ndarray, info: dict)
```

`info` keys: `g_ms2, H_m, H_over_Rp, mu_used, mu_source ("chemistry"|"user"), T_chem_K, T_clamped (bool), zeta_ref, zeta_shift, feature_amplitude_ppm, band_um`. Also exposes `GRID_WAVELENGTHS = lambda tables_dir=None: load_zeta_grid(tables_dir)["wavelength_um"]` via `grid_wavelengths(tables_dir=None)`.

- [ ] **Step 1: Add the analytic fixture to `tests/conftest.py`**

```python
# analytic zeta grid: zeta = ZETA0 + ZETA_AMP * sin(2 pi wl / 0.5um) + 0.1*log_met,
# independent of T and C/O; mu = 2.3 + 0.5*log_met
ZETA0, ZETA_AMP = 9.0, 1.0
FID = {"R_p_m": 7.1492e7, "g_ms2": 10.0, "R_s_m": 6.957e8, "P_ref_bar": 10.0}


@pytest.fixture
def zeta_fixture(tmp_path):
    tdir = tmp_path / "tables"
    T = np.array([300.0, 600.0, 1000.0, 2000.0])
    met = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    co = np.array([0.3, 0.55, 1.0])
    wl = np.geomspace(0.35, 1.85, 600)
    base = ZETA0 + ZETA_AMP * np.sin(2 * np.pi * wl / 0.5)
    zeta = base[None, None, None, :] + 0.1 * met[None, :, None, None] \
        + 0.0 * T[:, None, None, None] + 0.0 * co[None, None, :, None]
    mu = 2.3 + 0.5 * met[None, :, None] + 0.0 * T[:, None, None] + 0.0 * co[None, None, :]
    tables.write_zeta_grid(tdir, T_K=T, log_met=met, c_to_o=co,
                           wavelength_um=wl, zeta=zeta, mu=mu,
                           meta={"fiducial": FID})
    return tdir
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_atmosphere.py
import numpy as np
import pytest

from conftest import ZETA0, ZETA_AMP
from terminator import atmosphere as atm

HJ = dict(T_eq_K=1000.0, R_p_RJ=1.0, M_p_MJ=0.6, R_s_RSun=1.0)
K_B, M_H, G, R_J, M_J, R_SUN = 1.380649e-23, 1.6735575e-27, 6.674e-11, 7.1492e7, 1.898e27, 6.957e8


def radius_m(depth, R_s_RSun):
    return np.sqrt(depth) * R_s_RSun * R_SUN


def test_node_recovery_shape_and_median_radius(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    depth, info = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    g = G * 0.6 * M_J / R_J ** 2
    H = K_B * 1000.0 / (2.3 * M_H * g)
    np.testing.assert_allclose(info["g_ms2"], g)
    np.testing.assert_allclose(info["H_m"], H)
    band = (wl >= 0.40) & (wl <= 1.70)
    # R_p is the band-median transit radius
    np.testing.assert_allclose(np.median(radius_m(depth[band], 1.0)), R_J, rtol=1e-9)
    # amplitude in radius = H * (max - min of the analytic zeta) = 2 H
    r = radius_m(depth, 1.0)
    np.testing.assert_allclose(r.max() - r.min(), 2 * ZETA_AMP * H, rtol=1e-6)


def test_doubling_teq_doubles_amplitude(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    d1, _ = atm.transmission_depth(wl, **{**HJ, "T_eq_K": 600.0}, tables_dir=zeta_fixture)
    d2, _ = atm.transmission_depth(wl, **{**HJ, "T_eq_K": 1200.0}, tables_dir=zeta_fixture)
    a1 = np.ptp(radius_m(d1, 1.0)); a2 = np.ptp(radius_m(d2, 1.0))
    np.testing.assert_allclose(a2 / a1, 2.0, rtol=1e-6)


def test_mu_default_and_override(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    _, i0 = atm.transmission_depth(wl, **HJ, log_met=1.0, tables_dir=zeta_fixture)
    assert i0["mu_source"] == "chemistry" and abs(i0["mu_used"] - 2.8) < 1e-6
    _, i1 = atm.transmission_depth(wl, **HJ, log_met=1.0, mu=18.0, tables_dir=zeta_fixture)
    assert i1["mu_source"] == "user" and i1["mu_used"] == 18.0
    assert i1["H_m"] < i0["H_m"]


def test_cloud_deck_monotonic_and_deep_deck_is_clear(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    clear, _ = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    amps = []
    for logp in (1.0, -3.0, -4.0, -5.0, -8.0):
        d, _ = atm.transmission_depth(wl, **HJ, log_P_cloud_bar=logp, tables_dir=zeta_fixture)
        amps.append(np.ptp(d))
    np.testing.assert_allclose(amps[0], np.ptp(clear))     # deck below photosphere -> clear
    assert all(a >= b for a, b in zip(amps, amps[1:]))      # higher deck -> flatter
    assert amps[-1] < 1e-12                                 # deck above everything -> flat


def test_haze_steepens_blue_end(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    clear, _ = atm.transmission_depth(wl, **HJ, tables_dir=zeta_fixture)
    hazy, _ = atm.transmission_depth(wl, **HJ, log_a_haze=3.0, haze_slope=-4.0, tables_dir=zeta_fixture)
    blue = wl < 0.5; red = wl > 1.5
    assert (hazy[blue].mean() - hazy[red].mean()) > (clear[blue].mean() - clear[red].mean())


def test_cold_planet_clamps_chemistry_but_not_H(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    _, info = atm.transmission_depth(wl, T_eq_K=250.0, R_p_RJ=0.2, M_p_MJ=0.03, R_s_RSun=0.4,
                                     tables_dir=zeta_fixture)
    assert info["T_clamped"] is True and info["T_chem_K"] == 300.0
    g = G * 0.03 * M_J / (0.2 * R_J) ** 2
    np.testing.assert_allclose(info["H_m"], K_B * 250.0 / (2.3 * M_H * g))


def test_ranges_raise(zeta_fixture):
    wl = atm.grid_wavelengths(zeta_fixture)
    with pytest.raises(ValueError, match=r"log_met"):
        atm.transmission_depth(wl, **HJ, log_met=4.0, tables_dir=zeta_fixture)
    with pytest.raises(ValueError, match=r"c_to_o"):
        atm.transmission_depth(wl, **HJ, c_to_o=2.0, tables_dir=zeta_fixture)
    with pytest.raises(ValueError, match=r"wavelength"):
        atm.transmission_depth(np.array([0.30, 1.0]), **HJ, tables_dir=zeta_fixture)
    with pytest.raises(ValueError, match=r"T_eq_K"):
        atm.transmission_depth(wl, **{**HJ, "T_eq_K": 0.0}, tables_dir=zeta_fixture)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_atmosphere.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'terminator.atmosphere'`

- [ ] **Step 4: Implement**

```python
# src/terminator/atmosphere.py
"""Parametric isothermal transmission spectrum from the shared zeta grid.

zeta(lambda) = -ln P_photo(lambda)/bar is tabulated for a fiducial planet on a
(T, [M/H], C/O) grid (precompute/make_zeta_grid.py). For any other planet:

    zeta_p = zeta_fid + 0.5 ln[(R_p H)/(R_p,fid H_fid)] - ln(T_eq/T_chem)
    zeta   = max(logaddexp(zeta_p, zeta_haze), -ln P_cloud)
    R(lam) = R_p + H (zeta - median_band(zeta)),   depth = (R / R_s)^2

so R_p is the band-median transit radius. Fidelity: isothermal, equilibrium
chemistry, gray deck, power-law haze; no pressure-broadening correction.
"""
import numpy as np

from . import tables

K_B = 1.380649e-23
M_H = 1.6735575e-27
G = 6.674e-11
R_J = 7.1492e7
M_J = 1.898e27
R_SUN = 6.957e8
SIGMA_H2_RAYLEIGH_350NM = 5.31e-31   # m^2, POSEIDON MacMad17 haze anchor
HAZE_LAMBDA0_UM = 0.35
BAND_UM = (0.40, 1.70)


def grid_wavelengths(tables_dir=None):
    return tables.load_zeta_grid(tables_dir)["wavelength_um"]


def _interp_weights(axis, x):
    """Index and fractional weight for linear interpolation along `axis`."""
    axis = np.asarray(axis, float)
    if len(axis) == 1:
        return 0, 0.0
    i = int(np.clip(np.searchsorted(axis, x, side="right") - 1, 0, len(axis) - 2))
    t = (x - axis[i]) / (axis[i + 1] - axis[i])
    return i, float(t)


def _trilinear(values, axes, point):
    """Trilinear interpolation of values[a, b, c, ...] at point=(x, y, z)."""
    out = 0.0
    for (i, t), ax in zip([_interp_weights(a, p) for a, p in zip(axes, point)], range(3)):
        pass  # (weights computed below; kept explicit for readability)
    (i0, t0), (i1, t1), (i2, t2) = [_interp_weights(a, p) for a, p in zip(axes, point)]
    for di, w0 in ((0, 1 - t0), (1, t0)):
        for dj, w1 in ((0, 1 - t1), (1, t1)):
            for dk, w2 in ((0, 1 - t2), (1, t2)):
                w = w0 * w1 * w2
                if w:
                    out = out + w * values[min(i0 + di, values.shape[0] - 1),
                                           min(i1 + dj, values.shape[1] - 1),
                                           min(i2 + dk, values.shape[2] - 1)]
    return out


def _check_range(name, x, axis):
    lo, hi = float(np.min(axis)), float(np.max(axis))
    if not (lo <= x <= hi):
        raise ValueError(f"{name}={x} outside the grid range [{lo}, {hi}]")


def transmission_depth(wl_um, *, T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun,
                       log_met=0.0, c_to_o=0.55, mu=None,
                       log_P_cloud_bar=None, log_a_haze=None, haze_slope=-4.0,
                       tables_dir=None):
    g = tables.load_zeta_grid(tables_dir)
    wl_grid = g["wavelength_um"]
    wl_um = np.atleast_1d(np.asarray(wl_um, float))
    for name, x in (("T_eq_K", T_eq_K), ("R_p_RJ", R_p_RJ),
                    ("M_p_MJ", M_p_MJ), ("R_s_RSun", R_s_RSun)):
        if not (np.isfinite(x) and x > 0):
            raise ValueError(f"{name} must be > 0, got {x}")
    if mu is not None and not (np.isfinite(mu) and mu > 0):
        raise ValueError(f"mu must be > 0, got {mu}")
    if wl_um.min() < wl_grid.min() or wl_um.max() > wl_grid.max():
        raise ValueError(
            f"wavelength range [{wl_um.min():.3f}, {wl_um.max():.3f}] um outside "
            f"the grid [{wl_grid.min():.3f}, {wl_grid.max():.3f}] um")
    _check_range("log_met", log_met, g["log_met"])
    _check_range("c_to_o", c_to_o, g["c_to_o"])

    T_chem = float(np.clip(T_eq_K, g["T_K"].min(), g["T_K"].max()))
    T_clamped = T_chem != float(T_eq_K)
    axes = (g["T_K"], g["log_met"], g["c_to_o"])
    point = (T_chem, log_met, c_to_o)
    zeta_fid = _trilinear(g["zeta"].astype(float), axes, point)
    mu_chem = float(_trilinear(g["mu"].astype(float), axes, point))
    mu_used = mu_chem if mu is None else float(mu)

    fid = g["meta"]["fiducial"]
    R_p = R_p_RJ * R_J
    R_s = R_s_RSun * R_SUN
    g_p = G * M_p_MJ * M_J / R_p ** 2
    H = K_B * T_eq_K / (mu_used * M_H * g_p)
    H_fid = K_B * T_chem / (mu_chem * M_H * fid["g_ms2"])
    shift = 0.5 * np.log((R_p * H) / (fid["R_p_m"] * H_fid)) - np.log(T_eq_K / T_chem)
    zeta = zeta_fid + shift

    if log_a_haze is not None:
        kappa = 10.0 ** log_a_haze * SIGMA_H2_RAYLEIGH_350NM * (
            wl_grid / HAZE_LAMBDA0_UM) ** haze_slope          # m^2 per molecule
        P_haze_bar = K_B * T_eq_K / (kappa * np.sqrt(2 * np.pi * R_p * H)) / 1e5
        zeta = np.logaddexp(zeta, -np.log(P_haze_bar))
    if log_P_cloud_bar is not None:
        zeta = np.maximum(zeta, -log_P_cloud_bar * np.log(10.0))

    band = (wl_grid >= BAND_UM[0]) & (wl_grid <= BAND_UM[1])
    zeta_ref = float(np.median(zeta[band]))
    radius = R_p + H * (zeta - zeta_ref)
    depth_grid = (radius / R_s) ** 2
    depth = np.interp(wl_um, wl_grid, depth_grid)

    info = {"g_ms2": float(g_p), "H_m": float(H), "H_over_Rp": float(H / R_p),
            "mu_used": mu_used, "mu_source": "chemistry" if mu is None else "user",
            "T_chem_K": T_chem, "T_clamped": bool(T_clamped),
            "zeta_ref": zeta_ref, "zeta_shift": float(shift),
            "feature_amplitude_ppm": float(np.ptp(depth_grid[band]) * 1e6),
            "band_um": list(BAND_UM)}
    return depth, info
```

(Remove the dead `for ... pass` loop in `_trilinear` before committing; it is shown only so the two-step weight computation is obvious. The final function is the second form.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_atmosphere.py -q`
Expected: 7 PASS. If `test_node_recovery` fails on the median check, the cause is an even number of band samples (median averages two values); the fixture uses 600 geomspace points, and the band has an odd count — verify with `band.sum()` and adjust the fixture to 601 points if needed.

- [ ] **Step 6: Commit**

```bash
git add src/terminator/atmosphere.py tests/test_atmosphere.py tests/conftest.py
git commit -m "feat: parametric transmission_depth from the zeta grid"
```

---

### Task 4: `precompute/make_zeta_grid.py` (POSEIDON, memory-safe, resumable) and run it

**Files:**
- Create: `precompute/make_zeta_grid.py`
- Modify: `precompute/scenarios.py` (add the grid axes so the validation script and the generator share them)
- Modify: `.gitignore` (add `precompute/*.log`, `data/tables/atmo/*.partial.npz`)
- Output: `data/tables/atmo/zeta_grid.npz` (committed, ≈ 8 MB)

**Interfaces:**
- Consumes: `tables.write_zeta_grid`, `scenarios.make_wl_grid`, POSEIDON (`define_model`, `read_opacities`, `make_atmosphere`, `compute_spectrum`, `create_planet`, `create_star`, `load_chemistry_grid`).
- Produces: the grid file with `meta = {"fiducial": {"R_p_m", "g_ms2", "R_s_m", "P_ref_bar", "M_p_kg"}, "P_grid_log10bar": [hi, lo, n], "species": [...], "poseidon_version", "created", "R_model", "mu_layers_bar": [1e-4, 1e-1], "out_of_column": {...}}`.

- [ ] **Step 1: Add axes to `precompute/scenarios.py`**

```python
# Parametric zeta grid (precompute/make_zeta_grid.py). Shared with the
# validation script so both read one definition.
ZETA_T_K = [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800,
            2000, 2300, 2600, 3000]
ZETA_LOG_MET = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ZETA_C_TO_O = [0.3, 0.55, 0.8, 1.0]
ZETA_R_STORE = 2000          # constant-R wavelength grid of the stored zeta
FIDUCIAL = dict(R_p_RJ=1.0, g_ms2=10.0, R_s_RSun=1.0)
```

- [ ] **Step 2: Write the generator**

```python
# precompute/make_zeta_grid.py
"""Build data/tables/atmo/zeta_grid.npz: photospheric log-pressure zeta(lambda)
for a fiducial planet on a (T, [M/H], C/O) grid.

Run (poseidon env, ~1 h; resumable, memory-safe):
  POSEIDON_input_data=$HOME/inputs/ PYSYN_CDBS=$HOME/inputs/stellar_grids \
  nice ~/anaconda3/envs/poseidon/bin/python precompute/make_zeta_grid.py \
      [--smoke] [--resume]

Memory: opacities are loaded per temperature node with a 10-point T_fine
window (the footprint that already worked for the per-planet tables, <2 GB
RSS) and freed before the next node; results are flushed to a .partial.npz
after every node so a crash costs at most one node.
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

from scenarios import (FIDUCIAL, P_GRID_ARGS, P_REF_BAR, PARAM_SPECIES,
                       ZETA_C_TO_O, ZETA_LOG_MET, ZETA_R_STORE, ZETA_T_K,
                       WL_MAX_UM, WL_MIN_UM, make_wl_grid)
from terminator import binning, tables

import POSEIDON
from POSEIDON.chemistry import load_chemistry_grid
from POSEIDON.constants import R_J, R_Sun
from POSEIDON.core import (compute_spectrum, create_planet, create_star,
                           define_model, make_atmosphere, read_opacities)

TABLES = ROOT / "data" / "tables"
PARTIAL = TABLES / "atmo" / "zeta_grid.partial.npz"
LOG = Path(__file__).with_suffix(".log")
G_SI = 6.674e-11
MU_LAYERS_BAR = (1e-4, 1e-1)


def log(msg):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    line = f"[{time.strftime('%H:%M:%S')} rss={rss_mb:.0f}MB] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def zeta_from_spectrum(spectrum, atmosphere, R_s_m):
    """Map depth(lambda) -> R(lambda) -> P through the atmosphere's own r(P)."""
    P = np.asarray(atmosphere["P"], float)                    # descending
    r = np.asarray(atmosphere["r"], float).reshape(len(P), -1)[:, 0]
    R_lam = np.sqrt(np.asarray(spectrum, float)) * R_s_m
    order = np.argsort(r)                                     # r ascending
    lnP = np.interp(R_lam, r[order], np.log(P[order]))
    below = R_lam < r.min()
    above = R_lam > r.max()
    return -lnP, int(below.sum()), int(above.sum())


def mu_photosphere(atmosphere):
    P = np.asarray(atmosphere["P"], float)
    mu = np.asarray(atmosphere["mu"], float).reshape(len(P), -1)[:, 0]
    m = (P >= MU_LAYERS_BAR[0]) & (P <= MU_LAYERS_BAR[1])
    return float(np.median(mu[m]) / 1.66053907e-27)   # POSEIDON mu is in kg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 T x 2 met x 1 C/O at R=1000; writes zeta_grid.smoke.npz")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    T_K = ZETA_T_K[:2] if args.smoke else ZETA_T_K
    log_met = ZETA_LOG_MET[2:4] if args.smoke else ZETA_LOG_MET
    c_to_o = ZETA_C_TO_O[1:2] if args.smoke else ZETA_C_TO_O
    if args.smoke:
        import scenarios
        scenarios.R_MODEL = 1000
    WL = make_wl_grid()
    edges = binning.constant_R_edges(WL_MIN_UM, WL_MAX_UM, ZETA_R_STORE)
    wl_store = binning.bin_centers(edges)

    R_p = FIDUCIAL["R_p_RJ"] * R_J
    R_s = FIDUCIAL["R_s_RSun"] * R_Sun
    M_p = FIDUCIAL["g_ms2"] * R_p ** 2 / G_SI
    planet = create_planet("zeta_fiducial", R_p, mass=M_p)
    star = create_star(R_s, 5500.0, 4.5, 0.0, stellar_grid="blackbody", wl=WL)
    model = define_model("zeta_grid", bulk_species=["H2", "He"],
                         param_species=PARAM_SPECIES, PT_profile="isotherm",
                         X_profile="chem_eq", cloud_model="cloud-free")
    chem = load_chemistry_grid(PARAM_SPECIES, grid="fastchem")
    lo, hi, n = P_GRID_ARGS
    P = np.logspace(hi, lo, n)
    log_P_fine = np.linspace(-6.0, 2.0, 40)

    shape = (len(T_K), len(log_met), len(c_to_o), len(wl_store))
    zeta = np.full(shape, np.nan, np.float32)
    mu = np.full(shape[:3], np.nan, np.float32)
    done = np.zeros(len(T_K), bool)
    out_of_column = {}
    if args.resume and PARTIAL.exists():
        with np.load(PARTIAL) as f:
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
                atmo = make_atmosphere(planet, model, P, P_REF_BAR, R_p,
                                       PT_params=[float(T)],
                                       log_X_params=[float(co), float(met)],
                                       cloud_params=[], chemistry_grid=chem)
                spec = compute_spectrum(planet, star, model, atmo, opac, WL,
                                        spectrum_type="transmission")
                assert np.isfinite(spec).all(), (T, met, co)
                z, n_below, n_above = zeta_from_spectrum(spec, atmo, R_s)
                zeta[it, im, ic] = binning.bin_model(WL, z, edges)
                mu[it, im, ic] = mu_photosphere(atmo)
                if n_below or n_above:
                    out_of_column[f"{T}_{met}_{co}"] = [n_below, n_above]
        done[it] = True
        del opac
        gc.collect()
        PARTIAL.parent.mkdir(parents=True, exist_ok=True)
        np.savez(PARTIAL, zeta=zeta, mu=mu, done=done,
                 out_of_column=json.dumps(out_of_column))
        elapsed = (time.time() - t_start) / 60
        log(f"T={T} K done ({done.sum()}/{len(T_K)}), {elapsed:.1f} min elapsed, "
            f"zeta range {np.nanmin(zeta[it]):.2f}..{np.nanmax(zeta[it]):.2f}, "
            f"mu {np.nanmin(mu[it]):.2f}..{np.nanmax(mu[it]):.2f}")

    meta = {"generator": "make_zeta_grid.py", "created": str(date.today()),
            "fiducial": {"R_p_m": float(R_p), "M_p_kg": float(M_p),
                         "g_ms2": FIDUCIAL["g_ms2"], "R_s_m": float(R_s),
                         "P_ref_bar": P_REF_BAR},
            "P_grid_log10bar": list(P_GRID_ARGS), "species": PARAM_SPECIES,
            "chemistry": "fastchem", "PT_profile": "isotherm",
            "poseidon_version": getattr(POSEIDON, "__version__", "unknown"),
            "R_model": int(round(1.0 / np.log(WL[1] / WL[0]))),
            "R_store": ZETA_R_STORE, "mu_layers_bar": list(MU_LAYERS_BAR),
            "zeta_definition": "-ln(P_tau=1 / 1 bar) at slant optical depth "
                               "unity, via depth -> R -> r(P) column",
            "out_of_column": out_of_column, "smoke": bool(args.smoke)}
    target = TABLES / "atmo" / ("zeta_grid.smoke.npz" if args.smoke else "zeta_grid.npz")
    tables.write_zeta_grid(target.parent, T_K=T_K, log_met=log_met, c_to_o=c_to_o,
                           wavelength_um=wl_store, zeta=zeta, mu=mu, meta=meta)
    if args.smoke:
        (target.parent / "zeta_grid.npz").rename(target)
    else:
        PARTIAL.unlink(missing_ok=True)
        tables.rebuild_index(TABLES)
    log(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

Verify POSEIDON's `mu` units before trusting `mu_photosphere`: in `POSEIDON/core.py` `make_atmosphere`, `mu` is computed by `POSEIDON.atmosphere.profiles` in **kg** (it divides `k T / (mu g)` to get H in metres). Run once in the smoke step and check that the returned value is ≈ 2.3 for solar metallicity; if it is ≈ 3.8e-27 the division is missing, if it is ≈ 1.4e27 it is doubled.

- [ ] **Step 3: Smoke run (foreground, ~3–5 min) and inspect**

```bash
cd ~/Dropbox/mypylib/notebooks/GIT/terminator
POSEIDON_input_data=$HOME/inputs/ PYSYN_CDBS=$HOME/inputs/stellar_grids \
  ~/anaconda3/envs/poseidon/bin/python precompute/make_zeta_grid.py --smoke
~/anaconda3/envs/py313/bin/python - <<'EOF'
import numpy as np, json
f = np.load("data/tables/atmo/zeta_grid.smoke.npz")
z = f["zeta"]; print(z.shape, np.nanmin(z), np.nanmax(z), "mu", f["mu"].ravel())
print(json.loads(str(f["meta"]))["out_of_column"])
EOF
```

Expected: zeta between about 4 and 16 (P_photo between 1e-2 and 1e-7 bar), mu ≈ 2.3 at [M/H]=0 and ≈ 2.4–2.6 at +0.5, empty or tiny `out_of_column`, peak RSS under 3 GB in the log. Then delete the smoke file:

```bash
rm data/tables/atmo/zeta_grid.smoke.npz
```

- [ ] **Step 4: Full run in the background, monitored**

```bash
cd ~/Dropbox/mypylib/notebooks/GIT/terminator
POSEIDON_input_data=$HOME/inputs/ PYSYN_CDBS=$HOME/inputs/stellar_grids \
  nohup nice -n 10 ~/anaconda3/envs/poseidon/bin/python precompute/make_zeta_grid.py --resume \
  > precompute/make_zeta_grid.stdout 2>&1 &
echo $! > precompute/make_zeta_grid.pid
```

Monitor with `tail -3 precompute/make_zeta_grid.log` and `ps -o rss= -p $(cat precompute/make_zeta_grid.pid)`. If RSS exceeds 8 GB, kill it (`kill $(cat precompute/make_zeta_grid.pid)`), halve `T_fine` to 6 points and `log_P_fine` to 25, and relaunch with `--resume`. Continue with Tasks 5–7 while it runs; Task 8 needs the finished file.

- [ ] **Step 5: Commit the script now; commit the grid when the run finishes**

```bash
printf 'precompute/*.log\nprecompute/*.stdout\nprecompute/*.pid\ndata/tables/atmo/*.partial.npz\ndata/tables/atmo/*.smoke.npz\n' >> .gitignore
git add precompute/make_zeta_grid.py precompute/scenarios.py .gitignore
git commit -m "feat(precompute): zeta grid generator (memory-safe, resumable)"
# later, after the run:
git add data/tables/atmo/zeta_grid.npz data/tables/index.json
git commit -m "data: zeta grid 16x9x4, R=2000, POSEIDON fiducial 1 RJ / 10 m s-2"
```

---

### Task 5: `terminator.star.phoenix_sed` and `precompute/make_phoenix_grid.py`

**Files:**
- Create: `src/terminator/star.py`
- Create: `precompute/make_phoenix_grid.py`
- Test: `tests/test_star.py`
- Modify: `tests/conftest.py` (add `phoenix_fixture`)
- Output: `data/tables/star/phoenix_grid.npz` (committed, ≈ 8 MB)

**Interfaces:**
- Consumes: `tables.load_phoenix_grid`, `binning.constant_R_edges/bin_centers/bin_model`.
- Produces: `phoenix_sed(teff_K, logg, feh, tables_dir=None) -> (wl_angstrom, f_lambda, info)`; `info = {"fallback": None|"blackbody", "clamped": {"logg": (5.3, 5.0)} or {}, "teff_K", "logg_used", "feh_used", "grid": "phoenix"}`. `f_lambda` in erg s⁻¹ cm⁻² Å⁻¹ (surface), absolute scale irrelevant downstream.

- [ ] **Step 1: Fixture in `tests/conftest.py`**

```python
@pytest.fixture
def phoenix_fixture(tmp_path):
    """Analytic 'PHOENIX' grid: log_flam = log10(teff) + 0.1*logg + 0.01*feh - 2*log10(wl)."""
    tdir = tmp_path / "tables"
    teff = np.array([2000.0, 3000.0, 4000.0, 6000.0, 8000.0])
    logg = np.array([3.5, 4.0, 4.5, 5.0])
    feh = np.array([-1.0, -0.5, 0.0, 0.3, 0.5])
    wl = np.geomspace(0.3, 2.5, 300)
    lf = (np.log10(teff)[:, None, None, None] + 0.1 * logg[None, :, None, None]
          + 0.01 * feh[None, None, :, None] - 2 * np.log10(wl)[None, None, None, :])
    tables.write_phoenix_grid(tdir, teff_K=teff, logg=logg, feh=feh,
                              wavelength_um=wl, log_flam=lf, meta={"src": "test"})
    return tdir
```

- [ ] **Step 2: Failing tests**

```python
# tests/test_star.py
import numpy as np
import pytest

from terminator import star


def test_node_recovery(phoenix_fixture):
    wl_A, f, info = star.phoenix_sed(4000.0, 4.5, 0.0, tables_dir=phoenix_fixture)
    wl_um = wl_A / 1e4
    expect = 10 ** (np.log10(4000.0) + 0.45 - 2 * np.log10(wl_um))
    np.testing.assert_allclose(f, expect, rtol=2e-6)
    assert info["fallback"] is None and info["clamped"] == {}


def test_interpolates_in_log_teff(phoenix_fixture):
    _, f_lo, _ = star.phoenix_sed(3000.0, 4.5, 0.0, tables_dir=phoenix_fixture)
    _, f_mid, _ = star.phoenix_sed(np.sqrt(3000.0 * 4000.0), 4.5, 0.0, tables_dir=phoenix_fixture)
    _, f_hi, _ = star.phoenix_sed(4000.0, 4.5, 0.0, tables_dir=phoenix_fixture)
    # log-space midpoint -> geometric mean of the fluxes
    np.testing.assert_allclose(f_mid, np.sqrt(f_lo * f_hi), rtol=1e-6)


def test_clamping_is_reported(phoenix_fixture):
    _, f, info = star.phoenix_sed(3500.0, 5.3, 0.7, tables_dir=phoenix_fixture)
    assert info["clamped"] == {"logg": (5.3, 5.0), "feh": (0.7, 0.5)}
    _, f_edge, _ = star.phoenix_sed(3500.0, 5.0, 0.5, tables_dir=phoenix_fixture)
    np.testing.assert_allclose(f, f_edge)


def test_blackbody_fallback_and_ceiling(phoenix_fixture):
    wl_A, f, info = star.phoenix_sed(1800.0, 5.0, 0.0, tables_dir=phoenix_fixture)
    assert info["fallback"] == "blackbody" and np.all(f > 0)
    # Wien peak of 1800 K is 1.6 um: flux rises from 0.5 to 1.5 um
    assert f[np.argmin(abs(wl_A - 15000))] > f[np.argmin(abs(wl_A - 5000))]
    with pytest.raises(ValueError, match="8000"):
        star.phoenix_sed(9000.0, 4.0, 0.0, tables_dir=phoenix_fixture)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_star.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'terminator.star'`

- [ ] **Step 4: Implement `src/terminator/star.py`**

```python
"""Stellar SED from the PHOENIX mini-grid (precompute/make_phoenix_grid.py)."""
import numpy as np

from . import tables
from .atmosphere import _trilinear, _interp_weights  # noqa: F401 (shared helper)

H_PLANCK, C_LIGHT, K_B = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def _blackbody_flam(wl_um, T):
    """pi * B_lambda in erg s^-1 cm^-2 A^-1 (surface flux)."""
    lam = wl_um * 1e-6
    B = 2 * H_PLANCK * C_LIGHT ** 2 / lam ** 5 / np.expm1(H_PLANCK * C_LIGHT / (lam * K_B * T))
    return np.pi * B * 1e-7            # W m^-2 m^-1 -> erg s^-1 cm^-2 A^-1


def phoenix_sed(teff_K, logg, feh, tables_dir=None):
    g = tables.load_phoenix_grid(tables_dir)
    wl_um = g["wavelength_um"]
    teff_axis = g["teff_K"]
    info = {"fallback": None, "clamped": {}, "teff_K": float(teff_K),
            "logg_used": float(logg), "feh_used": float(feh), "grid": "phoenix"}
    if teff_K > teff_axis.max():
        raise ValueError(
            f"T_eff={teff_K} K above the PHOENIX grid ceiling {teff_axis.max():.0f} K")
    if teff_K < teff_axis.min():
        info["fallback"] = "blackbody"
        info["grid"] = "blackbody"
        return wl_um * 1e4, _blackbody_flam(wl_um, float(teff_K)), info
    for name, x, axis in (("logg", logg, g["logg"]), ("feh", feh, g["feh"])):
        xc = float(np.clip(x, axis.min(), axis.max()))
        if xc != float(x):
            info["clamped"][name] = (float(x), xc)
            info[f"{name}_used"] = xc
    axes = (np.log10(teff_axis), g["logg"], g["feh"])
    point = (np.log10(teff_K), info["logg_used"], info["feh_used"])
    log_flam = _trilinear(g["log_flam"].astype(float), axes, point)
    return wl_um * 1e4, 10.0 ** log_flam, info
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_star.py -q`
Expected: 4 PASS

- [ ] **Step 6: Write `precompute/make_phoenix_grid.py` and run it (poseidon env, astropy, ~2 min)**

```python
"""Build data/tables/star/phoenix_grid.npz from the local pysynphot PHOENIX grid.

Run: PYSYN_CDBS=$HOME/inputs/stellar_grids \
     ~/anaconda3/envs/poseidon/bin/python precompute/make_phoenix_grid.py
"""
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terminator import binning, tables  # noqa: E402

CDBS = Path(os.environ.get("PYSYN_CDBS", Path.home() / "inputs" / "stellar_grids"))
GRID = CDBS / "grid" / "phoenix"
TABLES = ROOT / "data" / "tables"
FEH = {-1.0: "phoenixm10", -0.5: "phoenixm05", 0.0: "phoenixm00",
       0.3: "phoenixp03", 0.5: "phoenixp05"}
LOGG = [3.5, 4.0, 4.5, 5.0]
TEFF_MIN, TEFF_MAX = 2000, 8000
WL_MIN_UM, WL_MAX_UM, R_STORE = 0.30, 2.50, 1000
FLOOR = 1e-30    # erg s-1 cm-2 A-1; keeps log10 finite in deep absorption


def teff_of(path):
    return int(re.findall(r"_(\d+)\.fits$", path.name)[0])


def main():
    edges = binning.constant_R_edges(WL_MIN_UM, WL_MAX_UM, R_STORE)
    wl_store = binning.bin_centers(edges)
    files0 = sorted(GRID.joinpath(FEH[0.0]).glob("*.fits"), key=teff_of)
    teffs = [teff_of(p) for p in files0 if TEFF_MIN <= teff_of(p) <= TEFF_MAX]
    fehs = sorted(FEH)
    log_flam = np.full((len(teffs), len(LOGG), len(fehs), len(wl_store)), np.nan)
    for k, feh in enumerate(fehs):
        for i, teff in enumerate(teffs):
            path = GRID / FEH[feh] / f"{FEH[feh]}_{teff}.fits"
            with fits.open(path) as h:
                d = h[1].data
                wl_um = np.asarray(d["WAVELENGTH"], float) / 1e4
                sel = (wl_um >= WL_MIN_UM * 0.98) & (wl_um <= WL_MAX_UM * 1.02)
                for j, g in enumerate(LOGG):
                    col = f"g{int(round(g * 10)):02d}"
                    if col not in d.columns.names:
                        raise SystemExit(f"{path.name}: no column {col}")
                    flux = np.asarray(d[col], float)[sel]
                    if not np.any(flux > 0):
                        raise SystemExit(f"{path.name} {col}: all-zero model")
                    binned = binning.bin_model(wl_um[sel], flux, edges)
                    log_flam[i, j, k] = np.log10(np.maximum(binned, FLOOR))
        print(f"[Fe/H]={feh:+.1f}: {len(teffs)} T_eff files done")
    assert np.isfinite(log_flam).all()
    tables.write_phoenix_grid(
        TABLES, teff_K=teffs, logg=LOGG, feh=fehs, wavelength_um=wl_store,
        log_flam=log_flam,
        meta={"generator": "make_phoenix_grid.py", "created": str(date.today()),
              "source": "pysynphot PHOENIX (STScI CDBS grid/phoenix)",
              "units": "log10 F_lambda [erg s-1 cm-2 A-1] at the stellar surface, "
                       "bin-averaged at constant R",
              "R_store": R_STORE, "folders": FEH})
    tables.rebuild_index(TABLES)
    print("wrote", TABLES / tables.PHOENIX_GRID)


if __name__ == "__main__":
    main()
```

Run it, then check size and a sanity curve:

```bash
PYSYN_CDBS=$HOME/inputs/stellar_grids ~/anaconda3/envs/poseidon/bin/python precompute/make_phoenix_grid.py
ls -la data/tables/star/phoenix_grid.npz
~/anaconda3/envs/py313/bin/python -c "
from terminator import star; import numpy as np
wl,f,i = star.phoenix_sed(5778,4.44,0.0); print(i, wl[np.argmax(f)])  # peak near 4500-5000 A
wl,f,i = star.phoenix_sed(2566,5.2,0.05); print(i)                    # TRAPPIST-1: grid, logg clamped
"
```

Expected: file ≈ 8 MB, Sun-like peak 4000–6000 Å, TRAPPIST-1 served from the grid with `clamped == {"logg": (5.2, 5.0)}`.

- [ ] **Step 7: Full suite and commit**

Run: `~/anaconda3/envs/py313/bin/python -m pytest -q`

```bash
git add src/terminator/star.py precompute/make_phoenix_grid.py tests/test_star.py tests/conftest.py data/tables/star/phoenix_grid.npz data/tables/index.json
git commit -m "feat: PHOENIX SED mini-grid and star.phoenix_sed"
```

---

### Task 6: `terminator.slicer.depth_error` and rewrite `make_noise_tables.py` over it

**Files:**
- Create: `src/terminator/slicer.py`
- Modify: `precompute/make_noise_tables.py` (delete `desaturate`, call `depth_error`)
- Test: `tests/test_slicer.py`

**Interfaces:**
- Produces:

```python
BANDS = {"J": "2massj", "H": "2massh", "Ks": "2massks", "V": "bessellv"}
vega_to_ab_offset(band) -> float                      # m_AB - m_Vega for BANDS[band]
depth_error(wl_A, f_lambda, *, mag, band="J", mag_system="vega", t14_s,
            n_transits=1, instrument="lazuli_cbe") -> (wl_um, depth_err, info)
```

`info` keys: `slicersim_version, instrument, band, sncosmo_band, mag_input, mag_system, vega_ab_offset, mag_ab, t14_s, t_obs_s, nmd_default, nmd_used, nsaturated_detpx, fully_desaturated, trimmed_channels, assumption`.

- [ ] **Step 1: Failing tests (skipped when slicersim is absent)**

```python
# tests/test_slicer.py
import numpy as np
import pytest

slicersim = pytest.importorskip("slicersim")
sncosmo = pytest.importorskip("sncosmo")

from terminator import slicer, star  # noqa: E402


def test_vega_ab_offsets_are_sane():
    assert 0.85 < slicer.vega_to_ab_offset("J") < 0.95
    assert abs(slicer.vega_to_ab_offset("V")) < 0.05
    assert slicer.vega_to_ab_offset("Ks") > slicer.vega_to_ab_offset("H") > slicer.vega_to_ab_offset("J")


def test_unknown_band_raises():
    with pytest.raises(ValueError, match="band"):
        slicer.vega_to_ab_offset("Z")


@pytest.fixture(scope="module")
def sun():
    wl_A, f, _ = star.phoenix_sed(5778.0, 4.44, 0.0)
    return wl_A, f


def test_ab_input_is_not_converted(sun):
    wl, e_vega, i_v = slicer.depth_error(*sun, mag=10.0, band="J", mag_system="vega", t14_s=7200.0)
    wl2, e_ab, i_a = slicer.depth_error(*sun, mag=10.0, band="J", mag_system="ab", t14_s=7200.0)
    assert i_a["vega_ab_offset"] == 0.0 and i_a["mag_ab"] == 10.0
    assert abs(i_v["mag_ab"] - (10.0 + i_v["vega_ab_offset"])) < 1e-9
    # Vega J=10 is a brighter AB source than AB J=10 -> smaller errors
    assert np.nanmedian(e_vega) > np.nanmedian(e_ab) * 1.0  # Vega J=10 -> AB 10.9 -> fainter -> larger err
    assert i_v["fully_desaturated"] in (True, False) and i_v["slicersim_version"]


def test_bright_star_hits_the_floor_and_reports_it(sun):
    _, _, info = slicer.depth_error(*sun, mag=6.0, band="J", t14_s=7200.0)
    assert tuple(info["nmd_used"][:2]) == (2, 1) or info["nsaturated_detpx"] == 0
    assert info["nsaturated_detpx"] >= 0


def test_n_transits_scaling(sun):
    _, e1, _ = slicer.depth_error(*sun, mag=10.0, t14_s=7200.0, n_transits=1)
    _, e4, _ = slicer.depth_error(*sun, mag=10.0, t14_s=7200.0, n_transits=4)
    np.testing.assert_allclose(e1 / e4, 2.0)
```

Fix the sign confusion in `test_ab_input_is_not_converted` before running: Vega J = 10 corresponds to AB J ≈ 10.9, which is *fainter* than AB J = 10, so `e_vega` must be **larger** than `e_ab`. Keep only `assert np.nanmedian(e_vega) > np.nanmedian(e_ab)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_slicer.py -q`
Expected: FAIL with `ImportError: cannot import name 'slicer'`

- [ ] **Step 3: Implement `src/terminator/slicer.py`**

```python
"""Live slicersim noise: single-transit depth uncertainty per IFS channel.

slicersim and sncosmo are imported inside the functions so that
`import terminator` stays numpy-only. slicersim normalizes magnitudes as AB
(scene/pointsource.py hard-codes bandmag(band, "ab")); archive magnitudes are
Vega, so the default path converts with sncosmo's own band definitions.
"""
import warnings

import numpy as np

BANDS = {"J": "2massj", "H": "2massh", "Ks": "2massks", "V": "bessellv"}
ANGSTROM_PER_UM = 1.0e4


def _sncosmo_band(band):
    try:
        return BANDS[band]
    except KeyError:
        raise ValueError(f"band must be one of {sorted(BANDS)}, got {band!r}") from None


def vega_to_ab_offset(band):
    """m_AB - m_Vega for `band`, from sncosmo (independent of the spectrum)."""
    import sncosmo
    b = _sncosmo_band(band)
    bp = sncosmo.get_bandpass(b)
    wave = np.linspace(bp.minwave() * 0.9, bp.maxwave() * 1.1, 2000)
    s = sncosmo.Spectrum(wave, np.full_like(wave, 1e-15))
    return float(s.bandmag(b, "ab") - s.bandmag(b, "vega"))


def _desaturate(target):
    """Shrink the MACC ramp (groups, then frames/group) until the point
    source stops saturating, or the 2x1 floor is reached. Moved verbatim
    from precompute/make_noise_tables.py; see its docstring for why."""
    sim = target.simulation
    n, m, d = sim.detector.nmd
    min_group = sim.detector.min_group
    snr = np.asarray(sim.get_snr(), dtype=float)
    nsat = sim.detector.nsaturated_detpx or 0
    while nsat > 0 and n > min_group:
        n = max(min_group, n // 2)
        target.change_detector(nmd=(n, m, d))
        snr = np.asarray(sim.get_snr(), dtype=float)
        nsat = sim.detector.nsaturated_detpx or 0
    while nsat > 0 and m > 1:
        m = max(1, m // 2)
        target.change_detector(nmd=(n, m, d))
        snr = np.asarray(sim.get_snr(), dtype=float)
        nsat = sim.detector.nsaturated_detpx or 0
    return snr, (int(n), int(m), int(d)), int(nsat)


def depth_error(wl_A, f_lambda, *, mag, band="J", mag_system="vega", t14_s,
                n_transits=1, instrument="lazuli_cbe"):
    """Per-channel 1-sigma transit-depth uncertainty for `n_transits` transits.

    depth_err = sqrt(2) / (SNR * sqrt(t14 / t_obs)) / sqrt(n_transits): equal
    in- and out-of-transit baselines, photon + read noise only.
    """
    import slicersim
    from slicersim.lazuli import LazuliTarget

    if n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    if t14_s <= 0:
        raise ValueError("t14_s must be > 0")
    mag_system = mag_system.lower()
    if mag_system not in ("vega", "ab"):
        raise ValueError(f"mag_system must be 'vega' or 'ab', got {mag_system!r}")
    sn_band = _sncosmo_band(band)
    offset = vega_to_ab_offset(band) if mag_system == "vega" else 0.0
    mag_ab = float(mag) + offset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        target = LazuliTarget(np.asarray(wl_A, float), np.asarray(f_lambda, float),
                              mag=mag_ab, band=sn_band, instrument=instrument)
        sim = target.simulation
        nmd_default = tuple(int(x) for x in sim.detector.nmd)
        snr, nmd_used, nsat = _desaturate(target)
        wl_um = np.asarray(sim.get_lbda(), float) / ANGSTROM_PER_UM
        t_obs = float(sim.observing_time)

    snr_in = snr * np.sqrt(t14_s / t_obs)
    err = np.sqrt(2.0) / snr_in / np.sqrt(n_transits)
    good = np.isfinite(err) & (snr_in > 0)
    info = {"slicersim_version": getattr(slicersim, "__version__", "unknown"),
            "instrument": instrument, "band": band, "sncosmo_band": sn_band,
            "mag_input": float(mag), "mag_system": mag_system,
            "vega_ab_offset": offset, "mag_ab": mag_ab,
            "t14_s": float(t14_s), "t_obs_s": t_obs,
            "nmd_default": list(nmd_default), "nmd_used": list(nmd_used),
            "nsaturated_detpx": nsat, "fully_desaturated": nsat == 0,
            "trimmed_channels": int((~good).sum()),
            "assumption": "equal out-of-transit baseline; SNR ~ sqrt(t); "
                          "readout tuned to remove point-source saturation"}
    return wl_um[good], err[good], info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_slicer.py -q`
Expected: 5 PASS (≈ 10 s; the first slicersim call is slow).

- [ ] **Step 5: Rewrite `precompute/make_noise_tables.py`**

Replace everything from `INSTRUMENT = "lazuli_cbe"` through the end of `run_planet` with:

```python
from terminator.slicer import depth_error

INSTRUMENT = "lazuli_cbe"
BAND = "J"            # 2MASS J; archive J magnitudes are Vega


def run_planet(name):
    t = get_target(name)
    wl_star_um, f_star, star_meta = tables.load_star(TABLES, name)
    wl_um, err, info = depth_error(wl_star_um * 1.0e4, f_star, mag=t["J_mag"],
                                   band=BAND, mag_system="vega",
                                   t14_s=t["t14_s"], instrument=INSTRUMENT)
    tables.write_noise_table(
        TABLES, name, wl_um, err,
        meta={"generator": "make_noise_tables.py", "created": str(date.today()),
              "J_mag": t["J_mag"], "norm_band": info["sncosmo_band"],
              "stellar_grid": star_meta.get("stellar_grid", "phoenix"),
              **info})
    sat = "" if info["fully_desaturated"] else f"  RESIDUAL SATURATION nsat={info['nsaturated_detpx']}"
    print(f"noise: {name}  J={t['J_mag']:.2f} (AB {info['mag_ab']:.2f})  "
          f"t14={t['t14_s']/3600:.2f} h  nmd={info['nmd_used']}  "
          f"median err = {np.nanmedian(err)*1e6:.0f} ppm"
          + (f"  (trimmed {info['trimmed_channels']} ch)" if info["trimmed_channels"] else "")
          + sat)
```

Keep the module docstring (trim the paragraph describing `desaturate`, pointing at `terminator.slicer` instead), keep `main()`. Remove the now-unused `from slicersim.lazuli import LazuliTarget` and `NORM_BAND`/`ANGSTROM_PER_UM`.

- [ ] **Step 6: Regenerate the noise tables (py313, ~2 min) and inspect the change**

```bash
~/anaconda3/envs/py313/bin/python precompute/make_noise_tables.py 2>&1 | tee precompute/make_noise_tables.stdout
git diff --stat data/tables/noise
~/anaconda3/envs/py313/bin/python -m pytest -q
```

Expected: every planet prints `(AB J+0.90)`, median errors 1.4–1.5× the old values (compare against `git stash`-free numbers in `2026-09-26_audit_report.md` table), `nsaturated_detpx` for HD 189733 b / HD 209458 b / HAT-P-11 b smaller than before (fainter in AB) but possibly still non-zero. Tests: `test_end_to_end.py` may assert specific S/N numbers; if so, update the expected values to the regenerated ones and say so in the commit body.

- [ ] **Step 7: Commit**

```bash
git add src/terminator/slicer.py tests/test_slicer.py precompute/make_noise_tables.py data/tables/noise tests/test_end_to_end.py
git commit -m "feat: slicer.depth_error (Vega->AB via sncosmo); regenerate noise tables

Closes audit #1: slicersim normalizes AB, the archive J is Vega. Tables are
now 1.4-1.5x more conservative and carry slicersim version + nmd provenance."
```

---

### Task 7: `terminator.presets.showcase` and `terminator.simulate`

**Files:**
- Create: `src/terminator/presets.py`
- Create: `src/terminator/simulate.py`
- Modify: `src/terminator/__init__.py`
- Test: `tests/test_presets.py`, `tests/test_simulate.py`

**Interfaces:**
- Consumes: `targets.get_target`, `atmosphere.transmission_depth`, `atmosphere.grid_wavelengths`, `star.phoenix_sed`, `slicer.depth_error`, `binning.*`, `observe.SimulatedSpectrum`.
- Produces:

```python
presets.SCENARIO_PARAMS = {
    "clear":            dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=None, log_a_haze=None, haze_slope=-4.0),
    "cloudy":           dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=-3.0, log_a_haze=None, haze_slope=-4.0),
    "high_metallicity": dict(log_met=2.0, c_to_o=0.55, log_P_cloud_bar=None, log_a_haze=None, haze_slope=-4.0),
    "hazy":             dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=None, log_a_haze=2.0, haze_slope=-6.0),
    "flat":             dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=-6.0, log_a_haze=None, haze_slope=-4.0),
}
presets.showcase(planet, scenario="clear") -> dict   # full simulate() params
simulate.simulate(params, R=100, n_transits=1, tables_dir=None, instrument="lazuli_cbe") -> SimulatedSpectrum
```

`params` keys: `name, scenario, T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun, teff_K, logg, feh, mag, band, mag_system, t14_s, log_met, c_to_o, mu, log_P_cloud_bar, log_a_haze, haze_slope`. Missing optional keys default (`mu=None`, atmosphere keys as in `clear`, `band="J"`, `mag_system="vega"`, `scenario="custom"`, `name="custom"`). Missing required keys raise `ValueError` naming them.

- [ ] **Step 1: Failing tests**

```python
# tests/test_presets.py
import pytest

from terminator import presets


def test_showcase_maps_registry_and_scenario():
    p = presets.showcase("WASP-39 b", "hazy")
    assert p["name"] == "WASP-39 b" and p["scenario"] == "hazy"
    assert p["R_p_RJ"] == 1.279 and p["teff_K"] == 5485.0 and p["mag"] == 10.663
    assert p["band"] == "J" and p["mag_system"] == "vega"
    assert p["log_a_haze"] == 2.0 and p["haze_slope"] == -6.0 and p["log_P_cloud_bar"] is None


def test_flat_is_a_high_deck():
    assert presets.showcase("GJ 1214 b", "flat")["log_P_cloud_bar"] == -6.0


def test_unknown_raise():
    with pytest.raises(KeyError):
        presets.showcase("Nope b", "clear")
    with pytest.raises(KeyError, match="scenario"):
        presets.showcase("GJ 1214 b", "foggy")
```

```python
# tests/test_simulate.py
import numpy as np
import pytest

import terminator
from terminator import simulate as sim_mod


def fake_depth_error(wl_A, f_lambda, *, mag, band, mag_system, t14_s, n_transits=1, instrument):
    wl_um = np.geomspace(0.4, 1.7, 500)
    return wl_um, np.full(500, 1e-4) / np.sqrt(n_transits), {
        "mag_ab": mag + 0.9, "vega_ab_offset": 0.9, "nsaturated_detpx": 0,
        "fully_desaturated": True, "t_obs_s": 60.0, "nmd_used": [16, 8, 0],
        "slicersim_version": "fake", "instrument": instrument, "trimmed_channels": 0}


@pytest.fixture
def both_grids(zeta_fixture, phoenix_fixture):
    # the two fixtures write into the same tmp_path/tables
    return zeta_fixture


PARAMS = dict(name="Testar-1 b", T_eq_K=1000.0, R_p_RJ=1.0, M_p_MJ=0.6, R_s_RSun=1.0,
              teff_K=5000.0, logg=4.5, feh=0.0, mag=10.0, t14_s=7200.0)


def test_simulate_returns_simulated_spectrum(both_grids, monkeypatch):
    monkeypatch.setattr(sim_mod, "depth_error", fake_depth_error)
    obs = terminator.simulate(PARAMS, R=50, n_transits=4, tables_dir=both_grids)
    assert isinstance(obs, terminator.SimulatedSpectrum)
    assert obs.planet == "Testar-1 b" and obs.scenario == "custom"
    assert obs.wavelength.shape == obs.depth.shape == obs.depth_err.shape
    np.testing.assert_allclose(obs.flat_depth, (7.1492e7 / 6.957e8) ** 2)
    assert obs.meta["noise"]["mag_ab"] == 10.9 and obs.meta["atmosphere"]["mu_source"] == "chemistry"
    assert obs.detect()["template_snr_diagonal"] > 0
    # n_transits reaches the noise function (not applied twice)
    obs1 = terminator.simulate(PARAMS, R=50, n_transits=1, tables_dir=both_grids)
    np.testing.assert_allclose(obs1.depth_err / obs.depth_err, 2.0)


def test_missing_required_key_raises(both_grids, monkeypatch):
    monkeypatch.setattr(sim_mod, "depth_error", fake_depth_error)
    with pytest.raises(ValueError, match="t14_s"):
        terminator.simulate({k: v for k, v in PARAMS.items() if k != "t14_s"}, tables_dir=both_grids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/anaconda3/envs/py313/bin/python -m pytest tests/test_presets.py tests/test_simulate.py -q`
Expected: FAIL with `ImportError: cannot import name 'presets'`

- [ ] **Step 3: Implement**

```python
# src/terminator/presets.py
"""Showcase planets x scenarios expressed as simulate() parameters."""
from .targets import PLANET_NAMES, get_target  # noqa: F401

SCENARIO_PARAMS = {
    "clear": dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=None, log_a_haze=None, haze_slope=-4.0),
    "cloudy": dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=-3.0, log_a_haze=None, haze_slope=-4.0),
    "high_metallicity": dict(log_met=2.0, c_to_o=0.55, log_P_cloud_bar=None, log_a_haze=None, haze_slope=-4.0),
    "hazy": dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=None, log_a_haze=2.0, haze_slope=-6.0),
    "flat": dict(log_met=0.0, c_to_o=0.55, log_P_cloud_bar=-6.0, log_a_haze=None, haze_slope=-4.0),
}
SCENARIOS = sorted(SCENARIO_PARAMS)


def showcase(planet, scenario="clear"):
    t = get_target(planet)
    if scenario not in SCENARIO_PARAMS:
        raise KeyError(f"Unknown scenario {scenario!r}. Available: {SCENARIOS}")
    return dict(name=t["name"], scenario=scenario,
                T_eq_K=t["T_eq_K"], R_p_RJ=t["R_p_RJ"], M_p_MJ=t["M_p_MJ"],
                R_s_RSun=t["R_s_RSun"], teff_K=t["T_eff_K"], logg=t["log_g_star"],
                feh=t["Met_star"], mag=t["J_mag"], band="J", mag_system="vega",
                t14_s=t["t14_s"], mu=None, **SCENARIO_PARAMS[scenario])
```

```python
# src/terminator/simulate.py
"""Assemble a parametric simulation into the existing SimulatedSpectrum."""
import numpy as np

from . import binning
from .atmosphere import R_J, R_SUN, grid_wavelengths, transmission_depth
from .observe import SimulatedSpectrum
from .slicer import depth_error
from .star import phoenix_sed

REQUIRED = ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "teff_K", "logg", "feh",
            "mag", "t14_s")
ATMO_DEFAULTS = dict(log_met=0.0, c_to_o=0.55, mu=None, log_P_cloud_bar=None,
                     log_a_haze=None, haze_slope=-4.0)


def simulate(params, R=100, n_transits=1, tables_dir=None, instrument="lazuli_cbe"):
    missing = [k for k in REQUIRED if params.get(k) is None]
    if missing:
        raise ValueError(f"simulate() params missing {missing}")
    if n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    p = {**ATMO_DEFAULTS, **params}

    wl_model = grid_wavelengths(tables_dir)
    depth_model, info_a = transmission_depth(
        wl_model, T_eq_K=p["T_eq_K"], R_p_RJ=p["R_p_RJ"], M_p_MJ=p["M_p_MJ"],
        R_s_RSun=p["R_s_RSun"], log_met=p["log_met"], c_to_o=p["c_to_o"],
        mu=p["mu"], log_P_cloud_bar=p["log_P_cloud_bar"],
        log_a_haze=p["log_a_haze"], haze_slope=p["haze_slope"],
        tables_dir=tables_dir)
    wl_A, flam, info_s = phoenix_sed(p["teff_K"], p["logg"], p["feh"], tables_dir=tables_dir)
    wl_n, err, info_n = depth_error(
        wl_A, flam, mag=p["mag"], band=p.get("band", "J"),
        mag_system=p.get("mag_system", "vega"), t14_s=p["t14_s"],
        n_transits=n_transits, instrument=instrument)

    edges = binning.constant_R_edges(wl_n.min(), wl_n.max(), R)
    depth = binning.bin_model(wl_model, depth_model, edges)
    sigma = binning.bin_sigma(wl_n, err, edges)
    keep = (wl_model >= edges[0]) & (wl_model <= edges[-1])
    return SimulatedSpectrum(
        planet=str(p.get("name", "custom")), scenario=str(p.get("scenario", "custom")),
        n_transits=int(n_transits), R=int(R),
        wavelength=binning.bin_centers(edges), depth=depth, depth_err=sigma,
        model_wavelength=wl_model[keep], model_depth=depth_model[keep],
        flat_depth=float((p["R_p_RJ"] * R_J / (p["R_s_RSun"] * R_SUN)) ** 2),
        meta={"atmosphere": info_a, "star": info_s, "noise": info_n,
              "params": dict(p), "tables_dir": str(tables_dir) if tables_dir else None})
```

In `src/terminator/__init__.py`:

```python
"""terminator: Lazuli IFS exoplanet transmission spectroscopy simulator."""
from . import atmosphere, binning, lightcurve, presets, slicer, star, tables
from .observe import SimulatedSpectrum, observe
from .simulate import simulate
from .tables import list_planets, list_scenarios

__all__ = ["observe", "simulate", "SimulatedSpectrum", "list_planets",
           "list_scenarios", "tables", "binning", "lightcurve", "atmosphere",
           "star", "slicer", "presets"]
```

Confirm `import terminator` is still numpy-only: `slicer` imports sncosmo/slicersim lazily, so this holds; verify with the test below.

- [ ] **Step 4: Add an import-purity test to `tests/test_simulate.py`**

```python
def test_import_terminator_is_numpy_only():
    import subprocess, sys
    code = ("import sys, terminator; "
            "bad=[m for m in ('slicersim','sncosmo','matplotlib','flask') if m in sys.modules]; "
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
```

- [ ] **Step 5: Run and commit**

Run: `~/anaconda3/envs/py313/bin/python -m pytest -q`
Expected: all PASS

```bash
git add src/terminator/presets.py src/terminator/simulate.py src/terminator/__init__.py tests/test_presets.py tests/test_simulate.py
git commit -m "feat: simulate(params) and showcase presets over the parametric model"
```

---

### Task 8: Validation against the committed POSEIDON tables (needs the Task 4 grid)

**Files:**
- Create: `precompute/validate_zeta_grid.py`
- Create: `notebooks/06_parametric_model.ipynb` (built with the `notebook-demo` skill from a builder script `notebooks/build_06.py`, which is not committed)
- Test: `tests/test_validation_report.py` (reads the JSON the script writes; skipped if absent)
- Output: `notebooks/figs/06_validation.json`, `notebooks/figs/06_*.png`

**Interfaces:**
- Consumes: `tables.load_model`, `presets.showcase`, `atmosphere.transmission_depth`, `binning.*`.
- Produces: `validate(tables_dir=None) -> list[dict]` with per (planet, scenario): `max_abs_dev_ppm, amp_table_ppm, amp_model_ppm, amp_ratio, offset_ppm, pass`.

- [ ] **Step 1: Write the script**

```python
# precompute/validate_zeta_grid.py
"""Compare the parametric model against the 14 x 4 committed POSEIDON tables.

Run: ~/anaconda3/envs/py313/bin/python precompute/validate_zeta_grid.py
Writes notebooks/figs/06_validation.json and prints a markdown table.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terminator import binning, presets, tables  # noqa: E402
from terminator.atmosphere import transmission_depth  # noqa: E402

TOL = {"clear": 0.10, "high_metallicity": 0.10, "cloudy": 0.20, "hazy": 0.20}
R_CMP, BAND = 100, (0.40, 1.70)
OUT = ROOT / "notebooks" / "figs" / "06_validation.json"


def validate(tables_dir=None):
    tdir = tables.resolve_tables_dir(tables_dir)
    edges = binning.constant_R_edges(*BAND, R_CMP)
    rows = []
    for planet in presets.PLANET_NAMES:
        for scen, tol in TOL.items():
            wl, d_tab, _ = tables.load_model(tdir, planet, scen)
            p = presets.showcase(planet, scen)
            d_mod, info = transmission_depth(
                wl, **{k: p[k] for k in ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun",
                                         "log_met", "c_to_o", "mu", "log_P_cloud_bar",
                                         "log_a_haze", "haze_slope")}, tables_dir=tdir)
            bt = binning.bin_model(wl, d_tab, edges)
            bm = binning.bin_model(wl, d_mod, edges)
            offset = float(np.mean(bm - bt))
            dev = bm - bt - offset
            amp_t, amp_m = float(np.ptp(bt)), float(np.ptp(bm))
            ratio = amp_m / amp_t if amp_t > 0 else np.nan
            rows.append(dict(planet=planet, scenario=scen,
                             max_abs_dev_ppm=float(np.max(np.abs(dev)) * 1e6),
                             amp_table_ppm=amp_t * 1e6, amp_model_ppm=amp_m * 1e6,
                             amp_ratio=float(ratio), offset_ppm=offset * 1e6,
                             tol=tol, T_clamped=info["T_clamped"],
                             passed=bool(abs(ratio - 1) <= tol)))
    return rows


def main():
    rows = validate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print("| planet | scenario | amp table [ppm] | amp model [ppm] | ratio | max dev [ppm] | pass |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['planet']} | {r['scenario']} | {r['amp_table_ppm']:.0f} | "
              f"{r['amp_model_ppm']:.0f} | {r['amp_ratio']:.2f} | "
              f"{r['max_abs_dev_ppm']:.0f} | {'yes' if r['passed'] else 'NO'} |")
    n_fail = sum(not r["passed"] for r in rows)
    print(f"\n{len(rows) - n_fail}/{len(rows)} within tolerance")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it once the grid exists**

Run: `~/anaconda3/envs/py313/bin/python precompute/validate_zeta_grid.py`

Read the table. Expected physics: clear/high_metallicity ratios 0.9–1.1; cloudy within 0.8–1.2; hazy within 0.8–1.2; `T_clamped` planets (K2-18 b, LHS 1140 b, TRAPPIST-1 e) may sit at the edge because POSEIDON's own tables were computed with T_iso clamped to 300 K for **both** chemistry and H, whereas the parametric path uses the true T_eq for H. For those three, also report the ratio with `T_eq_K=300.0` passed in (append a `note` column) so the comparison is like-for-like; the plan accepts the clamped-T comparison for them.

If a planet fails outside those three, the first suspects, in order: (1) the ζ shift sign (Task 3 `shift`), check by printing `info["zeta_shift"]` and confirming a hot Jupiter with H ≈ 500 km gives a shift near +0.5 ln(H/H_fid); (2) pressure broadening at high metallicity (ratios systematically < 1 at log_met = 2); (3) haze column mismatch if only `hazy` fails, which points at the √(2π R_p H) term. Fix the cause in `atmosphere.py` with a test that reproduces the discrepancy; do not loosen the tolerance.

- [ ] **Step 3: Test that reads the report**

```python
# tests/test_validation_report.py
import json
from pathlib import Path

import pytest

REPORT = Path(__file__).resolve().parents[1] / "notebooks" / "figs" / "06_validation.json"


@pytest.mark.skipif(not REPORT.exists(), reason="run precompute/validate_zeta_grid.py")
def test_validation_report_passes():
    rows = json.loads(REPORT.read_text())
    assert len(rows) == 14 * 4
    failed = [(r["planet"], r["scenario"], round(r["amp_ratio"], 2))
              for r in rows if not r["passed"] and not r["T_clamped"]]
    assert failed == []
```

- [ ] **Step 4: Notebook 06 via the `notebook-demo` skill**

Builder script content (cells): (1) title + the ζ equation from the spec; (2) load grid, plot ζ(λ) at T = 600/1200/2000 K for solar metallicity and at log_met = 0/1/2 at 1000 K; (3) μ_chem vs log_met; (4) `showcase("WASP-39 b", s)` for the four scenarios: table vs parametric overlay at R = 100 (offset removed); (5) the validation table from `06_validation.json` rendered with pandas; (6) a custom planet: TOI-270 d with μ = 5, cloud at 10 mbar, 20 transits, through `terminator.simulate` with `detect()` printed. Save figures to `notebooks/figs/06_*.png` using `plt.style.use('gks')` per the `gks-plotting` skill. Execute end to end; the skill verifies outputs.

- [ ] **Step 5: Commit**

```bash
git add precompute/validate_zeta_grid.py tests/test_validation_report.py notebooks/06_parametric_model.ipynb notebooks/figs/06_*
git commit -m "test: validate the parametric model against the POSEIDON tables; notebook 06"
```

---

### Task 9: Docs, spec amendments, README

**Files:**
- Modify: `README.md` (new "Parametric model" section with the `simulate` example, the Vega/AB note, and the regenerated noise numbers caveat), `docs/sphinx/` API page if one lists modules
- Modify: `docs/superpowers/specs/2026-10-04-parametric-transit-simulator-design.md` (fold in the four amendments at the top of this plan under a "Amendments 2026-10-04" heading; change haze knob in the form section to log_a + γ; PHOENIX floor 2000 K)
- Modify: memory note `audit-2026-09-26-vega-ab-and-gh-issues.md` (audit #1 closed on the tables; P_ref defined away for the parametric path)

- [ ] **Step 1: README section**

```markdown
## Parametric model (any planet)

`terminator.simulate(params)` builds a spectrum for any transiting planet from a
shared POSEIDON-derived opacity grid and a PHOENIX stellar grid, with slicersim
computing the noise live (needs `slicersim` and `sncosmo`; `import terminator`
itself stays numpy-only):

    import terminator
    p = terminator.presets.showcase("WASP-39 b", "cloudy")   # or build the dict by hand
    p.update(mu=None, log_P_cloud_bar=-2.0)
    obs = terminator.simulate(p, R=100, n_transits=5)
    obs.detect()

Knobs: `T_eq_K, R_p_RJ, M_p_MJ, R_s_RSun, log_met (−1…3), c_to_o (0.3…1.0), mu,
log_P_cloud_bar, log_a_haze, haze_slope, teff_K, logg, feh, mag, band (J/H/Ks/V),
mag_system ("vega" default, or "ab"), t14_s`. `R_p` is the band-median
(0.40–1.70 µm) transit radius. Fidelity is unchanged from the showcase tables:
isothermal, equilibrium chemistry, gray deck, power-law haze, photon + read
noise only. Validation against the 14 POSEIDON planets is in
`notebooks/06_parametric_model.ipynb`.

**Magnitudes.** slicersim normalizes in AB; archive J magnitudes are Vega.
`terminator.slicer.depth_error` converts (J: +0.90 mag via sncosmo) and reports
the offset in `info`. The committed noise tables were regenerated with this
conversion on 2026-10-04 and are 1.4–1.5× more conservative than before.
```

- [ ] **Step 2: Spec amendments and memory update** as listed in the Files block. Keep the spec's original text and add the amendment section; do not rewrite history.

- [ ] **Step 3: Full suite, commit**

Run: `~/anaconda3/envs/py313/bin/python -m pytest -q`

```bash
git add README.md docs/superpowers/specs/2026-10-04-parametric-transit-simulator-design.md
git commit -m "docs: parametric model usage, spec amendments"
```

Note: `README.md` has pre-existing uncommitted edits from earlier sessions. Stage only the new section with `git add -p README.md` if those edits should stay out of this commit; if they are all consistent with the current code (they are from the 2026-08-30 review pass that was never committed), committing them together is acceptable and the commit body should say so.

---

## Self-review notes

- Spec coverage: data products (T2, T4, T5), runtime API (T3, T5, T6, T7), presets (T7), validation (T8), audit items (T6 Vega/AB and provenance; P_ref defined away in T3), testing section (each bullet has a test above). The web page and its tests are deferred to the second plan by design; the Heroku pin prerequisite (push terminator to GitHub) is a human step and is listed in the handoff.
- Type consistency: `transmission_depth` keyword names match between T3, T7 (`simulate`), and T8 (`validate`); `depth_error` keyword names match between T6, T7 and the fake in `test_simulate.py`; `showcase()` returns every key `simulate` reads.
- Review Focus items 1–5 are tested in T3 (`test_cold_planet_clamps_chemistry_but_not_H`, `test_cloud_deck_monotonic_and_deep_deck_is_clear`, `test_ranges_raise`), T5 (`test_blackbody_fallback_and_ceiling`, `test_clamping_is_reported`), T6 (`test_ab_input_is_not_converted`).
