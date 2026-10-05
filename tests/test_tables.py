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
    tables.rebuild_index(tdir)
    wl2, d2, meta = tables.load_model(tdir, "Testar-1 b", "clear")
    np.testing.assert_allclose(wl2, wl)
    np.testing.assert_allclose(d2, depth)
    assert meta["scenario"] == "clear"


def test_noise_roundtrip(tdir):
    wl = np.linspace(0.4, 1.7, 20)
    err = np.full(20, 2e-4)
    # Write a model table first so rebuild_index discovers the planet
    tables.write_model_table(tdir, "Testar-1 b", "clear", wl, np.full(20, 0.01),
                             meta={})
    tables.write_noise_table(tdir, "Testar-1 b", wl, err, meta={})
    tables.rebuild_index(tdir)
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


def test_ld_roundtrip_and_index(tdir):
    _write_minimal(tdir)
    wl = np.geomspace(0.4, 1.7, 48)
    u1 = np.linspace(0.6, 0.2, 48)
    u2 = np.linspace(0.1, 0.25, 48)
    tables.write_ld_table(tdir, "Testar-1 b", wl, u1, u2,
                          meta={"ld_grid": "phoenix"})
    tables.rebuild_index(tdir)
    wl2, u1b, u2b, meta = tables.load_ld(tdir, "Testar-1 b")
    np.testing.assert_allclose(wl2, wl)
    np.testing.assert_allclose(u1b, u1)
    np.testing.assert_allclose(u2b, u2)
    assert meta["ld_grid"] == "phoenix"
    assert meta["planet"] == "Testar-1 b"
    idx = tables.load_index(tdir)
    assert idx["planets"]["Testar-1 b"]["has_ld"] is True


def test_ld_missing_raises(tdir):
    _write_minimal(tdir)
    with pytest.raises(FileNotFoundError, match="make_ld_tables"):
        tables.load_ld(tdir, "Testar-1 b")


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
    assert g["zeta"].dtype == np.float16 and g["zeta"].shape == (2, 2, 1, 50)
    assert g["mu"].dtype == np.float32 and g["mu"].shape == (2, 2, 1)
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
