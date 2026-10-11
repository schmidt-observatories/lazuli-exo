import numpy as np
import pytest

from lazuli_exo import star, tables


def test_node_recovery(phoenix_fixture):
    wl_A, f, info = star.phoenix_sed(4000.0, 4.5, 0.0, tables_dir=phoenix_fixture)
    wl_um = wl_A / 1e4
    expect = 10 ** (np.log10(4000.0) + 0.45 - 2 * np.log10(wl_um))
    np.testing.assert_allclose(f, expect, rtol=2e-6)
    assert info["fallback"] is None and info["clamped"] == {}
    assert info["grid"] == "phoenix" and info["teff_used"] == 4000.0


def test_interpolates_in_log_teff(phoenix_fixture):
    _, f_lo, _ = star.phoenix_sed(3000.0, 4.5, 0.0, tables_dir=phoenix_fixture)
    _, f_mid, _ = star.phoenix_sed(np.sqrt(3000.0 * 4000.0), 4.5, 0.0, tables_dir=phoenix_fixture)
    _, f_hi, _ = star.phoenix_sed(4000.0, 4.5, 0.0, tables_dir=phoenix_fixture)
    # log-space midpoint -> geometric mean of the fluxes
    np.testing.assert_allclose(f_mid, np.sqrt(f_lo * f_hi), rtol=1e-6)


def test_logg_and_feh_clamping_is_reported(phoenix_fixture):
    _, f, info = star.phoenix_sed(3500.0, 5.7, 0.7, tables_dir=phoenix_fixture)
    assert info["clamped"] == {"logg": (5.7, 5.5), "feh": (0.7, 0.0)}
    assert info["logg_used"] == 5.5 and info["feh_used"] == 0.0
    _, f_edge, _ = star.phoenix_sed(3500.0, 5.5, 0.0, tables_dir=phoenix_fixture)
    np.testing.assert_allclose(f, f_edge)
    # negative [Fe/H] clamps up to the single solar node too
    _, _, info = star.phoenix_sed(3500.0, 4.5, -0.3, tables_dir=phoenix_fixture)
    assert info["clamped"] == {"feh": (-0.3, 0.0)}


def test_teff_floor_clamps_and_ceiling_raises(phoenix_fixture):
    wl_A, f, info = star.phoenix_sed(1800.0, 5.0, 0.0, tables_dir=phoenix_fixture)
    assert info["fallback"] is None and info["grid"] == "phoenix"
    assert info["clamped"] == {"teff": (1800.0, 2000.0)}
    assert info["teff_K"] == 1800.0 and info["teff_used"] == 2000.0
    _, f_floor, _ = star.phoenix_sed(2000.0, 5.0, 0.0, tables_dir=phoenix_fixture)
    np.testing.assert_allclose(f, f_floor)
    with pytest.raises(ValueError, match="12000"):
        star.phoenix_sed(13000.0, 4.0, 0.0, tables_dir=phoenix_fixture)


def test_nonpositive_teff_raises(phoenix_fixture):
    for bad in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError, match="T_eff"):
            star.phoenix_sed(bad, 4.5, 0.0, tables_dir=phoenix_fixture)


def test_shipped_grid_axes():
    """Issue #34: solar [Fe/H] only, log g to 5.5, T_eff 2000-12000 K."""
    g = tables.load_phoenix_grid()
    np.testing.assert_array_equal(g["feh"], [0.0])
    np.testing.assert_array_equal(g["logg"], [3.5, 4.0, 4.5, 5.0, 5.5])
    assert g["teff_K"].min() == 2000 and g["teff_K"].max() == 12000
    assert g["log_flam"].shape == (len(g["teff_K"]), 5, 1, len(g["wavelength_um"]))
    assert g["meta"]["folder"] == "phoenixm00" and "solar" in g["meta"]["metallicity"]


def test_shipped_grid_hot_star_and_high_logg():
    _, f, info = star.phoenix_sed(9000.0, 4.0, 0.0)          # KELT-20-like: used to raise
    assert np.all(np.isfinite(f)) and info["clamped"] == {}
    _, f, info = star.phoenix_sed(2566.0, 5.24, 0.05)        # TRAPPIST-1: used to clamp log g
    assert info["clamped"] == {"feh": (0.05, 0.0)} and info["logg_used"] == 5.24
    _, f50, _ = star.phoenix_sed(2566.0, 5.0, 0.0)
    _, f55, _ = star.phoenix_sed(2566.0, 5.5, 0.0)
    assert not np.allclose(f, f50) and not np.allclose(f, f55)
    _, _, info = star.phoenix_sed(3101.0, 5.03, 0.24)         # GJ 1214: [Fe/H] ignored
    assert info["feh_used"] == 0.0 and info["clamped"] == {"feh": (0.24, 0.0)}
