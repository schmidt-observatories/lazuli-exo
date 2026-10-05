import numpy as np
import pytest

from lazuli_exo import star


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


def test_nonpositive_teff_raises(phoenix_fixture):
    for bad in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError, match="T_eff"):
            star.phoenix_sed(bad, 4.5, 0.0, tables_dir=phoenix_fixture)
