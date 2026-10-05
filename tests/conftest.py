import numpy as np
import pytest

from lazuli_exo import tables

PLANET = "Testar-1 b"
SIGMA_CH = 1.0e-4    # noise per native channel, one transit
# Fake registry entry for observe(): k = R_p/R_s = 0.1 exactly, so the flat
# scenario sits at BASE_DEPTH; no orbit geometry, so light curves fall back
# to the trapezoid.
R_J_M, R_SUN_M = 7.1492e7, 6.957e8
FIXTURE_TARGET = dict(
    name=PLANET, slug="testar-1-b", provenance="test fixture",
    T_eq_K=1000.0, R_p_RJ=0.1 * R_SUN_M / R_J_M, M_p_MJ=0.6, R_s_RSun=1.0,
    T_eff_K=5000.0, log_g_star=4.5, Met_star=0.0, J_mag=10.0, t14_s=7200.0,
    depth=0.01, a_over_rs=None, inc_deg=None, period_d=None, ecc=None,
    w_deg=None)
BASE_DEPTH = (FIXTURE_TARGET["R_p_RJ"] * R_J_M / R_SUN_M) ** 2


@pytest.fixture
def fixture_tables(zeta_fixture, monkeypatch):
    """Analytic zeta grid + flat noise table + a fake registry planet."""
    from lazuli_exo import targets
    monkeypatch.setitem(targets.TARGETS, PLANET, FIXTURE_TARGET)
    tdir = zeta_fixture
    wl_n = np.geomspace(0.4, 1.7, 500)
    tables.write_noise_table(tdir, PLANET, wl_n,
                             np.full_like(wl_n, SIGMA_CH),
                             meta={"t14_s": 7200.0, "t_obs_s": 60.0})
    tables.rebuild_index(tdir)
    return tdir


# analytic zeta grid: zeta = ZETA0 + ZETA_AMP * sin(2 pi wl / 0.5um) + 0.1*log_met,
# independent of T and C/O; mu = 2.3 + 0.5*log_met
ZETA0, ZETA_AMP = 9.0, 1.0
FIDS = [{"R_p_m": 7.1492e7, "g_ms2": 10.0, "R_s_m": 6.957e8, "P_ref_bar": 10.0, "x_column": 0.0},
        {"R_p_m": 0.3 * 7.1492e7, "g_ms2": 30.0, "R_s_m": 6.957e8, "P_ref_bar": 10.0,
         "x_column": 0.5 * np.log(0.3 * 10.0 / 30.0)}]
FID = FIDS[0]
ZETA_X_SLOPE = 0.8          # d zeta / d x between the two nodes (1 = pure column scaling)


@pytest.fixture
def zeta_fixture(tmp_path):
    tdir = tmp_path / "tables"
    T = np.array([300.0, 600.0, 1000.0, 2000.0])
    met = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    co = np.array([0.3, 0.55, 1.0])
    wl = np.geomspace(0.35, 1.85, 601)
    base = ZETA0 + ZETA_AMP * np.sin(2 * np.pi * wl / 0.5)
    zeta0 = (base[None, None, None, :] + 0.1 * met[None, :, None, None]
             + 0.0 * T[:, None, None, None] + 0.0 * co[None, None, :, None])
    zeta = np.stack([zeta0, zeta0 + ZETA_X_SLOPE * FIDS[1]["x_column"]], axis=3)
    mu = (2.3 + 0.5 * met[None, :, None] + 0.0 * T[:, None, None]
          + 0.0 * co[None, None, :])
    tables.write_zeta_grid(tdir, T_K=T, log_met=met, c_to_o=co,
                           wavelength_um=wl, zeta=zeta, mu=mu,
                           meta={"fiducials": FIDS, "fiducial": FID})
    return tdir


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
