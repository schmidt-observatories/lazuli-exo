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
                             meta={"t14_s": 7200.0, "t_obs_s": 60.0})
    tables.rebuild_index(tdir)
    return tdir


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
    wl = np.geomspace(0.35, 1.85, 601)
    base = ZETA0 + ZETA_AMP * np.sin(2 * np.pi * wl / 0.5)
    zeta = (base[None, None, None, :] + 0.1 * met[None, :, None, None]
            + 0.0 * T[:, None, None, None] + 0.0 * co[None, None, :, None])
    mu = (2.3 + 0.5 * met[None, :, None] + 0.0 * T[:, None, None]
          + 0.0 * co[None, None, :])
    tables.write_zeta_grid(tdir, T_K=T, log_met=met, c_to_o=co,
                           wavelength_um=wl, zeta=zeta, mu=mu,
                           meta={"fiducial": FID})
    return tdir
