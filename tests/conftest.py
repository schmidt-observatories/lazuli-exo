import numpy as np
import pytest

from slixer import tables

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
