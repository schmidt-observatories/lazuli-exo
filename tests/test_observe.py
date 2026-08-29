import numpy as np
import pytest

import slixer
from conftest import PLANET, BASE_DEPTH, SIGMA_CH  # pytest puts tests/ on sys.path


def test_observe_shapes(fixture_tables):
    obs = slixer.observe(PLANET, scenario="clear", n_transits=1, R=50,
                         tables_dir=fixture_tables)
    assert obs.wavelength.shape == obs.depth.shape == obs.depth_err.shape
    assert np.isfinite(obs.depth).all() and np.isfinite(obs.depth_err).all()
    assert obs.planet == PLANET and obs.scenario == "clear"


def test_flat_scenario_is_flat(fixture_tables):
    obs = slixer.observe(PLANET, scenario="flat", R=50,
                         tables_dir=fixture_tables)
    np.testing.assert_allclose(obs.depth, BASE_DEPTH, rtol=1e-12)


def test_error_scaling_with_transits(fixture_tables):
    o1 = slixer.observe(PLANET, n_transits=1, R=50, tables_dir=fixture_tables)
    o9 = slixer.observe(PLANET, n_transits=9, R=50, tables_dir=fixture_tables)
    np.testing.assert_allclose(o1.depth_err / o9.depth_err, 3.0)


def test_error_magnitude(fixture_tables):
    # 500 uniform channels at R such that ~10 channels/bin
    obs = slixer.observe(PLANET, n_transits=1, R=50, tables_dir=fixture_tables)
    # every bin: sigma = SIGMA_CH / sqrt(n_channels_in_bin); just check bounds
    assert (obs.depth_err < SIGMA_CH).all()
    assert (obs.depth_err > SIGMA_CH / np.sqrt(30)).all()


def test_r_too_high_raises(fixture_tables):
    with pytest.raises(ValueError, match="native noise sampling"):
        slixer.observe(PLANET, R=2000, tables_dir=fixture_tables)


def test_simulate_points_reproducible(fixture_tables):
    obs = slixer.observe(PLANET, R=50, tables_dir=fixture_tables)
    a = obs.simulate_points(seed=42)
    b = obs.simulate_points(seed=42)
    np.testing.assert_array_equal(a, b)
    scatter = np.std((a - obs.depth) / obs.depth_err)
    assert 0.7 < scatter < 1.3


def test_unknown_planet_raises(fixture_tables):
    with pytest.raises(KeyError, match="Testar-1 b"):
        slixer.observe("Nope-1 b", tables_dir=fixture_tables)
