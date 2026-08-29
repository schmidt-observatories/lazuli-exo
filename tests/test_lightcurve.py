"""Tests for terminator.lightcurve against the analytic fixture tables."""
import numpy as np
import pytest

from terminator import lightcurve, observe

from conftest import PLANET  # pytest puts tests/ on sys.path

T14_S = 7200.0
T_OBS_S = 60.0


@pytest.fixture
def obs(fixture_tables):
    return observe(PLANET, scenario="clear", n_transits=4, R=50,
                   tables_dir=fixture_tables)


def test_white_light_shape_and_grid(obs):
    lc = lightcurve.light_curves(obs, n_groups=1, seed=0)
    assert lc.flux_model.shape == lc.flux.shape == (1, lc.time_hr.size)
    # window is +/- t14 around mid-transit, native cadence
    assert lc.t14_s == T14_S
    assert lc.cadence_s == pytest.approx(T_OBS_S)
    assert lc.time_hr.max() <= T14_S / 3600.0
    assert lc.time_hr.min() >= -T14_S / 3600.0
    dt = np.diff(lc.time_hr) * 3600.0
    np.testing.assert_allclose(dt, T_OBS_S)


def test_white_light_depth_is_weighted_mean(obs):
    lc = lightcurve.light_curves(obs, n_groups=1, seed=0)
    w = 1.0 / obs.depth_err**2
    expected = np.sum(w * obs.depth) / np.sum(w)
    assert lc.depth[0] == pytest.approx(expected)
    # model bottoms out at 1 - depth and is 1 out of transit
    assert lc.flux_model.min() == pytest.approx(1.0 - expected)
    assert lc.flux_model[0, 0] == pytest.approx(1.0)
    assert lc.flux_model[0, -1] == pytest.approx(1.0)


def test_point_noise_consistent_with_depth_precision(obs):
    """sigma_point = sigma_band * sqrt(t14 / (2 * cadence)): recombining
    the light curve must recover the band depth precision."""
    lc = lightcurve.light_curves(obs, n_groups=1, seed=0)
    w = 1.0 / obs.depth_err**2
    sigma_band = 1.0 / np.sqrt(np.sum(w))
    expected = sigma_band * np.sqrt(T14_S / (2.0 * lc.cadence_s))
    assert lc.sigma_point[0] == pytest.approx(expected)


def test_grouping(obs):
    lc = lightcurve.light_curves(obs, n_groups=3, seed=0)
    assert lc.flux.shape[0] == 3
    # contiguous bands tile the binned range in order
    assert lc.wl_lo[0] == pytest.approx(obs.wavelength[0])
    assert lc.wl_hi[-1] == pytest.approx(obs.wavelength[-1])
    assert (lc.wl_lo[1:] > lc.wl_lo[:-1]).all()
    # fixture spectrum has its 1.4 um bump in the reddest band
    assert lc.depth[2] > lc.depth[0]
    assert lc.depth[2] > lc.depth[1]
    assert len(lc.labels) == 3


def test_seed_reproducibility(obs):
    a = lightcurve.light_curves(obs, n_groups=2, seed=11)
    b = lightcurve.light_curves(obs, n_groups=2, seed=11)
    c = lightcurve.light_curves(obs, n_groups=2, seed=12)
    np.testing.assert_array_equal(a.flux, b.flux)
    assert not np.array_equal(a.flux, c.flux)


def test_noise_scatter_matches_sigma(obs):
    lc = lightcurve.light_curves(obs, n_groups=1, seed=5)
    resid = lc.flux[0] - lc.flux_model[0]
    assert np.std(resid) == pytest.approx(lc.sigma_point[0], rel=0.15)


def test_more_transits_less_noise(fixture_tables):
    o1 = observe(PLANET, n_transits=1, R=50, tables_dir=fixture_tables)
    o9 = observe(PLANET, n_transits=9, R=50, tables_dir=fixture_tables)
    s1 = lightcurve.light_curves(o1, seed=0).sigma_point[0]
    s9 = lightcurve.light_curves(o9, seed=0).sigma_point[0]
    assert s9 == pytest.approx(s1 / 3.0)


def test_max_points_caps_cadence(obs):
    lc = lightcurve.light_curves(obs, n_groups=1, seed=0, max_points=50)
    assert lc.time_hr.size <= 50
    assert lc.cadence_s == pytest.approx(2.0 * T14_S / 50)


def test_n_groups_validation(obs):
    with pytest.raises(ValueError):
        lightcurve.light_curves(obs, n_groups=0)
    with pytest.raises(ValueError):
        lightcurve.light_curves(obs, n_groups=obs.wavelength.size + 1)


def test_convenience_method(obs):
    lc = obs.light_curves(n_groups=2, seed=3)
    direct = lightcurve.light_curves(obs, n_groups=2, seed=3)
    np.testing.assert_array_equal(lc.flux, direct.flux)
