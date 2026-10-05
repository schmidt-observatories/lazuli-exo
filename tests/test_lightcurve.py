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


# --- transit_flux: quadratic-LD occultation model ------------------------

def _uniform_disk_analytic(z, k):
    """Mandel & Agol (2002) lambda_e: occulted flux fraction, u1=u2=0."""
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    out[z <= 1.0 - k] = k**2
    part = (z > 1.0 - k) & (z < 1.0 + k)
    zp = z[part]
    k0 = np.arccos((zp**2 + k**2 - 1.0) / (2.0 * k * zp))
    k1 = np.arccos((zp**2 + 1.0 - k**2) / (2.0 * zp))
    out[part] = (k**2 * k0 + k1
                 - np.sqrt(np.maximum(
                     zp**2 - (1.0 + zp**2 - k**2)**2 / 4.0, 0.0))) / np.pi
    return out


def test_transit_flux_uniform_disk_matches_analytic():
    """Documented accuracy: <= 1e-6 absolute for k <= 0.2 at the default
    quadrature resolution (covers every registry target's radius ratio)."""
    z = np.linspace(0.0, 1.3, 400)
    for k in (0.05, 0.1, 0.2):
        flux = lightcurve.transit_flux(z, k, 0.0, 0.0)
        np.testing.assert_allclose(flux, 1.0 - _uniform_disk_analytic(z, k),
                                   atol=1e-6)


def test_transit_flux_small_planet_ld():
    """k -> 0: dF = k^2 I(z) / Ibar with quadratic I(mu)."""
    k, u1, u2 = 0.01, 0.4, 0.2
    z = np.array([0.0, 0.3, 0.6, 0.85])
    mu = np.sqrt(1.0 - z**2)
    inten = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu)**2
    ibar = 1.0 - u1 / 3.0 - u2 / 6.0
    expected = 1.0 - k**2 * inten / ibar
    flux = lightcurve.transit_flux(z, k, u1, u2)
    np.testing.assert_allclose(flux, expected, atol=3e-6)


def test_transit_flux_limits():
    flux = lightcurve.transit_flux(np.array([1.2, 5.0]), 0.1, 0.5, 0.1)
    np.testing.assert_allclose(flux, 1.0)
    # limb-darkened center-of-disk crossing is deeper than uniform
    f_ld = lightcurve.transit_flux(np.array([0.0]), 0.1, 0.5, 0.1)
    f_uni = lightcurve.transit_flux(np.array([0.0]), 0.1, 0.0, 0.0)
    assert f_ld[0] < f_uni[0]


# --- LD + orbital-geometry path (real repo tables + registry) -------------

@pytest.fixture
def obs_real():
    return observe("WASP-39 b", scenario="clear", n_transits=1, R=100)


def test_real_planet_uses_quadratic_ld(obs_real):
    lc = lightcurve.light_curves(obs_real, n_groups=3, seed=0)
    assert lc.meta["shape"] == "quadratic-ld"
    assert 0.0 <= lc.meta["b"] < 1.0
    assert lc.meta["a_over_rs"] > 1.0
    assert lc.u1.shape == lc.u2.shape == (3,)
    assert np.isfinite(lc.u1).all() and np.isfinite(lc.u2).all()
    # limb darkening weakens toward the red
    assert lc.u1[0] > lc.u1[-1]
    # mid-transit is deeper than the box depth for u1 > 0
    i_mid = np.argmin(np.abs(lc.time_hr))
    assert lc.flux_model[0, i_mid] < 1.0 - lc.depth[0]
    # out of transit is unity
    assert lc.flux_model[0, 0] == pytest.approx(1.0)
    assert lc.flux_model[0, -1] == pytest.approx(1.0)


def test_fixture_planet_falls_back_to_trapezoid(obs):
    lc = lightcurve.light_curves(obs, n_groups=1, seed=0)
    assert lc.meta["shape"] == "trapezoid"
    assert lc.u1 is None and lc.u2 is None


def test_eccentric_mid_transit_separation_equals_b():
    """The flux model must use the same (eccentric) impact parameter the
    metadata reports: z(0) = b, not a/R* cos i."""
    obs = observe("HAT-P-11 b", scenario="clear", n_transits=1, R=100)
    lc = lightcurve.light_curves(obs, n_groups=1, seed=0)
    assert lc.meta["shape"] == "quadratic-ld"
    i_mid = np.argmin(np.abs(lc.time_hr))
    k = np.sqrt(lc.depth[0])
    expected = lightcurve.transit_flux(
        np.array([lc.meta["b"]]), k, lc.u1[0], lc.u2[0])[0]
    assert lc.flux_model[0, i_mid] == pytest.approx(expected, abs=1e-5)


def test_omega_zero_is_not_treated_as_missing(monkeypatch):
    """A valid w_deg = 0.0 must not be replaced by the 90-deg default."""
    from terminator import targets as targets_mod
    t = dict(targets_mod.TARGETS["WASP-39 b"], ecc=0.3, w_deg=0.0)
    monkeypatch.setitem(targets_mod.TARGETS, "WASP-39 b", t)
    obs = observe("WASP-39 b", scenario="clear", n_transits=1, R=100)
    lc = lightcurve.light_curves(obs, n_groups=1, seed=0)
    # with w = 0: b = a/R* cos i (1 - e^2) / (1 + e sin 0)
    expected_b = t["a_over_rs"] * np.cos(np.radians(t["inc_deg"])) \
        * (1.0 - 0.3**2)
    assert lc.meta["b"] == pytest.approx(expected_b)
