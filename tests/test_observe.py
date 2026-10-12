import numpy as np
import pytest

import lazuli_exo
from conftest import PLANET, BASE_DEPTH, SIGMA_CH  # pytest puts tests/ on sys.path


def test_observe_shapes(fixture_tables):
    obs = lazuli_exo.observe(PLANET, scenario="clear", n_transits=1, R=50,
                         tables_dir=fixture_tables)
    assert obs.wavelength.shape == obs.depth.shape == obs.depth_err.shape
    assert np.isfinite(obs.depth).all() and np.isfinite(obs.depth_err).all()
    assert obs.planet == PLANET and obs.scenario == "clear"


def test_flat_scenario_is_flat(fixture_tables):
    obs = lazuli_exo.observe(PLANET, scenario="flat", R=50,
                         tables_dir=fixture_tables)
    assert obs.flat_depth == pytest.approx(BASE_DEPTH)
    # deck at 10 nbar: the Ein tail leaves < 1e-3 of the depth
    np.testing.assert_allclose(obs.depth, BASE_DEPTH, rtol=1e-3)
    assert obs.meta["k_source"].startswith("archive")


def test_unknown_scenario_raises(fixture_tables):
    with pytest.raises(KeyError, match="stormy"):
        lazuli_exo.observe(PLANET, scenario="stormy", tables_dir=fixture_tables)


def test_error_scaling_with_transits(fixture_tables):
    o1 = lazuli_exo.observe(PLANET, n_transits=1, R=50, tables_dir=fixture_tables)
    o9 = lazuli_exo.observe(PLANET, n_transits=9, R=50, tables_dir=fixture_tables)
    np.testing.assert_allclose(o1.depth_err / o9.depth_err, 3.0)


def test_error_magnitude(fixture_tables):
    # 500 uniform channels at R such that ~10 channels/bin
    obs = lazuli_exo.observe(PLANET, n_transits=1, R=50, tables_dir=fixture_tables)
    # every bin: sigma = SIGMA_CH / sqrt(n_channels_in_bin); just check bounds
    assert (obs.depth_err < SIGMA_CH).all()
    assert (obs.depth_err > SIGMA_CH / np.sqrt(30)).all()


def test_r_too_high_raises(fixture_tables):
    # whichever grid runs out first (fixture zeta grid or noise channels)
    with pytest.raises(ValueError, match="coarser R|native noise sampling"):
        lazuli_exo.observe(PLANET, R=2000, tables_dir=fixture_tables)


def test_r_above_instrument_resolution_warns(fixture_tables):
    # fixture channels: R_px ~ 345 -> R_instr ~ 150
    import warnings
    with pytest.warns(UserWarning, match="exceeds the instrument resolving power"):
        obs = lazuli_exo.observe(PLANET, R=300, tables_dir=fixture_tables)
    assert len(obs.wavelength) > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs = lazuli_exo.observe(PLANET, R=50, tables_dir=fixture_tables)
    ir = obs.meta["instrument_response"]
    assert ir["edges_snapped"] and ir["lsf_sigma_px"] == 1.15
    assert 140 < ir["R_instr_min"] < ir["R_instr_max"] < 160
    assert ir["photon_weighting"] == "uniform (no star table)"


def test_simulate_points_reproducible(fixture_tables):
    obs = lazuli_exo.observe(PLANET, R=50, tables_dir=fixture_tables)
    a = obs.simulate_points(seed=42)
    b = obs.simulate_points(seed=42)
    np.testing.assert_array_equal(a, b)
    scatter = np.std((a - obs.depth) / obs.depth_err)
    assert 0.7 < scatter < 1.3


def test_unknown_planet_raises(fixture_tables):
    with pytest.raises(KeyError, match="Nope-1 b"):
        lazuli_exo.observe("Nope-1 b", tables_dir=fixture_tables)


# --- unvalidated (residually saturated) noise tables ----------------------

def test_saturated_noise_table_warns():
    """A host whose noise table retains detector saturation (HD 189733 b,
    324 px at the (2,1,0) floor with the 0.7 s subframe; HD 209458 b and
    HAT-P-11 b are clean since the 2026-10-11 tables) must be flagged, not
    silently treated as a validated forecast."""
    with pytest.warns(UserWarning, match="saturat"):
        lazuli_exo.observe("HD 189733 b", scenario="flat", R=50)


def test_clean_noise_table_does_not_warn():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lazuli_exo.observe("WASP-39 b", scenario="flat", R=50)


@pytest.mark.parametrize("bad", [dict(R=0), dict(R=-5), dict(R=2.5), dict(R="100"),
                                 dict(n_transits=0), dict(n_transits=2.5), dict(n_transits=True)])
def test_observe_rejects_non_positive_int_R_and_n_transits(bad):
    # #31 section 1: R=0 was a ZeroDivisionError; n_transits=2.5 was used in
    # the errors but reported as 2
    with pytest.raises(ValueError, match="positive integer"):
        lazuli_exo.observe("WASP-39 b", scenario="flat", **bad)


def test_observe_accepts_integral_floats():
    obs = lazuli_exo.observe("WASP-39 b", scenario="flat", R=50.0, n_transits=3.0)
    assert obs.R == 50 and obs.n_transits == 3
