import importlib

import numpy as np
import pytest

import lazuli_exo

# lazuli_exo.simulate (the attribute) is the function; grab the module
sim_mod = importlib.import_module("lazuli_exo.simulate")


def fake_depth_error(wl_A, f_lambda, *, mag, band, mag_system, t14_s,
                     n_transits=1, instrument, **setup):
    wl_um = np.geomspace(0.4, 1.7, 500)
    return wl_um, np.full(500, 1e-4) / np.sqrt(n_transits), {
        "mag_ab": mag + 0.9, "vega_ab_offset": 0.9, "nsaturated_detpx": 0,
        "fully_desaturated": True, "t_obs_s": 60.0,
        "nmd_used": list(setup.get("nmd") or [16, 8, 0]),
        "readout_mode": "fixed" if setup.get("nmd") else "auto", "setup": setup,
        "slicersim_version": "fake", "instrument": instrument,
        "trimmed_channels": 0}


@pytest.fixture
def both_grids(zeta_fixture, phoenix_fixture):
    # the two fixtures write into the same tmp_path/tables
    return zeta_fixture


PARAMS = dict(name="Testar-1 b", T_eq_K=1000.0, R_p_RJ=1.0, M_p_MJ=0.6,
              R_s_RSun=1.0, teff_K=5000.0, logg=4.5, feh=0.0, mag=10.0,
              t14_s=7200.0)


def test_simulate_returns_simulated_spectrum(both_grids, monkeypatch):
    monkeypatch.setattr(sim_mod, "depth_error", fake_depth_error)
    obs = lazuli_exo.simulate(PARAMS, R=50, n_transits=4, tables_dir=both_grids)
    assert isinstance(obs, lazuli_exo.SimulatedSpectrum)
    assert obs.planet == "Testar-1 b" and obs.scenario == "custom"
    assert obs.wavelength.shape == obs.depth.shape == obs.depth_err.shape
    np.testing.assert_allclose(obs.flat_depth, (7.1492e7 / 6.957e8) ** 2)
    assert obs.meta["noise"]["mag_ab"] == 10.9
    assert obs.meta["atmosphere"]["mu_source"] == "chemistry"
    assert obs.detect()["template_snr_diagonal"] > 0
    # n_transits reaches the noise function (not applied twice)
    obs1 = lazuli_exo.simulate(PARAMS, R=50, n_transits=1, tables_dir=both_grids)
    np.testing.assert_allclose(obs1.depth_err / obs.depth_err, 2.0)


def test_simulate_instrument_response(both_grids, monkeypatch):
    import warnings
    monkeypatch.setattr(sim_mod, "depth_error", fake_depth_error)
    with pytest.warns(UserWarning, match="exceeds the instrument resolving power"):
        lazuli_exo.simulate(PARAMS, R=300, tables_dir=both_grids)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs = lazuli_exo.simulate(PARAMS, R=50, tables_dir=both_grids)
    ir = obs.meta["instrument_response"]
    assert ir["photon_weighting"] == "phoenix_sed R=1000" and ir["edges_snapped"]
    assert 140 < ir["R_instr_min"] < 160


def test_missing_required_key_raises(both_grids, monkeypatch):
    monkeypatch.setattr(sim_mod, "depth_error", fake_depth_error)
    with pytest.raises(ValueError, match="t14_s"):
        lazuli_exo.simulate({k: v for k, v in PARAMS.items() if k != "t14_s"},
                            tables_dir=both_grids)


def test_simulate_forwards_the_instrument_setup(both_grids, monkeypatch):
    monkeypatch.setattr(sim_mod, "depth_error", fake_depth_error)
    obs = sim_mod.simulate(PARAMS, tables_dir=both_grids, nmd=(8, 4, 0), tframe=0.7,
                           spectrograph_mode="narrow", guiding_sigma=0.01, zodi_ab_mag=22.5)
    n = obs.meta["noise"]
    assert n["nmd_used"] == [8, 4, 0] and n["readout_mode"] == "fixed"
    assert n["setup"] == dict(nmd=(8, 4, 0), tframe=0.7, spectrograph_mode="narrow",
                              guiding_sigma=0.01, zodi_ab_mag=22.5)
