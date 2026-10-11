"""End-to-end checks against the real committed tables in src/lazuli_exo/data/tables."""
import numpy as np
import pytest

import lazuli_exo

PLANETS = lazuli_exo.list_planets()


def test_fourteen_planets_all_scenarios():
    assert len(PLANETS) == 14
    for p in PLANETS:
        assert lazuli_exo.list_scenarios(p) == [
            "clear", "cloudy", "flat", "hazy", "high_metallicity"]


@pytest.mark.parametrize("planet", PLANETS)
def test_every_planet_observes(planet):
    for scen in lazuli_exo.list_scenarios(planet):
        obs = lazuli_exo.observe(planet, scenario=scen, n_transits=3, R=80)
        assert np.isfinite(obs.depth).all()
        assert np.isfinite(obs.depth_err).all()
        assert (obs.depth_err > 0).all()
        assert 0.0005 < np.median(obs.depth) < 0.1
        sig = obs.detect()
        assert np.isfinite(sig["template_snr_diagonal"])


def test_flat_scenario_detects_nothing():
    # deck at 10 nbar: the Ein tail leaves < 1 ppm of ripple (S/N < 0.004)
    for planet in PLANETS:
        obs = lazuli_exo.observe(planet, scenario="flat", R=80)
        assert obs.detect()["template_snr_diagonal"] < 0.01
        np.testing.assert_allclose(obs.depth, obs.flat_depth, rtol=1e-3)


def test_observe_matches_simulate():
    """Issue #3: the showcase path and the parametric path are one model."""
    pytest.importorskip("slicersim")
    pytest.importorskip("sncosmo")
    obs = lazuli_exo.observe("WASP-39 b", scenario="clear", R=100)
    sim = lazuli_exo.simulate(lazuli_exo.presets.showcase("WASP-39 b", "clear"), R=100)
    np.testing.assert_allclose(obs.model_depth, sim.model_depth, rtol=1e-12)
    # binned depths differ only through the within-pixel photon weights (star
    # table R=10000 vs phoenix_sed R=1000) and the channel 1/sigma^2 weights
    np.testing.assert_allclose(obs.depth, sim.depth, rtol=2e-3)
    assert obs.meta["instrument_response"]["photon_weighting"] == "star table R=10000"
    assert sim.meta["instrument_response"]["photon_weighting"] == "phoenix_sed R=1000"
    assert obs.flat_depth == pytest.approx(sim.flat_depth)
    # noise: precomputed table (POSEIDON star) vs live slicersim (PHOENIX grid)
    ratio = np.median(obs.depth_err / sim.depth_err)
    assert 0.8 < ratio < 1.25, ratio


def test_clear_beats_cloudy():
    # a grey deck mutes features -> lower template S/N
    for planet in ["WASP-39 b", "HD 209458 b"]:
        clear = lazuli_exo.observe(planet, scenario="clear", R=80).detect()
        cloudy = lazuli_exo.observe(planet, scenario="cloudy", R=80).detect()
        assert clear["template_snr_diagonal"] > cloudy["template_snr_diagonal"]


def test_bright_hot_jupiter_beats_faint_temperate():
    hj = lazuli_exo.observe("HD 209458 b", scenario="clear", R=80).detect()
    tp = lazuli_exo.observe("TRAPPIST-1 e", scenario="clear", R=80).detect()
    assert hj["template_snr_diagonal"] > tp["template_snr_diagonal"]


def test_sqrt_n_scaling_on_significance():
    s1 = lazuli_exo.observe("WASP-39 b", n_transits=1, R=80).detect()["template_snr_diagonal"]
    s4 = lazuli_exo.observe("WASP-39 b", n_transits=4, R=80).detect()["template_snr_diagonal"]
    assert s4 == pytest.approx(2 * s1, rel=1e-6)


def test_runtime_has_no_heavy_imports():
    """`import lazuli_exo` must stay numpy-only (checked in a fresh process,
    because other tests legitimately import slicersim/sncosmo here)."""
    import subprocess
    import sys
    heavy = ["POSEIDON", "slicersim", "sncosmo", "jax", "jaxoplanet", "scipy",
             "pandas", "astropy", "matplotlib", "flask"]
    code = ("import sys, lazuli_exo; print(sorted(m for m in sys.modules "
            f"if m.split('.')[0] in {heavy!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
