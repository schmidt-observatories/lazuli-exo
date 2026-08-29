"""End-to-end checks against the real committed tables in data/tables."""
import numpy as np
import pytest

import terminator

PLANETS = terminator.list_planets()


def test_fourteen_planets_all_scenarios():
    assert len(PLANETS) == 14
    for p in PLANETS:
        assert terminator.list_scenarios(p) == [
            "clear", "cloudy", "flat", "hazy", "high_metallicity"]


@pytest.mark.parametrize("planet", PLANETS)
def test_every_planet_observes(planet):
    for scen in terminator.list_scenarios(planet):
        obs = terminator.observe(planet, scenario=scen, n_transits=3, R=80)
        assert np.isfinite(obs.depth).all()
        assert np.isfinite(obs.depth_err).all()
        assert (obs.depth_err > 0).all()
        assert 0.0005 < np.median(obs.depth) < 0.1
        sig = obs.detect()
        assert np.isfinite(sig["atmosphere"])


def test_flat_scenario_detects_nothing():
    for planet in PLANETS:
        obs = terminator.observe(planet, scenario="flat", R=80)
        assert obs.detect()["atmosphere"] < 1e-6


def test_clear_beats_cloudy():
    # a grey deck mutes features -> lower detection significance
    for planet in ["WASP-39 b", "HD 209458 b"]:
        clear = terminator.observe(planet, scenario="clear", R=80).detect()
        cloudy = terminator.observe(planet, scenario="cloudy", R=80).detect()
        assert clear["atmosphere"] > cloudy["atmosphere"]


def test_bright_hot_jupiter_beats_faint_temperate():
    hj = terminator.observe("HD 209458 b", scenario="clear", R=80).detect()
    tp = terminator.observe("TRAPPIST-1 e", scenario="clear", R=80).detect()
    assert hj["atmosphere"] > tp["atmosphere"]


def test_sqrt_n_scaling_on_significance():
    s1 = terminator.observe("WASP-39 b", n_transits=1, R=80).detect()["atmosphere"]
    s4 = terminator.observe("WASP-39 b", n_transits=4, R=80).detect()["atmosphere"]
    assert s4 == pytest.approx(2 * s1, rel=1e-6)


def test_runtime_has_no_heavy_imports():
    import sys
    for mod in ["POSEIDON", "slicersim", "jax", "jaxoplanet", "scipy",
                "pandas", "astropy"]:
        assert not any(m == mod or m.startswith(mod + ".") for m in sys.modules), mod
