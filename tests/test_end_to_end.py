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
    for planet in PLANETS:
        obs = lazuli_exo.observe(planet, scenario="flat", R=80)
        assert obs.detect()["template_snr_diagonal"] < 1e-6


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
