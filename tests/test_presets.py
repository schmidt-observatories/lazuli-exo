import pytest

from terminator import presets


def test_showcase_maps_registry_and_scenario():
    p = presets.showcase("WASP-39 b", "hazy")
    assert p["name"] == "WASP-39 b" and p["scenario"] == "hazy"
    assert p["R_p_RJ"] == 1.279 and p["teff_K"] == 5485.0 and p["mag"] == 10.663
    assert p["band"] == "J" and p["mag_system"] == "vega"
    assert p["log_a_haze"] == 2.0 and p["haze_slope"] == -6.0 and p["log_P_cloud_bar"] is None


def test_flat_is_a_high_deck():
    # -8: with native-R line cores reaching 1e-7 bar and the Ein deck keeping
    # partial absorption above the deck, -6 left 240-760 ppm pseudo-features
    assert presets.showcase("GJ 1214 b", "flat")["log_P_cloud_bar"] == -8.0


def test_unknown_raise():
    with pytest.raises(KeyError):
        presets.showcase("Nope b", "clear")
    with pytest.raises(KeyError, match="scenario"):
        presets.showcase("GJ 1214 b", "foggy")
