import numpy as np
import pytest

slicersim = pytest.importorskip("slicersim")
sncosmo = pytest.importorskip("sncosmo")

from lazuli_exo import slicer, star  # noqa: E402


def test_vega_ab_offsets_are_sane():
    assert 0.85 < slicer.vega_to_ab_offset("J") < 0.95
    assert abs(slicer.vega_to_ab_offset("V")) < 0.05
    assert slicer.vega_to_ab_offset("Ks") > slicer.vega_to_ab_offset("H") > slicer.vega_to_ab_offset("J")


def test_unknown_band_raises():
    with pytest.raises(ValueError, match="band"):
        slicer.vega_to_ab_offset("Z")


@pytest.fixture(scope="module")
def sun():
    wl_A, f, _ = star.phoenix_sed(5778.0, 4.44, 0.0)
    return wl_A, f


def test_ab_input_is_not_converted(sun):
    _, e_vega, i_v = slicer.depth_error(*sun, mag=10.0, band="J", mag_system="vega", t14_s=7200.0)
    _, e_ab, i_a = slicer.depth_error(*sun, mag=10.0, band="J", mag_system="ab", t14_s=7200.0)
    assert i_a["vega_ab_offset"] == 0.0 and i_a["mag_ab"] == 10.0
    assert abs(i_v["mag_ab"] - (10.0 + i_v["vega_ab_offset"])) < 1e-9
    # Vega J=10 is AB J~10.9: fainter than AB J=10 -> larger errors
    assert np.nanmedian(e_vega) > np.nanmedian(e_ab)
    assert i_v["fully_desaturated"] in (True, False) and i_v["slicersim_version"]


def test_bright_star_hits_the_floor_and_reports_it(sun):
    _, _, info = slicer.depth_error(*sun, mag=6.0, band="J", t14_s=7200.0)
    assert tuple(info["nmd_used"][:2]) == (2, 1) or info["nsaturated_detpx"] == 0
    assert info["nsaturated_detpx"] >= 0


def test_n_transits_scaling(sun):
    _, e1, _ = slicer.depth_error(*sun, mag=10.0, t14_s=7200.0, n_transits=1)
    _, e4, _ = slicer.depth_error(*sun, mag=10.0, t14_s=7200.0, n_transits=4)
    np.testing.assert_allclose(e1 / e4, 2.0)


def test_unknown_instrument_is_a_valueerror(sun):
    # slicersim raises assorted exceptions for a bad/broken config (KeyError for a
    # toml missing a section); the web layers only translate ValueError into 400s
    with pytest.raises(ValueError, match="instrument"):
        slicer.depth_error(*sun, mag=10.0, t14_s=7200.0, instrument="lazuli_nonexistent")
