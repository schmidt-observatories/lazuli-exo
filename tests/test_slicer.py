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


def test_eol_instrument_builds_via_overlay(sun):
    _, err, info = slicer.depth_error(*sun, mag=10.0, t14_s=7200.0, instrument="lazuli_eol")
    assert np.all(np.isfinite(err)) and info["instrument"] == "lazuli_eol"


@pytest.mark.parametrize('mag,instrument', [
    (6.0, 'lazuli_cbe'), (10.0, 'lazuli_cbe'), (10.0, 'lazuli_eol')])
def test_cached_noise_matches_live_uncached(sun, mag, instrument):
    slicer.clear_noise_cache()
    kwargs = dict(mag=mag, instrument=instrument, t14_s=3600., n_transits=3)
    cached = slicer.depth_error(*sun, **kwargs)
    hit = slicer.depth_error(*sun, **kwargs)
    reference = slicer.depth_error(*sun, **kwargs, use_cache=False)
    for actual in (cached, hit):
        np.testing.assert_array_equal(actual[0], reference[0])
        np.testing.assert_array_equal(actual[1], reference[1])
        assert actual[2] == reference[2]


def test_clear_noise_cache_invalidates_magnitude_zero_points(monkeypatch):
    slicer.clear_noise_cache()
    original = slicer._sncosmo_band
    calls = []
    def counted(band):
        calls.append(band)
        return original(band)
    monkeypatch.setattr(slicer, '_sncosmo_band', counted)
    first = slicer.vega_to_ab_offset('J')
    assert slicer.vega_to_ab_offset('J') == first
    assert calls == ['J']
    slicer.clear_noise_cache()
    assert slicer.vega_to_ab_offset('J') == first
    assert calls == ['J', 'J']
