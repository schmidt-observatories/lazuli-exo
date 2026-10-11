"""Cache behavior is tested without the optional instrument dependencies."""
import sys
from functools import lru_cache
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from lazuli_exo import slicer


@pytest.fixture
def instrument(monkeypatch):
    package = ModuleType('slicersim')
    package.__version__ = 'test-1'
    lazuli = ModuleType('slicersim.lazuli')

    class Target:
        calls = 0
        fail = False

        def __init__(self, wl, flux, *, mag, band, instrument):
            Target.calls += 1
            Target.last = self
            if Target.fail:
                raise RuntimeError('configuration unavailable')
            self.mag = mag
            self.simulation = self
            self.detector = SimpleNamespace(nmd=(8, 2, 0), min_group=2,
                                            nsaturated_detpx=0)
            self.observing_time = 60.
            self.extraction = {'nramps': 1}

        def change_configuration(self, instrument):
            self.observing_time = 80.

        def change_detector(self, *, nmd):
            self.detector.nmd = nmd

        def change_spectrograph(self, mode):
            self.mode = mode

        def update(self, **kw):
            if 'tframe' in kw:
                self.observing_time = 60. * kw['tframe'] / 2.83

        def get_parameter(self, name):
            return {'tframe': 2.83}[name]

        def get_snr(self):
            self.detector.nsaturated_detpx = int(self.mag < 8 or self.detector.nmd[0] > 4)
            result = np.full(500, 100. * 10**(-0.2 * (self.mag-10)))
            result[-1] = np.nan
            return result

        def get_lbda(self):
            return np.geomspace(4000., 17000., 500)

    lazuli.LazuliTarget = Target
    monkeypatch.setitem(sys.modules, 'slicersim', package)
    monkeypatch.setitem(sys.modules, 'slicersim.lazuli', lazuli)
    monkeypatch.setattr(slicer, 'vega_to_ab_offset', lru_cache(maxsize=32)(lambda band: 0.9))
    slicer.clear_noise_cache()
    yield Target, package, lazuli
    slicer.clear_noise_cache()


def run(**kwargs):
    params = dict(wl_A=np.array([4000., 10000., 17000.]),
                  f_lambda=np.array([1., 2., 1.]), mag=10., t14_s=7200.,
                  mag_system='ab')
    params.update(kwargs)
    return slicer.depth_error(**params)


def assert_equal(left, right):
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2] == right[2]


def test_hit_reuses_instrument_and_scales_request(instrument):
    target, _, _ = instrument
    first = run()
    assert_equal(first, run())
    scaled = run(t14_s=28800., n_transits=9)
    np.testing.assert_allclose(first[1] / scaled[1], 6.)
    assert scaled[2]['t14_s'] == 28800.
    assert scaled[2]['trimmed_channels'] == 1
    assert target.calls == 1
    assert_equal(scaled, run(t14_s=28800., n_transits=9, use_cache=False))
    assert target.calls == 2


@pytest.mark.parametrize('changed', [
    {'wl_A': np.array([4000., 11000., 17000.])},
    {'f_lambda': np.array([1., 3., 1.])},
    {'mag': 10.1}, {'band': 'H'}, {'mag_system': 'vega'},
    {'instrument': 'lazuli_eol'},
])
def test_physical_input_changes_miss(instrument, changed):
    target, _, _ = instrument
    run()
    run(**changed)
    assert target.calls == 2


def test_version_and_implementation_changes_miss(instrument):
    target, package, lazuli = instrument
    run()
    package.__version__ = 'test-2'
    run()
    class Replacement(target):
        pass
    lazuli.LazuliTarget = Replacement
    run()
    assert target.calls == 3


def test_same_physical_brightness_reuses_but_metadata_is_current(instrument):
    target, _, _ = instrument
    vega = run(mag=10., mag_system='vega')
    ab = run(mag=10.9, mag_system='ab')
    np.testing.assert_array_equal(vega[1], ab[1])
    assert ab[2]['mag_input'] == 10.9 and ab[2]['vega_ab_offset'] == 0.
    assert vega[2]['mag_input'] == 10. and vega[2]['vega_ab_offset'] == 0.9
    assert target.calls == 1


def test_callers_cannot_poison_cache(instrument):
    original = run()
    mutated = run()
    mutated[0][:] = -1
    mutated[1][:] = -1
    mutated[2]['nmd_used'][0] = 999
    mutated[2]['default_nmd'][0] = 999
    mutated[2]['t_obs_s'] = -1
    assert_equal(original, run())


def test_input_mutation_changes_key(instrument):
    target, _, _ = instrument
    flux = np.array([1., 2., 1.])
    run(f_lambda=flux)
    flux[1] = 3.
    run(f_lambda=flux)
    assert target.calls == 2


def test_clear_bypass_and_bounded_eviction(instrument):
    target, _, _ = instrument
    run(use_cache=False)
    assert slicer._cached_native_noise.cache_info().currsize == 0
    run()
    run(use_cache=False)
    run()
    assert target.calls == 3
    slicer.clear_noise_cache()
    run()
    assert target.calls == 4
    for mag in range(11, 43):
        run(mag=mag)
    assert slicer._cached_native_noise.cache_info().currsize == 32
    count = target.calls
    run()  # magnitude 10 was evicted
    assert target.calls == count + 1


def test_errors_are_not_cached(instrument):
    target, _, _ = instrument
    target.fail = True
    with pytest.raises(ValueError, match='configuration unavailable'):
        run()
    assert slicer._cached_native_noise.cache_info().currsize == 0
    target.fail = False
    run()
    assert target.calls == 2


@pytest.mark.parametrize('mag', [6., 10.])
@pytest.mark.parametrize('config', ['lazuli_cbe', 'lazuli_eol'])
def test_readout_and_saturation_metadata_match_uncached(instrument, mag, config):
    result = run(mag=mag, instrument=config)
    assert_equal(result, run(mag=mag, instrument=config, use_cache=False))
    assert result[2]['fully_desaturated'] == (mag >= 8)
    if mag < 8:
        assert result[2]['nmd_used'] == [2, 1, 0]


def test_atmosphere_and_resolution_changes_reuse_noise(
        instrument, zeta_fixture, phoenix_fixture):
    from lazuli_exo import simulate
    target, _, _ = instrument
    params = dict(T_eq_K=1000., R_p_RJ=1., M_p_MJ=0.6, R_s_RSun=1.,
                  teff_K=5000., logg=4.5, feh=0., mag=10., t14_s=7200.)
    a = simulate(params, R=50, tables_dir=zeta_fixture)
    b = simulate({**params, 'log_P_cloud_bar': -4., 't14_s': 3600.},
                 R=100, n_transits=4, tables_dir=zeta_fixture)
    assert target.calls == 1
    assert a.depth.size != b.depth.size
    assert b.meta['noise']['t14_s'] == 3600.


@pytest.mark.parametrize('kwargs', [
    {'t14_s': np.nan}, {'n_transits': np.inf}, {'mag': np.nan},
    {'f_lambda': [1., np.nan, 2.]}, {'wl_A': [1., 1., 2.]},
    {'f_lambda': [1., 2.]},
])
def test_nonfinite_or_invalid_inputs_do_not_populate_cache(instrument, kwargs):
    target, _, _ = instrument
    with pytest.raises(ValueError):
        run(**kwargs)
    assert target.calls == 0
    assert slicer._cached_native_noise.cache_info().currsize == 0


def test_setup_is_part_of_the_cache_key(instrument):
    Target, _, _ = instrument
    run()
    run(nmd=(4, 2, 0))
    run(nmd=(4, 2, 0))                       # hit
    run(tframe=0.7)
    run(spectrograph_mode='wide')
    assert slicer._cached_native_noise.cache_info().currsize == 4
    assert Target.calls == 4


def test_fixed_nmd_bypasses_desaturation_and_is_reported(instrument):
    # the fake saturates whenever groups > 4: auto halves 8 -> 4, fixed keeps 8
    _, _, auto = run(mag=10.)
    _, _, fixed = run(mag=10., nmd=(8, 2, 0))
    assert auto['readout_mode'] == 'auto' and auto['nmd_used'] == [4, 2, 0] and auto['fully_desaturated']
    assert fixed['readout_mode'] == 'fixed' and fixed['nmd_used'] == [8, 2, 0]
    assert fixed['nsaturated_detpx'] == 1 and fixed['fully_desaturated'] is False
    assert fixed['tframe_s'] == 2.83 and fixed['spectrograph_mode'] == 'narrow'


def test_setup_values_are_normalized_before_keying(instrument):
    Target, _, _ = instrument
    run(nmd=[4, 2, 0], tframe=1, spectrograph_mode='WIDE')
    run(nmd=(4.0, 2.0, 0.0), tframe=1.0, spectrograph_mode='wide')   # same setup -> hit
    assert Target.calls == 1


def test_default_setup_pins_the_narrow_field(instrument):
    # slicersim >= 1.3.3 no longer rescales a fresh LazuliTarget to a field
    # (half the throughput, 1.4x the noise); the live path must not depend on
    # that default, and the noise tables were built in the narrow field.
    Target, _, _ = instrument
    _, _, info = run()
    assert info['spectrograph_mode'] == 'narrow'
    assert Target.last.mode == 'narrow'
