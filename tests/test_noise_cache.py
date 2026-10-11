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

    class Detector:
        """slicersim's ramp timing; the full well is 10 frame-charges of a
        mag-10 source (rate * tframe = 1 at mag 10), so (8,2,0) saturates
        at the last read (16) and (4,2,0) does not (8)."""
        min_group = 2
        gain = 1.
        saturation = 10.

        def __init__(self):
            self.nmd, self.tframe, self.nsaturated_detpx = (8, 2, 0), 2.83, 0

        @property
        def nframes(self):
            n, m, d = self.nmd
            return n * (m + d) - d

        @property
        def integration_time(self):
            n, m, d = self.nmd
            return (n - 1) * (m + d) * self.tframe

        @property
        def exposure_time(self):
            return self.nframes * self.tframe

        def get_qe(self, lbda):
            return np.ones(len(lbda))

        def get_effective_dark(self):
            return 0.

    class Target:
        calls = 0
        fail = False
        nramps = 1
        saturation = 10.

        def __init__(self, wl, flux, *, mag, band, instrument):
            Target.calls += 1
            Target.last = self
            if Target.fail:
                raise RuntimeError('configuration unavailable')
            self.mag = mag
            self.simulation = self
            self.detector = Detector()
            self.detector.saturation = Target.saturation
            self.spectrograph = SimpleNamespace(_ANAMORPHOSE=2, lbda=self.get_lbda())
            self.scene = SimpleNamespace(pointsource=SimpleNamespace(position=(0, 0)))
            self.config_scale = 1.
            self.extraction = {'nramps': Target.nramps}

        @property
        def observing_time(self):
            return self.detector.exposure_time * self.extraction['nramps'] * self.config_scale

        def change_configuration(self, instrument):
            self.config_scale = 80. / 60.

        def change_detector(self, *, nmd):
            self.detector.nmd = nmd

        def change_spectrograph(self, mode):
            self.mode = mode

        def change_pointsource(self, pointsource):
            self.pointsource = pointsource

        def update(self, **kw):
            if 'tframe' in kw:
                self.detector.tframe = kw['tframe']

        def get_parameter(self, name):
            return {'tframe': self.detector.tframe}[name]

        def rate(self):
            """e-/s/pixel of the brightest pixel: one frame-charge per 2.83 s at mag 10."""
            return 10**(-0.4 * (self.mag - 10)) / 2.83

        def pixels(self):
            """Relative pixel rates: one PSF-core pixel, the rest 1 % of it."""
            return np.r_[1., np.full(499, 0.01)]

        def get_projected_scene(self, in_photons=True):
            return 2 * self.rate() * self.pixels()[:, None, None]      # ph/s/voxel, 2 px per voxel

        def get_snr(self):
            det = self.detector
            charge = self.rate() * self.pixels() * det.integration_time * det.gain
            det.nsaturated_detpx = int(np.count_nonzero(charge > det.saturation))
            result = np.full(500, 100. * 10**(-0.2 * (self.mag-10)))
            result[-1] = np.nan
            return result

        def get_lbda(self):
            return np.geomspace(4000., 17000., 500)

    class PointSource:
        @classmethod
        def from_spectrum(cls, wl, flux, **kw):
            return SimpleNamespace(wl=wl, flux=flux, **kw)

    scene = ModuleType('slicersim.scene')
    pointsource = ModuleType('slicersim.scene.pointsource')
    pointsource.PointSource = PointSource
    lazuli.LazuliTarget = Target
    monkeypatch.setitem(sys.modules, 'slicersim', package)
    monkeypatch.setitem(sys.modules, 'slicersim.lazuli', lazuli)
    monkeypatch.setitem(sys.modules, 'slicersim.scene', scene)
    monkeypatch.setitem(sys.modules, 'slicersim.scene.pointsource', pointsource)
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


def test_saturation_is_tested_at_the_last_read(instrument):
    # (8,2,0): charge 14 at slicersim's (n-1)-group test, 16 at the last read
    Target, _, _ = instrument
    Target.saturation = 15.
    _, _, fixed = run(nmd=(8, 2, 0))
    assert fixed['nsaturated_detpx_slicersim'] == 0
    assert fixed['nsaturated_detpx'] == 1 and fixed['fully_desaturated'] is False
    _, _, auto = run()
    assert auto['nmd_used'] == [4, 2, 0] and auto['fully_desaturated']


def test_subframe_rung_after_the_floor_unless_tframe_is_fixed(instrument):
    # mag 7: (2,1,0) holds 31.7 > 10 at 2.83 s, 7.8 at the 0.7 s subframe
    _, _, auto = run(mag=7.)
    assert auto['nmd_used'] == [2, 1, 0] and auto['tframe_s'] == 0.7
    assert auto['fully_desaturated'] and auto['readout_mode'] == 'auto'
    _, _, fixed_tframe = run(mag=7., tframe=2.83)
    assert fixed_tframe['tframe_s'] == 2.83 and fixed_tframe['nsaturated_detpx'] == 1
    _, _, hopeless = run(mag=5.)       # 2 * 100 * 0.7 / 2.83 = 49 > 10: still reported
    assert hopeless['tframe_s'] == 0.7 and hopeless['fully_desaturated'] is False


def test_t_obs_includes_the_reset_frame(instrument):
    Target, _, _ = instrument
    _, e1, i1 = run()                                   # (4,2,0): 8 frames of 2.83 s
    assert i1['nframes'] == 8 and i1['t_int_s'] == pytest.approx(8 * 2.83)
    assert i1['t_obs_s'] == pytest.approx(9 * 2.83) and i1['n_reset_frames'] == 1
    _, e0, i0 = run(n_reset_frames=0)
    assert i0['t_obs_s'] == i0['t_int_s']
    np.testing.assert_allclose(e1 / e0, np.sqrt(9 / 8))
    Target.nramps = 3
    _, e3, i3 = run(use_cache=False)
    assert i3['t_int_s'] == pytest.approx(3 * i1['t_int_s'])
    assert i3['t_obs_s'] == pytest.approx(3 * i1['t_obs_s'])
    with pytest.raises(ValueError, match='n_reset_frames'):
        run(n_reset_frames=-1)


def test_pixel_average_spreads_a_narrow_line_over_its_pixel():
    centres = np.geomspace(4000., 17000., 525)
    wl = np.arange(3000., 20000., 0.5)
    flux = np.ones_like(wl)
    k = 300
    width = 4.
    line = (wl > centres[k] - width / 2) & (wl < centres[k] + width / 2)
    flux[line] = 11.                                    # top-hat 4 A wide, 10 above continuum
    wl_px, f_px = slicer.pixel_average_sed(wl, flux, centres)
    pixel = np.sqrt(centres[k] * centres[k + 1]) - np.sqrt(centres[k] * centres[k - 1])
    assert width < pixel
    excess = np.sum(0.5 * ((flux - 1)[1:] + (flux - 1)[:-1]) * np.diff(wl))   # line integral
    sampled = np.interp(centres, wl_px, f_px)          # what slicersim does
    assert sampled[k] == pytest.approx(1 + excess / pixel, rel=1e-6)
    assert 2. < sampled[k] < 3.
    assert np.allclose(np.delete(sampled, k), 1.)
    # the grid continues geometrically to cover the SED (band normalization), with smooth spacing
    assert wl_px[0] < 3000. * 1.006 and wl_px[-1] > 20000. / 1.006    # within 2 pixels of the SED ends
    assert np.allclose(np.diff(np.log(wl_px)), np.log(centres[1] / centres[0]), rtol=1e-3)
    # a SED coarser than a pixel is unchanged (to the pixel containing its kink)
    coarse_wl, coarse_f = np.array([3000., 8000., 20000.]), np.array([1., 3., 2.])
    wl_px, f_px = slicer.pixel_average_sed(coarse_wl, coarse_f, centres)
    np.testing.assert_allclose(np.interp(centres, wl_px, f_px),
                               np.interp(centres, coarse_wl, coarse_f), rtol=1e-3)
    # pixels the SED does not fully cover are dropped, not extrapolated
    wl_px, f_px = slicer.pixel_average_sed(wl[wl >= centres[10]], flux[wl >= centres[10]], centres)
    assert wl_px[0] == centres[11]


def test_default_setup_pins_the_narrow_field(instrument):
    # slicersim >= 1.3.3 no longer rescales a fresh LazuliTarget to a field
    # (half the throughput, 1.4x the noise); the live path must not depend on
    # that default, and the noise tables were built in the narrow field.
    Target, _, _ = instrument
    _, _, info = run()
    assert info['spectrograph_mode'] == 'narrow'
    assert Target.last.mode == 'narrow'
