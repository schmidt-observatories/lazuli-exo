import numpy as np
import pytest

from lazuli_exo import binning


def test_constant_R_edges_spacing():
    edges = binning.constant_R_edges(0.4, 1.7, 50)
    ratios = edges[1:] / edges[:-1]
    np.testing.assert_allclose(ratios, ratios[0])
    assert edges[0] == pytest.approx(0.4)
    assert edges[-1] <= 1.7 + 1e-12
    # Δλ/λ ≈ 1/R at each bin
    assert ratios[0] == pytest.approx(1 + 1.0 / 50, rel=1e-6)


def test_bin_model_conserves_constant():
    wl = np.geomspace(0.4, 1.7, 5000)
    depth = np.full_like(wl, 0.0146)
    edges = binning.constant_R_edges(0.4, 1.7, 100)
    binned = binning.bin_model(wl, depth, edges)
    np.testing.assert_allclose(binned, 0.0146)


def test_bin_model_mean_of_linear():
    # binning a linear function gives the mean of the samples in each bin
    wl = np.linspace(1.0, 2.0, 10001)
    edges = np.array([1.0, 1.5, 2.0])
    binned = binning.bin_model(wl, wl.copy(), edges)
    np.testing.assert_allclose(binned, [1.25, 1.75], rtol=1e-3)


def test_bin_sigma_inverse_variance():
    wl = np.linspace(1.0, 2.0, 4)
    sigma = np.array([2.0, 2.0, 2.0, 2.0])
    edges = np.array([0.9, 2.1])  # one bin catching all 4 channels
    out = binning.bin_sigma(wl, sigma, edges)
    np.testing.assert_allclose(out, [1.0])  # 2/sqrt(4)


# --- instrument response (GH #5) -------------------------------------------

WL_CH = np.geomspace(0.4, 1.7, 500)        # R_px ~ 345, R_instr ~ 150


def test_instrument_response_spreads_unresolved_line():
    # delta-like line (sigma = 0.02 px) centred on channel 250: area conserved,
    # peak = fraction of a sigma=1.15 px Gaussian inside +-0.5 px
    from math import erf, sqrt
    wl = np.geomspace(0.39, 1.71, 400000)
    i = 250
    dpx = WL_CH[i] * np.log(WL_CH[1] / WL_CH[0])       # channel width
    line = 1e-3 * np.exp(-0.5 * ((wl - WL_CH[i]) / (0.02 * dpx)) ** 2)
    d = binning.instrument_response(wl, line, WL_CH)
    area_model = np.trapezoid(line, wl)
    area_chan = np.sum(d * np.diff(binning.channel_edges(WL_CH)))
    assert abs(area_chan / area_model - 1) < 1e-6
    assert d.argmax() == i
    expected_peak = area_model / dpx * erf(0.5 / (1.15 * sqrt(2)))
    assert d[i] == pytest.approx(expected_peak, rel=2e-3)


def test_instrument_response_constant_is_exact():
    wl = np.geomspace(0.39, 1.71, 20000)
    d = binning.instrument_response(wl, np.full_like(wl, 0.0146), WL_CH,
                                    f_phot=lambda lam: np.exp(-3 * lam) * (1 + np.sin(50 * lam)))
    np.testing.assert_allclose(d, 0.0146, rtol=1e-12)


def test_instrument_response_photon_weighting_direction():
    # depth ramp increasing with wavelength; rising stellar flux within the
    # pixel pulls the channel depth up, falling flux pulls it down
    wl = np.geomspace(0.39, 1.71, 20000)
    ramp = 0.01 + 0.01 * (wl - 0.4)
    d0 = binning.instrument_response(wl, ramp, WL_CH)
    up = binning.instrument_response(wl, ramp, WL_CH, f_phot=lambda lam: lam ** 40)
    dn = binning.instrument_response(wl, ramp, WL_CH, f_phot=lambda lam: lam ** -40)
    sl = slice(20, -20)
    assert (up[sl] > d0[sl]).all() and (dn[sl] < d0[sl]).all()


def test_snapped_edges_remove_sawtooth():
    edges = binning.snap_edges_to_channels(
        binning.constant_R_edges(WL_CH.min(), WL_CH.max(), 100), WL_CH)
    ce = binning.channel_edges(WL_CH)
    assert np.isin(edges, ce).all()
    counts = np.histogram(WL_CH, edges)[0]
    assert counts.max() - counts.min() <= 1          # 3 or 4 channels per bin, never 2 vs 9
    sig = binning.bin_sigma(WL_CH, np.ones_like(WL_CH), edges)
    assert sig.max() / sig.min() < np.sqrt(4 / 3) + 1e-9
    with pytest.raises(ValueError, match="native noise sampling"):
        binning.snap_edges_to_channels(binning.constant_R_edges(0.4, 1.7, 1000), WL_CH)


def test_bin_weighted_matches_bin_sigma_weights():
    wl = np.array([1.0, 1.1, 1.2])
    sigma = np.array([1.0, 2.0, 2.0])
    edges = np.array([0.9, 1.3])
    out = binning.bin_weighted(wl, np.array([0.0, 1.0, 1.0]), sigma, edges)
    np.testing.assert_allclose(out, [0.5 / 1.5])          # weights 1, 1/4, 1/4


def test_R_instr():
    r = binning.R_instr(WL_CH)
    np.testing.assert_allclose(r, 1 / np.log(WL_CH[1] / WL_CH[0]) / 2.3, rtol=1e-4)


def test_empty_bin_raises():
    wl = np.array([1.0, 2.0])
    edges = np.array([1.0, 1.2, 1.4, 2.1])
    with pytest.raises(ValueError, match="native noise sampling"):
        binning.bin_sigma(wl, np.ones(2), edges)
    with pytest.raises(ValueError):
        binning.bin_model(wl, np.ones(2), edges)
