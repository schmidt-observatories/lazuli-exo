import numpy as np
import pytest

from slixer import binning


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


def test_empty_bin_raises():
    wl = np.array([1.0, 2.0])
    edges = np.array([1.0, 1.2, 1.4, 2.1])
    with pytest.raises(ValueError, match="native noise sampling"):
        binning.bin_sigma(wl, np.ones(2), edges)
    with pytest.raises(ValueError):
        binning.bin_model(wl, np.ones(2), edges)
