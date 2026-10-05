import numpy as np
import pytest

from terminator.detect import BAND_MASKS, detect_features


def test_flat_spectrum_zero_snr():
    wl = np.linspace(0.4, 1.7, 100)
    out = detect_features(wl, np.full(100, 0.01), np.full(100, 1e-4))
    assert out["template_snr_diagonal"] == pytest.approx(0.0, abs=1e-9)


def test_single_bump_analytic():
    # one bin offset by k*sigma among N equal bins:
    # snr = k * sqrt(1 - 1/N)
    N, k = 100, 5.0
    wl = np.linspace(0.4, 1.7, N)
    sigma = np.full(N, 1e-4)
    depth = np.full(N, 0.01)
    depth[50] += k * sigma[50]
    out = detect_features(wl, depth, sigma)
    assert out["template_snr_diagonal"] == pytest.approx(
        k * np.sqrt(1 - 1 / N), rel=1e-9)


def test_band_snr_is_nested_not_sigma():
    """Per-molecule values live under band_snr — they are band-restricted
    template norms, not molecule detection significances."""
    wl = np.linspace(0.4, 1.7, 100)
    out = detect_features(wl, np.full(100, 0.01), np.full(100, 1e-4))
    assert set(out) == {"template_snr_diagonal", "band_snr"}
    assert set(out["band_snr"]) <= set(BAND_MASKS)


def test_band_isolation():
    # bump inside the Na band only -> Na band S/N ~ overall, K ~ small
    wl = np.linspace(0.4, 1.7, 1000)
    sigma = np.full_like(wl, 1e-4)
    depth = np.full_like(wl, 0.01)
    in_na = (wl > 0.575) & (wl < 0.605)
    depth[in_na] += 5e-4
    out = detect_features(wl, depth, sigma)
    bands = out["band_snr"]
    assert bands["Na"] > 0.9 * out["template_snr_diagonal"]
    assert bands["K"] < 0.2 * bands["Na"]


def test_masks_within_ifs_range():
    for mol, spans in BAND_MASKS.items():
        for lo, hi in spans:
            assert lo < hi, mol
            assert lo >= 0.4 or hi > 0.4, mol  # at least partially usable
