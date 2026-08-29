"""Predicted detection significances from a noiseless binned model spectrum.

The statistic is the expected significance of rejecting the best-fit flat
line: sigma = sqrt(sum w (d - dbar)^2), w = 1/sigma_d^2, dbar the weighted
mean depth. Per-molecule values evaluate the same sum restricted to that
molecule's wavelength band mask (relative to the same global dbar). Bands of
different molecules overlap (H2O and CH4 near 1.15 um), so per-molecule
values are approximate; per-molecule leave-one-out model tables are the
rigorous upgrade path.
"""
import numpy as np

BAND_MASKS = {
    "Na":  [(0.575, 0.605)],
    "K":   [(0.755, 0.785)],
    "H2O": [(0.90, 1.00), (1.10, 1.20), (1.30, 1.55)],
    "CH4": [(0.86, 0.91), (1.10, 1.20), (1.60, 1.70)],
    "CO2": [(1.42, 1.46), (1.56, 1.62)],
}


def _mask_array(wavelength, spans):
    m = np.zeros(wavelength.shape, dtype=bool)
    for lo, hi in spans:
        m |= (wavelength >= lo) & (wavelength <= hi)
    return m


def detect_features(wavelength, depth, depth_err):
    wavelength = np.asarray(wavelength, float)
    depth = np.asarray(depth, float)
    w = 1.0 / np.asarray(depth_err, float) ** 2
    dbar = np.sum(w * depth) / np.sum(w)
    chi2 = w * (depth - dbar) ** 2

    out = {"atmosphere": float(np.sqrt(np.sum(chi2)))}
    for mol, spans in BAND_MASKS.items():
        m = _mask_array(wavelength, spans)
        if m.any():
            out[mol] = float(np.sqrt(np.sum(chi2[m])))
    return out
