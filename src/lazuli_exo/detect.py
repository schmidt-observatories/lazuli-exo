"""Spectral-template S/N of a noiseless binned model spectrum.

The statistic is a matched-template amplitude S/N under a diagonal
covariance: snr = sqrt(sum w (d - dbar)^2), w = 1/sigma_d^2, dbar the
weighted mean depth. It is the expected S/N of fitting the (noiseless)
model spectrum as a template against the best-fit flat line, assuming
independent Gaussian errors and no nuisance parameters beyond the mean.
It is NOT a calibrated Gaussian-equivalent detection significance: there
is no defined null/alternative likelihood ratio, no covariance, and no
false-positive calibration.

Per-molecule values under "band_snr" evaluate the same sum restricted to
that molecule's wavelength band mask (relative to the same global dbar).
They are band-restricted template norms, not molecule detections: they
include absorption from every other species in the mask, haze/cloud/
continuum structure, and the offset of the masked region from the global
mean; bands of different molecules overlap (H2O and CH4 near 1.15 um).
The rigorous upgrade path is a likelihood ratio against leave-one-species-
out model tables. CO and NH3 are present in the models but have no band
masks in the 0.4-1.7 µm range, so they are never reported — their absence
from the output is not a non-detection.
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

    band_snr = {}
    for mol, spans in BAND_MASKS.items():
        m = _mask_array(wavelength, spans)
        if m.any():
            band_snr[mol] = float(np.sqrt(np.sum(chi2[m])))
    return {"template_snr_diagonal": float(np.sqrt(np.sum(chi2))),
            "band_snr": band_snr}
