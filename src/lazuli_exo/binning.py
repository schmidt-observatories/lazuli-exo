"""Constant-R wavelength binning for model spectra and noise channels."""
import warnings

import numpy as np

# slicersim config/lazuli_cbe.toml: dispersion_resolution = 2.3 px per
# resolution element; the LSF is a Gaussian of sigma = 2.3/2 px, constant in
# pixel space (spectrograph.py:1264-1304). Future noise tables carry these in
# meta as lsf_sigma_px / resolution_px.
RESOLUTION_PX = 2.3
LSF_SIGMA_PX = RESOLUTION_PX / 2


def constant_R_edges(wl_min, wl_max, R):
    """Wavelength bin edges for constant resolving power R.

    The grid keeps only complete bins, so coverage may stop up to one bin
    short of wl_max (the last partial bin is dropped).
    """
    step = 1.0 + 1.0 / float(R)
    n = int(np.floor(np.log(wl_max / wl_min) / np.log(step)))
    if n < 1:
        raise ValueError(f"R={R} gives no complete bin in [{wl_min}, {wl_max}]")
    return wl_min * step ** np.arange(n + 1)


def bin_centers(edges):
    edges = np.asarray(edges, dtype=float)
    return np.sqrt(edges[:-1] * edges[1:])


def _bin_indices(wl, edges, what):
    idx = np.digitize(wl, edges) - 1
    inside = (idx >= 0) & (idx < len(edges) - 1)
    counts = np.bincount(idx[inside], minlength=len(edges) - 1)
    if (counts == 0).any():
        empty = int(np.argmax(counts == 0))
        raise ValueError(
            f"Bin {empty} ([{edges[empty]:.4f}, {edges[empty + 1]:.4f}] um) "
            f"contains no {what} samples; requested R exceeds the native "
            "noise sampling" if what == "noise" else
            f"Bin {empty} ([{edges[empty]:.4f}, {edges[empty + 1]:.4f}] um) "
            "contains no model samples; use a coarser R")
    return idx, inside, counts


def bin_model(wl, depth, edges):
    wl, depth = np.asarray(wl, float), np.asarray(depth, float)
    idx, inside, counts = _bin_indices(wl, edges, "model")
    sums = np.bincount(idx[inside], weights=depth[inside],
                       minlength=len(edges) - 1)
    return sums / counts


def bin_sigma(wl, sigma, edges):
    wl, sigma = np.asarray(wl, float), np.asarray(sigma, float)
    idx, inside, _ = _bin_indices(wl, edges, "noise")
    inv_var = np.bincount(idx[inside], weights=1.0 / sigma[inside] ** 2,
                          minlength=len(edges) - 1)
    return 1.0 / np.sqrt(inv_var)


def channel_edges(wl_chan):
    """Pixel edges from channel centres: geometric midpoints, ends extended."""
    wl = np.asarray(wl_chan, float)
    mid = np.sqrt(wl[:-1] * wl[1:])
    return np.concatenate([[wl[0] ** 2 / mid[0]], mid, [wl[-1] ** 2 / mid[-1]]])


def R_instr(wl_chan, resolution_px=RESOLUTION_PX):
    """Instrument resolving power per channel, lambda / (resolution_px * dlambda_px)."""
    wl = np.asarray(wl_chan, float)
    return wl / (resolution_px * np.diff(channel_edges(wl)))


def snap_edges_to_channels(edges, wl_chan):
    """Move each bin edge to the nearest channel edge (bins = whole pixels)."""
    ce = channel_edges(wl_chan)
    out = ce[np.abs(np.asarray(edges, float)[:, None] - ce[None, :]).argmin(axis=1)]
    if (np.diff(out) <= 0).any():
        raise ValueError(
            "requested R exceeds the native noise sampling (bins narrower "
            f"than one channel; R_px >= {np.min(R_instr(wl_chan, 1.0)):.0f})")
    return out


def instrument_response(wl_model, depth_model, wl_chan, lsf_sigma_px=LSF_SIGMA_PX,
                        f_phot=None, oversample=100):
    """Per-channel transit depth seen through the spectrograph.

    Channels are contiguous pixels. The model and the stellar photon flux
    `f_phot(lambda_um)` (per unit wavelength; uniform if None) are put on a
    fine grid uniform in pixel coordinate, F*(1-d) and F are convolved with
    the Gaussian LSF (sigma in pixels) and integrated per pixel; the depth is
    1 - sum_in / sum_out. The absolute scale of f_phot cancels.
    """
    wl_model, depth_model = np.asarray(wl_model, float), np.asarray(depth_model, float)
    n = len(wl_chan)
    edges = channel_edges(wl_chan)
    dx = 1.0 / oversample
    x = np.arange(n * oversample) * dx - 0.5 + dx / 2      # pixel i spans [i-0.5, i+0.5)
    pix = np.arange(n).repeat(oversample)
    lam = np.interp(x, np.arange(n + 1) - 0.5, edges)
    w = np.diff(edges)[pix]                                 # dlambda/dx per fine sample
    if f_phot is not None:
        w = w * f_phot(lam)
    d = np.interp(lam, wl_model, depth_model)
    half = int(np.ceil(5 * lsf_sigma_px * oversample))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / lsf_sigma_px) ** 2)
    k /= k.sum()
    s_out = np.convolve(w, k, mode="same")
    s_in = np.convolve(w * (1.0 - d), k, mode="same")
    return 1.0 - (np.bincount(pix, weights=s_in, minlength=n)
                  / np.bincount(pix, weights=s_out, minlength=n))


def bin_weighted(wl, depth, sigma, edges):
    """Inverse-variance-weighted mean of per-channel depths: the same
    combination whose uncertainty bin_sigma returns."""
    wl, depth, sigma = (np.asarray(a, float) for a in (wl, depth, sigma))
    idx, inside, _ = _bin_indices(wl, edges, "noise")
    w = 1.0 / sigma ** 2
    n = len(edges) - 1
    return (np.bincount(idx[inside], weights=(w * depth)[inside], minlength=n)
            / np.bincount(idx[inside], weights=w[inside], minlength=n))


def bin_observation(wl_model, depth_model, wl_chan, sigma_chan, R, f_phot=None,
                    noise_meta=None):
    """Constant-R bins snapped to channel edges, LSF/photon-weighted per-channel
    depths combined with the 1/sigma^2 weights, and the matching sigma.

    Returns (edges, depth, sigma, info). Warns (does not raise) when R exceeds
    the instrument resolving power over the binned range, so oversampled
    display is still possible; bins narrower than a resolution element are
    then correlated by the LSF.
    """
    meta = noise_meta or {}
    lsf = float(meta.get("lsf_sigma_px", LSF_SIGMA_PX))
    res = float(meta.get("resolution_px", RESOLUTION_PX))
    wl_chan, sigma_chan = np.asarray(wl_chan, float), np.asarray(sigma_chan, float)
    edges = snap_edges_to_channels(
        constant_R_edges(wl_chan.min(), wl_chan.max(), R), wl_chan)
    r_inst = R_instr(wl_chan, res)[(wl_chan >= edges[0]) & (wl_chan <= edges[-1])]
    if R > r_inst.min():
        warnings.warn(
            f"requested R={R} exceeds the instrument resolving power "
            f"(R_instr = {r_inst.min():.0f}-{r_inst.max():.0f} over the binned "
            "range); bins narrower than a resolution element are correlated "
            "by the LSF", UserWarning, stacklevel=3)
    depth_chan = instrument_response(wl_model, depth_model, wl_chan, lsf, f_phot)
    info = {"lsf_sigma_px": lsf, "resolution_px": res,
            "R_instr_min": float(r_inst.min()), "R_instr_max": float(r_inst.max()),
            "photon_weighting": "uniform", "edges_snapped": True}
    return (edges, bin_weighted(wl_chan, depth_chan, sigma_chan, edges),
            bin_sigma(wl_chan, sigma_chan, edges), info)
