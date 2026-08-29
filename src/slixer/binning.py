"""Constant-R wavelength binning for model spectra and noise channels."""
import numpy as np


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
