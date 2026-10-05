"""Live slicersim noise: single-transit depth uncertainty per IFS channel.

slicersim and sncosmo are imported inside the functions so that
`import terminator` stays numpy-only. slicersim normalizes magnitudes as AB
(scene/pointsource.py hard-codes bandmag(band, "ab")); archive magnitudes are
Vega, so the default path converts with sncosmo's own band definitions.
"""
import warnings

import numpy as np

BANDS = {"J": "2massj", "H": "2massh", "Ks": "2massks", "V": "bessellv"}
ANGSTROM_PER_UM = 1.0e4


def _sncosmo_band(band):
    try:
        return BANDS[band]
    except KeyError:
        raise ValueError(
            f"band must be one of {sorted(BANDS)}, got {band!r}") from None


def vega_to_ab_offset(band):
    """m_AB - m_Vega for `band`, from sncosmo (independent of the spectrum)."""
    import sncosmo
    b = _sncosmo_band(band)
    bp = sncosmo.get_bandpass(b)
    wave = np.linspace(bp.minwave() * 0.9, bp.maxwave() * 1.1, 2000)
    s = sncosmo.Spectrum(wave, np.full_like(wave, 1e-15))
    return float(s.bandmag(b, "ab") - s.bandmag(b, "vega"))


def _desaturate(target):
    """Shrink the MACC ramp (groups, then frames/group) until the point
    source stops saturating, or the 2x1 floor is reached.

    slicersim's default readout (one long ramp regardless of brightness)
    partially saturates the PSF core of bright hosts and silently degrades
    the extracted SNR. `max_group` has no setter, so `nmd` is the only
    knob. Residual saturation at the floor is reported, never hidden.
    """
    sim = target.simulation
    n, m, d = sim.detector.nmd
    min_group = sim.detector.min_group
    snr = np.asarray(sim.get_snr(), dtype=float)
    nsat = sim.detector.nsaturated_detpx or 0
    while nsat > 0 and n > min_group:
        n = max(min_group, n // 2)
        target.change_detector(nmd=(n, m, d))
        snr = np.asarray(sim.get_snr(), dtype=float)
        nsat = sim.detector.nsaturated_detpx or 0
    while nsat > 0 and m > 1:
        m = max(1, m // 2)
        target.change_detector(nmd=(n, m, d))
        snr = np.asarray(sim.get_snr(), dtype=float)
        nsat = sim.detector.nsaturated_detpx or 0
    return snr, (int(n), int(m), int(d)), int(nsat)


def depth_error(wl_A, f_lambda, *, mag, band="J", mag_system="vega", t14_s,
                n_transits=1, instrument="lazuli_cbe"):
    """Per-channel 1-sigma transit-depth uncertainty for `n_transits` transits.

    depth_err = sqrt(2) / (SNR * sqrt(t14 / t_obs)) / sqrt(n_transits): equal
    in- and out-of-transit baselines, photon + read noise only. Returns
    (wavelength_um, depth_err, info) with non-finite channels trimmed.
    """
    import slicersim
    from slicersim.lazuli import LazuliTarget

    if n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    if t14_s <= 0:
        raise ValueError("t14_s must be > 0")
    mag_system = str(mag_system).lower()
    if mag_system not in ("vega", "ab"):
        raise ValueError(f"mag_system must be 'vega' or 'ab', got {mag_system!r}")
    sn_band = _sncosmo_band(band)
    offset = vega_to_ab_offset(band) if mag_system == "vega" else 0.0
    mag_ab = float(mag) + offset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        target = LazuliTarget(np.asarray(wl_A, float), np.asarray(f_lambda, float),
                              mag=mag_ab, band=sn_band, instrument=instrument)
        sim = target.simulation
        default_nmd = tuple(int(x) for x in sim.detector.nmd)
        snr, nmd_used, nsat = _desaturate(target)
        wl_um = np.asarray(sim.get_lbda(), float) / ANGSTROM_PER_UM
        t_obs = float(sim.observing_time)
        nramps = sim.extraction.get("nramps")

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_in = snr * np.sqrt(t14_s / t_obs)
        err = np.sqrt(2.0) / snr_in / np.sqrt(n_transits)
    good = np.isfinite(err) & (snr_in > 0)
    info = {"slicersim_version": getattr(slicersim, "__version__", "unknown"),
            "instrument": instrument, "band": band, "sncosmo_band": sn_band,
            "mag_input": float(mag), "mag_system": mag_system,
            "vega_ab_offset": offset, "mag_ab": mag_ab,
            "t14_s": float(t14_s), "t_obs_s": t_obs,
            "default_nmd": list(default_nmd), "nmd_used": list(nmd_used),
            "nramps": nramps,
            "nsaturated_detpx": nsat, "fully_desaturated": nsat == 0,
            "trimmed_channels": int((~good).sum()),
            "assumption": "equal out-of-transit baseline; SNR ~ sqrt(t); "
                          "readout tuned to remove point-source saturation"}
    return wl_um[good], err[good], info
