"""Live slicersim noise: single-transit depth uncertainty per IFS channel.

slicersim and sncosmo are imported inside the functions so that
`import lazuli_exo` stays numpy-only. slicersim normalizes magnitudes as AB
(scene/pointsource.py hard-codes bandmag(band, "ab")); archive magnitudes are
Vega, so the default path converts with sncosmo's own band definitions.
"""
from copy import deepcopy
from functools import lru_cache
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


@lru_cache(maxsize=32)
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


def _compute_native_noise(wl_bytes, flux_bytes, mag_ab, sn_band, instrument,
                          slicersim_version, target_class):
    """Instrument result before duration and transit-count scaling."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            target = target_class(np.frombuffer(wl_bytes, dtype=np.float64).copy(),
                                  np.frombuffer(flux_bytes, dtype=np.float64).copy(),
                                  mag=mag_ab, band=sn_band, instrument="lazuli_cbe")
            if instrument != "lazuli_cbe":
                # the eol/bol tomls are partial overlays on the cbe config
                target.change_configuration(instrument.removeprefix("lazuli_"))
            sim = target.simulation
        except Exception as e:      # missing/broken instrument config -> user-facing error
            raise ValueError(f"slicersim could not build instrument {instrument!r}: "
                             f"{type(e).__name__}: {e}") from e
        default_nmd = tuple(int(x) for x in sim.detector.nmd)
        snr, nmd_used, nsat = _desaturate(target)
        wl_um = np.asarray(sim.get_lbda(), float) / ANGSTROM_PER_UM
        t_obs = float(sim.observing_time)
        nramps = sim.extraction.get("nramps")

    info = {"slicersim_version": slicersim_version, "instrument": instrument,
            "t_obs_s": t_obs, "default_nmd": list(default_nmd),
            "nmd_used": list(nmd_used), "nramps": nramps,
            "nsaturated_detpx": nsat, "fully_desaturated": nsat == 0}
    # Own the cached arrays; upstream objects may otherwise retain views.
    return wl_um.copy(), snr.copy(), info


_cached_native_noise = lru_cache(maxsize=32)(_compute_native_noise)


def clear_noise_cache():
    """Discard this process's instrument results and magnitude zero points.

    Call after editing instrument/calibration files or replacing registered
    bandpasses in a running process. Restarting the process also clears it.
    """
    _cached_native_noise.cache_clear()
    vega_to_ab_offset.cache_clear()


def depth_error(wl_A, f_lambda, *, mag, band="J", mag_system="vega", t14_s,
                n_transits=1, instrument="lazuli_cbe", use_cache=True):
    """Per-channel 1-sigma transit-depth uncertainty for `n_transits` transits.

    depth_err = sqrt(2) / (SNR * sqrt(t14 / t_obs)) / sqrt(n_transits): equal
    in- and out-of-transit baselines, photon + read noise only. Returns
    (wavelength_um, depth_err, info) with non-finite channels trimmed.

    By default up to 32 native instrument results are cached per process,
    keyed by the exact SED contents, AB magnitude, band, instrument and
    slicersim version/target implementation. Duration and transit count are
    scaled per call; callers receive independent arrays and metadata.
    Instrument/calibration files and registered bandpasses are assumed fixed
    during a process: call clear_noise_cache() after changing them, or use
    use_cache=False to bypass lookup and storage for a reference calculation.
    """
    import slicersim
    from slicersim.lazuli import LazuliTarget

    if not np.isfinite(n_transits) or n_transits < 1:
        raise ValueError("n_transits must be >= 1")
    if not np.isfinite(t14_s) or t14_s <= 0:
        raise ValueError("t14_s must be > 0")
    mag_system = str(mag_system).lower()
    if mag_system not in ("vega", "ab"):
        raise ValueError(f"mag_system must be 'vega' or 'ab', got {mag_system!r}")
    sn_band = _sncosmo_band(band)
    offset = 0.0
    if mag_system == "vega":
        convert = vega_to_ab_offset if use_cache else vega_to_ab_offset.__wrapped__
        offset = convert(band)
    mag_ab = float(mag) + offset

    wl = np.asarray(wl_A, dtype=np.float64)
    flux = np.asarray(f_lambda, dtype=np.float64)
    if (wl.ndim != 1 or flux.shape != wl.shape or wl.size < 2
            or not np.isfinite(wl).all() or not np.isfinite(flux).all()
            or np.any(wl <= 0) or np.any(np.diff(wl) <= 0)):
        raise ValueError("SED must have matching finite 1D arrays and increasing positive wavelengths")
    if not np.isfinite(mag_ab):
        raise ValueError("mag must be finite")
    compute = _cached_native_noise if use_cache else _compute_native_noise
    wl_um, snr, native_info = compute(
        wl.tobytes(), flux.tobytes(), mag_ab, sn_band, instrument,
        getattr(slicersim, "__version__", "unknown"), LazuliTarget)
    info = deepcopy(native_info)
    t_obs = info["t_obs_s"]

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_in = snr * np.sqrt(t14_s / t_obs)
        err = np.sqrt(2.0) / snr_in / np.sqrt(n_transits)
    good = np.isfinite(err) & (snr_in > 0)
    info.update({"band": band, "sncosmo_band": sn_band,
                 "mag_input": float(mag), "mag_system": mag_system,
                 "vega_ab_offset": offset, "mag_ab": mag_ab,
                 "t14_s": float(t14_s),
                 "trimmed_channels": int((~good).sum()),
                 "assumption": "equal out-of-transit baseline; SNR ~ sqrt(t); "
                               "readout tuned to remove point-source saturation"})
    return wl_um[good], err[good], info
