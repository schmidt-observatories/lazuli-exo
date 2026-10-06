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


SPECTROGRAPH_MODES = ("narrow", "wide")
ZODI_REFERENCE_BAND = "bessellv"
_C_ANGSTROM_PER_S = 2.99792458e18
_AB_ZEROPOINT = 48.60


def normalize_setup(nmd=None, tframe=None, spectrograph_mode=None,
                    guiding_sigma=None, zodi_ab_mag=None):
    """Validate the optional instrument setup and return it as a hashable
    tuple (the native-noise cache key includes it). None means "slicersim's
    default", so an all-None setup reproduces the pre-setup behaviour.

    nmd              (groups, frames, drops) MACC readout; fixed, not desaturated.
    tframe           frame time [s] (e.g. 2.83 full frame, 0.7 rapid subframe).
    spectrograph_mode 'narrow' | 'wide' input field.
    guiding_sigma    pointing jitter [arcsec], >= 0.
    zodi_ab_mag      zodiacal surface brightness [AB mag/arcsec^2, V band].
    """
    if nmd is not None:
        try:
            n, m, d = (int(x) for x in nmd)
        except (TypeError, ValueError):
            raise ValueError(f"nmd must be three integers (groups, frames, drops), got {nmd!r}") from None
        if n < 1 or m < 1 or d < 0:
            raise ValueError(f"nmd needs groups >= 1, frames >= 1, drops >= 0, got {(n, m, d)}")
        nmd = (n, m, d)
    if tframe is not None:
        tframe = float(tframe)
        if not np.isfinite(tframe) or tframe <= 0:
            raise ValueError(f"tframe must be > 0 s, got {tframe!r}")
    if spectrograph_mode is not None:
        spectrograph_mode = str(spectrograph_mode).lower()
        if spectrograph_mode not in SPECTROGRAPH_MODES:
            raise ValueError(f"spectrograph_mode must be one of {SPECTROGRAPH_MODES}, got {spectrograph_mode!r}")
    if guiding_sigma is not None:
        guiding_sigma = float(guiding_sigma)
        if not np.isfinite(guiding_sigma) or guiding_sigma < 0:
            raise ValueError(f"guiding_sigma must be >= 0 arcsec, got {guiding_sigma!r}")
    if zodi_ab_mag is not None:
        zodi_ab_mag = float(zodi_ab_mag)
        if not np.isfinite(zodi_ab_mag):
            raise ValueError(f"zodi_ab_mag must be finite, got {zodi_ab_mag!r}")
    return (nmd, tframe, spectrograph_mode, guiding_sigma, zodi_ab_mag)


@lru_cache(maxsize=4)
def _zodi_ab_mag_at_unit_scale(band=ZODI_REFERENCE_BAND):
    """AB surface brightness (mag/arcsec^2) of slicersim's zodi model at scale=1,
    computed from the installed model so the conversion follows its updates."""
    import sncosmo
    from slicersim.scene.background import zodiacal_spectrum
    bp = sncosmo.get_bandpass(band)
    lbda = bp.wave
    flam = zodiacal_spectrum(lbda, scale=1.0)              # erg/s/cm^2/A/arcsec^2
    fnu = flam * lbda ** 2 / _C_ANGSTROM_PER_S              # erg/s/cm^2/Hz/arcsec^2
    trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz
    fnu_avg = trapz(fnu * bp.trans * lbda, lbda) / trapz(bp.trans * lbda, lbda)
    return float(-2.5 * np.log10(fnu_avg) - _AB_ZEROPOINT)


def zodi_scale_from_ab_mag(ab_mag, band=ZODI_REFERENCE_BAND):
    """slicersim's multiplicative zodi ``scale`` for an AB surface brightness;
    brighter sky (smaller magnitude) -> larger scale, the reference maps to 1."""
    return float(10 ** (-0.4 * (float(ab_mag) - _zodi_ab_mag_at_unit_scale(band))))


def _apply_setup(target, setup):
    """Apply a normalized setup to a built target, in slicersim's required order
    (field before per-run overrides: change_spectrograph reloads the field)."""
    nmd, tframe, spectrograph_mode, guiding_sigma, zodi_ab_mag = setup
    sim = target.simulation
    if spectrograph_mode is not None:
        target.change_spectrograph(spectrograph_mode)
    if tframe is not None:
        sim.update(tframe=tframe)
    if guiding_sigma is not None:
        sim.update(guiding_sigma=guiding_sigma)
    if zodi_ab_mag is not None:
        sim.scene.background.update(scale=zodi_scale_from_ab_mag(zodi_ab_mag))


def _fixed_readout(target, nmd):
    """Apply the caller's readout as is and report its saturation; the
    counterpart of _desaturate for a readout the observer chose."""
    sim = target.simulation
    target.change_detector(nmd=tuple(nmd))
    snr = np.asarray(sim.get_snr(), dtype=float)
    nsat = sim.detector.nsaturated_detpx or 0
    return snr, tuple(int(x) for x in nmd), int(nsat)


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
                          slicersim_version, target_class, setup=(None,) * 5):
    """Instrument result before duration and transit-count scaling.

    ``setup`` is a normalize_setup() tuple; its nmd, when given, is used as is
    (readout_mode 'fixed'), otherwise the ramp is desaturated ('auto')."""
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
            _apply_setup(target, setup)
        except Exception as e:      # missing/broken instrument config -> user-facing error
            raise ValueError(f"slicersim could not build instrument {instrument!r}: "
                             f"{type(e).__name__}: {e}") from e
        default_nmd = tuple(int(x) for x in sim.detector.nmd)
        nmd_fixed = setup[0]
        if nmd_fixed is not None:
            snr, nmd_used, nsat = _fixed_readout(target, nmd_fixed)
        else:
            snr, nmd_used, nsat = _desaturate(target)
        wl_um = np.asarray(sim.get_lbda(), float) / ANGSTROM_PER_UM
        t_obs = float(sim.observing_time)
        nramps = sim.extraction.get("nramps")
        tframe_s = _get_parameter(sim, "tframe")

    info = {"slicersim_version": slicersim_version, "instrument": instrument,
            "t_obs_s": t_obs, "default_nmd": list(default_nmd),
            "nmd_used": list(nmd_used), "nramps": nramps,
            "readout_mode": "fixed" if nmd_fixed is not None else "auto",
            "tframe_s": tframe_s, "spectrograph_mode": setup[2],
            "guiding_sigma": setup[3], "zodi_ab_mag": setup[4],
            "nsaturated_detpx": nsat, "fully_desaturated": nsat == 0}
    # Own the cached arrays; upstream objects may otherwise retain views.
    return wl_um.copy(), snr.copy(), info


def _get_parameter(sim, name):
    """sim.get_parameter(name) as float, or None when the simulation (or a test
    double) does not expose it."""
    try:
        return float(sim.get_parameter(name))
    except Exception:
        return None


_cached_native_noise = lru_cache(maxsize=32)(_compute_native_noise)


def clear_noise_cache():
    """Discard this process's instrument results and magnitude zero points.

    Call after editing instrument/calibration files or replacing registered
    bandpasses in a running process. Restarting the process also clears it.
    """
    _cached_native_noise.cache_clear()
    vega_to_ab_offset.cache_clear()


def depth_error(wl_A, f_lambda, *, mag, band="J", mag_system="vega", t14_s,
                n_transits=1, instrument="lazuli_cbe", use_cache=True,
                nmd=None, tframe=None, spectrograph_mode=None,
                guiding_sigma=None, zodi_ab_mag=None):
    """Per-channel 1-sigma transit-depth uncertainty for `n_transits` transits.

    depth_err = sqrt(2) / (SNR * sqrt(t14 / t_obs)) / sqrt(n_transits): equal
    in- and out-of-transit baselines, photon + read noise only. Returns
    (wavelength_um, depth_err, info) with non-finite channels trimmed.

    Instrument setup (all optional, None = slicersim's default; see
    normalize_setup): ``nmd`` fixes the MACC readout and reports its saturation
    instead of shortening the ramp (info["readout_mode"] == "fixed"); ``tframe``
    [s], ``spectrograph_mode`` ('narrow'|'wide'), ``guiding_sigma`` [arcsec] and
    ``zodi_ab_mag`` [AB mag/arcsec^2] are applied before the readout. An ETC
    that shows this result next to its own single-exposure result should pass
    the same setup so the two describe one observation.

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
    setup = normalize_setup(nmd=nmd, tframe=tframe, spectrograph_mode=spectrograph_mode,
                            guiding_sigma=guiding_sigma, zodi_ab_mag=zodi_ab_mag)
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
        getattr(slicersim, "__version__", "unknown"), LazuliTarget, setup)
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
                               + ("readout fixed by the caller (saturation reported, not removed)"
                                  if setup[0] is not None else
                                  "readout tuned to remove point-source saturation")})
    return wl_um[good], err[good], info
