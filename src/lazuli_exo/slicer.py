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
    """sncosmo bandpass name for `band`: the J/H/Ks/V aliases, or any name sncosmo knows."""
    if band in BANDS:
        return BANDS[band]
    import sncosmo
    try:
        sncosmo.get_bandpass(band)
    except Exception:
        raise ValueError(
            f"band must be one of the aliases {sorted(BANDS)} or an sncosmo bandpass "
            f"name (e.g. '2massj', 'sdssr'); got {band!r}") from None
    return band


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


def _nsat_lastread(sim):
    """Detector pixels whose charge at the LAST read of the ramp exceeds the
    full well.

    slicersim (detector.py estimate_pixel_signal) tests the slope signal
    `rate * integration_time`, with integration_time = (n-1)(m+d) tframe, but
    the last read is at exposure_time = nframes * tframe: a (2, m, 0) ramp is
    tested at half its real charge (#33). Same pixel rate as slicersim: the
    projected cube [ph/s/voxel] split over the anamorphose pixels, times QE,
    plus the effective dark."""
    det, sp = sim.detector, sim.spectrograph
    cube = np.asarray(sim.get_projected_scene(in_photons=True), float)   # (nlbda, y, x), LSF applied
    qe = np.reshape(np.asarray(det.get_qe(lbda=sp.lbda), float), (-1, 1, 1))
    rate = cube / sp._ANAMORPHOSE * qe + det.get_effective_dark()        # e-/s/pixel
    return int(np.count_nonzero(rate * det.exposure_time * det.gain > det.saturation))


def _fixed_readout(target, nmd):
    """Apply the caller's readout as is and report its saturation; the
    counterpart of _desaturate for a readout the observer chose."""
    sim = target.simulation
    target.change_detector(nmd=tuple(nmd))
    snr = np.asarray(sim.get_snr(), dtype=float)
    return snr, tuple(int(x) for x in nmd), _nsat_lastread(sim)


RAPID_TFRAME_S = 0.7      # H4RG rapid subframe tried after the (min_group, 1, d) floor


def _desaturate(target, try_subframe=True):
    """Shrink the MACC ramp (groups, then frames/group, then the frame time)
    until the point source stops saturating at the last read.

    slicersim's default readout (one long ramp regardless of brightness)
    partially saturates the PSF core of bright hosts and silently degrades
    the extracted SNR. `max_group` has no setter, so `nmd` is the only
    knob below the frame time; when the (min_group, 1, d) floor still
    saturates and the caller did not fix tframe, the RAPID_TFRAME_S subframe
    is tried. Residual saturation after that is reported, never hidden.
    """
    sim = target.simulation
    n, m, d = sim.detector.nmd
    min_group = sim.detector.min_group

    def run(nmd):
        target.change_detector(nmd=nmd)
        return np.asarray(sim.get_snr(), dtype=float), _nsat_lastread(sim)

    snr, nsat = np.asarray(sim.get_snr(), dtype=float), _nsat_lastread(sim)
    while nsat > 0 and n > min_group:
        n = max(min_group, n // 2)
        snr, nsat = run((n, m, d))
    while nsat > 0 and m > 1:
        m = max(1, m // 2)
        snr, nsat = run((n, m, d))
    if nsat > 0 and try_subframe and sim.detector.tframe > RAPID_TFRAME_S:
        sim.update(tframe=RAPID_TFRAME_S)
        snr, nsat = run((n, m, d))
    return snr, (int(n), int(m), int(d)), int(nsat)


def pixel_average_sed(wl, flux, centres):
    """Box-average the SED over the instrument pixels so that slicersim's
    point sampling at the pixel centres returns the pixel mean (#37).

    Pixel edges are the geometric midpoints of `centres` (0.1 % of a pixel
    from slicersim's own edges). The channel grid is continued geometrically
    beyond the instrument range to cover the whole SED, so the band used for
    the magnitude normalization stays covered and the grid spacing is smooth
    (sncosmo's bin-edge recovery rejects a spacing jump). The average is
    exact for the piecewise-linear SED, so a SED coarser than a pixel is
    unchanged. Pixels the SED does not fully cover are dropped (slicersim
    then trims those channels). Returns (wavelength, flux) on the centres."""
    wl, flux, c = (np.asarray(a, float) for a in (wl, flux, centres))
    r_lo, r_hi = c[1] / c[0], c[-1] / c[-2]
    n_lo = max(0, int(np.ceil(np.log(c[0] / wl[0]) / np.log(r_lo))))
    n_hi = max(0, int(np.ceil(np.log(wl[-1] / c[-1]) / np.log(r_hi))))
    c = np.concatenate((c[0] * r_lo ** -np.arange(n_lo, 0, -1), c,
                        c[-1] * r_hi ** np.arange(1, n_hi + 1)))
    inner = np.sqrt(c[1:] * c[:-1])
    edges = np.concatenate(([c[0] ** 2 / inner[0]], inner, [c[-1] ** 2 / inner[-1]]))
    grid = np.union1d(wl, edges[(edges >= wl[0]) & (edges <= wl[-1])])
    f = np.interp(grid, wl, flux)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(grid))))
    mean = np.diff(np.interp(edges, grid, cum, left=np.nan, right=np.nan)) / np.diff(edges)
    ok = np.isfinite(mean)
    return c[ok], mean[ok]


def _compute_native_noise(wl_bytes, flux_bytes, mag_ab, sn_band, instrument,
                          slicersim_version, target_class, setup=(None,) * 5):
    """Instrument result before duration and transit-count scaling.

    ``setup`` is a normalize_setup() tuple; its nmd, when given, is used as is
    (readout_mode 'fixed'), otherwise the ramp is desaturated ('auto'). The SED
    is pixel-averaged onto the instrument channels before the noise run."""
    from slicersim.scene.pointsource import PointSource

    wl = np.frombuffer(wl_bytes, dtype=np.float64).copy()
    flux = np.frombuffer(flux_bytes, dtype=np.float64).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            target = target_class(wl, flux, mag=mag_ab, band=sn_band, instrument="lazuli_cbe")
            if instrument != "lazuli_cbe":
                # the eol/bol tomls are partial overlays on the cbe config
                target.change_configuration(instrument.removeprefix("lazuli_"))
            sim = target.simulation
            _apply_setup(target, setup)
        except Exception as e:      # missing/broken instrument config -> user-facing error
            raise ValueError(f"slicersim could not build instrument {instrument!r}: "
                             f"{type(e).__name__}: {e}") from e
        wl_px, flux_px = pixel_average_sed(wl, flux, sim.get_lbda())
        sim.change_pointsource(PointSource.from_spectrum(
            wl_px, flux_px, mag=mag_ab, band=sn_band,
            position=sim.scene.pointsource.position, meta={}))
        default_nmd = tuple(int(x) for x in sim.detector.nmd)
        nmd_fixed = setup[0]
        if nmd_fixed is not None:
            snr, nmd_used, nsat = _fixed_readout(target, nmd_fixed)
        else:
            snr, nmd_used, nsat = _desaturate(target, try_subframe=setup[1] is None)
        nsat_slicersim = int(sim.detector.nsaturated_detpx or 0)
        wl_um = np.asarray(sim.get_lbda(), float) / ANGSTROM_PER_UM
        t_int = float(sim.observing_time)
        nramps = sim.extraction.get("nramps")
        tframe_s = _get_parameter(sim, "tframe")

    info = {"slicersim_version": slicersim_version, "instrument": instrument,
            "t_int_s": t_int, "default_nmd": list(default_nmd),
            "nmd_used": list(nmd_used), "nramps": nramps,
            "readout_mode": "fixed" if nmd_fixed is not None else "auto",
            "tframe_s": tframe_s, "spectrograph_mode": setup[2],
            "guiding_sigma": setup[3], "zodi_ab_mag": setup[4],
            "nsaturated_detpx": nsat, "fully_desaturated": nsat == 0,
            "nsaturated_detpx_slicersim": nsat_slicersim}
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
                guiding_sigma=None, zodi_ab_mag=None, n_reset_frames=1):
    """Per-channel 1-sigma transit-depth uncertainty for `n_transits` transits.

    depth_err = sqrt(2) / (SNR * sqrt(t14 / t_obs)) / sqrt(n_transits): equal
    in- and out-of-transit baselines, photon + read noise only. Returns
    (wavelength_um, depth_err, info) with non-finite channels trimmed.

    The SED is box-averaged over each instrument pixel before slicersim sees
    it (pixel_average_sed), because slicersim point-samples the spectrum at
    the pixel centres; without this the per-channel signal of an M dwarf is
    off by ~14 % rms (#37). Saturation is tested on the charge at the last
    read of the ramp, not on slicersim's (n-1)-group slope signal (#33); the
    auto readout shortens the ramp and then tries the RAPID_TFRAME_S subframe.
    t_obs counts ``n_reset_frames`` extra frame times per ramp (H4RG MACC
    reset overhead): t_obs = (nframes + n_reset_frames) * tframe * nramps,
    while info["t_int_s"] is slicersim's nframes * tframe * nramps.

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
    if not np.isfinite(n_reset_frames) or n_reset_frames < 0:
        raise ValueError("n_reset_frames must be >= 0")
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
    n, m, d = info["nmd_used"]
    nframes = n * (m + d) - d
    # wall clock per ramp = (nframes + n_reset_frames) * tframe; slicersim counts nframes only
    t_obs = info["t_int_s"] * (nframes + n_reset_frames) / nframes

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_in = snr * np.sqrt(t14_s / t_obs)
        err = np.sqrt(2.0) / snr_in / np.sqrt(n_transits)
    good = np.isfinite(err) & (snr_in > 0)
    info.update({"band": band, "sncosmo_band": sn_band,
                 "mag_input": float(mag), "mag_system": mag_system,
                 "vega_ab_offset": offset, "mag_ab": mag_ab,
                 "t14_s": float(t14_s), "t_obs_s": float(t_obs),
                 "nframes": int(nframes), "n_reset_frames": n_reset_frames,
                 "trimmed_channels": int((~good).sum()),
                 "assumption": "equal out-of-transit baseline; SNR ~ sqrt(t); "
                               + ("readout fixed by the caller (saturation reported, not removed)"
                                  if setup[0] is not None else
                                  "readout tuned to remove point-source saturation")})
    return wl_um[good], err[good], info
