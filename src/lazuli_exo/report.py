"""SimulatedSpectrum -> JSON-serialisable summary for web UIs.

Keys mirror the local demo's /api/observe payload so one plotting routine
renders both the showcase tables and the parametric model, plus `result`
(scalars) and `derived` (label/value/cls rows for a diagnostics panel).
"""
import numpy as np

CAVEATS = [
    "isothermal H2/He equilibrium-chemistry model atmosphere with a gray cloud deck and "
    "power-law haze — a comparison model, not a tailored atmospheric prediction",
    "statistical (photon/read) noise only: no systematics floor, red noise, "
    "or visit overheads; errors scale as 1/sqrt(n_transits)",
]


def _row(label, value, cls=""):
    return {"label": label, "value": value, "cls": cls}


def derived_rows(obs):
    a = obs.meta.get("atmosphere", {})
    s = obs.meta.get("star", {})
    n = obs.meta.get("noise", {})
    rows = []
    if a:
        rows += [
            _row("g", f"{a['g_ms2']:.1f} m/s²"),
            _row("H", f"{a['H_m'] / 1e3:.0f} km (H/Rp = {a['H_over_Rp']:.4f})"),
            _row("mu", f"{a['mu_used']:.2f} ({a['mu_source']})"),
            _row("T for chemistry",
                 f"{a['T_chem_K']:.0f} K" + (" (clamped to grid floor; H uses T_eq)" if a["T_clamped"] else ""),
                 "warn" if a["T_clamped"] else ""),
            _row("feature amplitude", f"{a['feature_amplitude_ppm']:.0f} ppm (0.40–1.70 µm)"),
            _row("column slope", f"{a.get('column_slope_median', float('nan')):.2f}"),
        ]
    if s:
        notes = []
        if s.get("fallback"):
            notes.append(f"{s['fallback']} fallback (T_eff outside the PHOENIX grid)")
        for k, (want, got) in (s.get("clamped") or {}).items():
            notes.append(f"{k} {want:g} clamped to {got:g}")
        rows.append(_row("stellar SED", s.get("grid", "?") + (": " + "; ".join(notes) if notes else ""),
                         "warn" if notes else ""))
    if n:
        nmd = n.get("nmd_used") or []
        rows += [
            _row("magnitude", f"{n.get('mag_input', float('nan')):.2f} {n.get('band', '')} "
                              f"{n.get('mag_system', '')} → {n.get('mag_ab', float('nan')):.2f} AB "
                              f"(offset {n.get('vega_ab_offset', 0.0):+.3f})"),
            _row("readout", (":".join(str(x) for x in nmd) if nmd else "?")
                 + f", t_obs = {n.get('t_obs_s', float('nan')):.1f} s, {n.get('instrument', '')}"),
            _row("saturated pixels", str(n.get("nsaturated_detpx", 0)),
                 "bad" if n.get("nsaturated_detpx") else ""),
            _row("slicersim", str(n.get("slicersim_version", "?"))),
        ]
    return rows


def summary(obs, seed=0):
    n = obs.meta.get("noise", {})
    validated = n.get("fully_desaturated") is not False
    caveats = list(CAVEATS)
    if not validated:
        caveats.append(f"readout at the desaturation floor still leaves "
                       f"{n.get('nsaturated_detpx')} saturated detector pixels — "
                       "uncertainties unvalidated")
    det = obs.detect()
    return {
        "planet": obs.planet, "scenario": obs.scenario,
        "n_transits": int(obs.n_transits), "R": int(obs.R), "seed": int(seed),
        "wavelength": obs.wavelength.tolist(), "depth": obs.depth.tolist(),
        "depth_err": obs.depth_err.tolist(),
        "points": obs.simulate_points(seed=seed).tolist(),
        "model_wavelength": obs.model_wavelength.tolist(),
        "model_depth": obs.model_depth.tolist(),
        "flat_depth": float(obs.flat_depth),
        "feature_snr": {"template_snr_diagonal": float(det["template_snr_diagonal"]),
                        "band_snr": {k: float(v) for k, v in det["band_snr"].items()}},
        "caveats": caveats, "noise_validated": validated,
        "result": {"template_snr_diagonal": float(det["template_snr_diagonal"]),
                   "feature_amplitude_ppm": obs.meta.get("atmosphere", {}).get("feature_amplitude_ppm"),
                   "median_err_ppm": float(np.median(obs.depth_err)) * 1e6,
                   "n_bins": int(obs.depth.size)},
        "derived": derived_rows(obs),
    }
