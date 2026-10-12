"""Local Flask demo UI for lazuli-exo (requires the [web] extra).

Run from the repo root:  flask --app webapp/app run
Not the deployable frontend — no packaging, persistence, or auth.
"""
import math

from flask import Flask, jsonify, render_template, request

import lazuli_exo
from lazuli_exo import forms, report, tables
from lazuli_exo.targets import get_target

N_TRANSITS_MAX = 1000
R_MIN, R_MAX = 10, 500
GROUPS_MAX = 12


class BadRequest(ValueError):
    pass


def _int_param(args, name, default):
    raw = args.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise BadRequest(f"{name} must be an integer, got {raw!r}")


def _validated_observe(args):
    """Parse/validate common query params and run observe().

    Returns (obs, seed); raises BadRequest with a user-facing message.
    """
    planet = args.get("planet")
    if planet is None:
        raise BadRequest("planet is required")
    if planet not in lazuli_exo.list_planets():
        raise BadRequest(f"unknown planet {planet!r}")
    scenario = args.get("scenario", "clear")
    if scenario not in lazuli_exo.list_scenarios(planet):
        raise BadRequest(f"unknown scenario {scenario!r} for {planet}")
    n_transits = _int_param(args, "n_transits", 1)
    R = _int_param(args, "R", 100)
    seed = _int_param(args, "seed", 0)
    if not 1 <= n_transits <= N_TRANSITS_MAX:
        raise BadRequest(f"n_transits must be in [1, {N_TRANSITS_MAX}]")
    if not R_MIN <= R <= R_MAX:
        raise BadRequest(f"R must be in [{R_MIN}, {R_MAX}]")
    try:
        obs = lazuli_exo.observe(planet, scenario=scenario,
                                 n_transits=n_transits, R=R)
    except ValueError as e:
        # e.g. R within our cap but beyond the noise tables' native
        # sampling — surface the library's message as JSON, not HTML 500
        raise BadRequest(str(e))
    return obs, seed


R_JUP_M = 7.1492e7
M_JUP_KG = 1.89813e27
G_SI = 6.674e-11
K_B = 1.380649e-23
M_PROTON = 1.6726e-27
MU_H2 = 2.3          # H2/He-dominated mean molecular weight (assumption)
R_SUN_M = 6.957e8


def _observing(noise_meta):
    """Slicersim observing setup: raw noise-table meta plus derived
    readout quantities (frame/ramp timing, ramps per transit, and the
    equal in/out-of-transit time accounting the tables assume)."""
    out = {k: noise_meta.get(k) for k in
           ("instrument", "J_mag", "t14_s", "t_obs_s", "nmd_used",
            "default_nmd", "nramps", "nsaturated_detpx",
            "fully_desaturated", "norm_band")}
    nmd, t_obs = noise_meta.get("nmd_used"), noise_meta.get("t_obs_s")
    t14 = noise_meta.get("t14_s")
    if nmd and t_obs:
        n, m, d = nmd
        nramps = noise_meta.get("nramps") or 1
        nframes = n * (m + d) - d
        # tables since 2026-10-11 record tframe_s and count a reset frame per
        # ramp in t_obs_s; older tables have neither, so derive from t_obs
        tframe = noise_meta.get("tframe_s") or (t_obs / nramps) / nframes
        ramp_s = tframe * nframes
        out.update(
            readout_mode="MACC up-the-ramp",
            macc=f"{n}:{m}:{d}",
            nframes_per_ramp=nframes,
            tframe_s=tframe,
            ramp_s=ramp_s,
            ramps_per_transit=(t14 / ramp_s) if t14 else None)
        if noise_meta.get("default_nmd"):
            dn, dm, dd = noise_meta["default_nmd"]
            out["default_macc"] = f"{dn}:{dm}:{dd}"
            out["default_ramp_s"] = (dn * (dm + dd) - dd) * tframe
    if t14:
        # noise tables assume the full transit observed plus an equal
        # out-of-transit baseline (the sqrt(2) in depth_err)
        out["t_in_s"] = t14
        out["t_out_s"] = t14
    return out


def _derived(t):
    """Quantities computed from the registry entry (None-safe)."""
    e = t["ecc"] if t["ecc"] is not None else 0.0
    w = math.radians(t["w_deg"] if t["w_deg"] is not None else 90.0)
    b = (t["a_over_rs"] * math.cos(math.radians(t["inc_deg"]))
         * (1.0 - e**2) / (1.0 + e * math.sin(w)))
    out = {"b": b}
    if t["M_p_MJ"] and t["R_p_RJ"] and t["T_eq_K"]:
        r_p = t["R_p_RJ"] * R_JUP_M
        g_p = G_SI * t["M_p_MJ"] * M_JUP_KG / r_p**2
        h_m = K_B * t["T_eq_K"] / (MU_H2 * M_PROTON * g_p)
        r_s = t["R_s_RSun"] * R_SUN_M
        out.update(
            g_p_ms2=g_p, H_km=h_m / 1e3, mu_assumed=MU_H2,
            feature_amp_ppm=2.0 * h_m * r_p / r_s**2 * 1e6)
    return out


def create_app():
    app = Flask(__name__)

    @app.errorhandler(BadRequest)
    def bad_request(e):
        return jsonify(error=str(e)), 400

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/targets")
    def targets():
        return jsonify({p: lazuli_exo.list_scenarios(p)
                        for p in lazuli_exo.list_planets()})

    @app.get("/api/target")
    def target():
        planet = request.args.get("planet")
        if planet is None:
            raise BadRequest("planet is required")
        try:
            t = get_target(planet)
        except KeyError:
            raise BadRequest(f"unknown planet {planet!r}")
        *_, noise_meta = tables.load_noise(
            tables.resolve_tables_dir(None), planet)
        return jsonify(target=t, derived=_derived(t),
                       observing=_observing(noise_meta))

    @app.get("/api/observe")
    def observe():
        obs, seed = _validated_observe(request.args)
        noise_meta = obs.meta["noise"]
        validated = noise_meta.get("fully_desaturated") is not False
        caveats = [
            "H2/He isothermal equilibrium-chemistry toy scenarios — "
            "not tailored atmospheric predictions",
            "statistical (photon/read) noise only: no systematics floor, "
            "red noise, or visit overheads; errors scale as 1/√n_transits",
        ]
        if not validated:
            caveats.append(
                f"noise table retains {noise_meta.get('nsaturated_detpx')} "
                "saturated detector pixels — uncertainties unvalidated")
        return jsonify(
            noise_validated=validated,
            caveats=caveats,
            planet=obs.planet, scenario=obs.scenario,
            n_transits=obs.n_transits, R=obs.R, seed=seed,
            wavelength=obs.wavelength.tolist(),
            depth=obs.depth.tolist(),
            depth_err=obs.depth_err.tolist(),
            points=obs.simulate_points(seed=seed).tolist(),
            model_wavelength=obs.model_wavelength.tolist(),
            model_depth=obs.model_depth.tolist(),
            flat_depth=obs.flat_depth,
            feature_snr=obs.detect(),
        )

    # --- parametric model: thin wrappers, the logic lives in lazuli_exo ---

    @app.get("/api/presets")
    def presets_list():
        return jsonify(planets=lazuli_exo.presets.PLANET_NAMES,
                       scenarios=lazuli_exo.presets.SCENARIOS,
                       defaults=forms.FIELD_DEFAULTS)

    @app.get("/api/preset")
    def preset():
        try:
            values = forms.from_preset(request.args.get("planet", ""),
                                       request.args.get("scenario", "clear"))
        except KeyError as e:
            raise BadRequest(str(e))
        return jsonify(values=values)

    @app.get("/api/simulate")
    def simulate():
        res = forms.parse(request.args)
        if res.errors:
            raise BadRequest("; ".join(res.errors))
        v = res.values
        try:
            obs = lazuli_exo.simulate(
                forms.to_params(v), R=v["resolution"], n_transits=v["n_transits"],
                instrument=forms.INSTRUMENTS[v["instrument_config"]])
        except ValueError as e:        # grid ranges, unbound atmosphere, PHOENIX ceiling
            raise BadRequest(str(e))
        return jsonify(**report.summary(obs, seed=v["seed"]), warnings=res.warnings)

    @app.get("/api/lightcurve")
    def light_curve():
        obs, seed = _validated_observe(request.args)
        groups = _int_param(request.args, "groups", 1)
        if not 1 <= groups <= GROUPS_MAX:
            raise BadRequest(f"groups must be in [1, {GROUPS_MAX}]")
        lc = obs.light_curves(n_groups=groups, seed=seed)
        return jsonify(
            planet=obs.planet, scenario=obs.scenario,
            n_transits=obs.n_transits, R=obs.R, seed=seed,
            time_hr=lc.time_hr.tolist(),
            t14_s=lc.t14_s, cadence_s=lc.cadence_s,
            shape=lc.meta["shape"],
            b=lc.meta.get("b"), a_over_rs=lc.meta.get("a_over_rs"),
            groups=[{
                "label": lc.labels[i],
                "wl_lo": float(lc.wl_lo[i]), "wl_hi": float(lc.wl_hi[i]),
                "depth": float(lc.depth[i]),
                "sigma_point": float(lc.sigma_point[i]),
                "u1": None if lc.u1 is None else float(lc.u1[i]),
                "u2": None if lc.u2 is None else float(lc.u2[i]),
                "flux_model": lc.flux_model[i].tolist(),
                "flux": lc.flux[i].tolist(),
            } for i in range(len(lc.labels))],
        )

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
