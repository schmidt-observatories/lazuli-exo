"""Local Flask demo UI for terminator (requires the [web] extra).

Run from the repo root:  flask --app webapp/app run
Not the deployable frontend — no packaging, persistence, or auth.
"""
from flask import Flask, jsonify, render_template, request

import terminator

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
    if planet not in terminator.list_planets():
        raise BadRequest(f"unknown planet {planet!r}")
    scenario = args.get("scenario", "clear")
    if scenario not in terminator.list_scenarios(planet):
        raise BadRequest(f"unknown scenario {scenario!r} for {planet}")
    n_transits = _int_param(args, "n_transits", 1)
    R = _int_param(args, "R", 100)
    seed = _int_param(args, "seed", 0)
    if not 1 <= n_transits <= N_TRANSITS_MAX:
        raise BadRequest(f"n_transits must be in [1, {N_TRANSITS_MAX}]")
    if not R_MIN <= R <= R_MAX:
        raise BadRequest(f"R must be in [{R_MIN}, {R_MAX}]")
    try:
        obs = terminator.observe(planet, scenario=scenario,
                                 n_transits=n_transits, R=R)
    except ValueError as e:
        # e.g. R within our cap but beyond the noise tables' native
        # sampling — surface the library's message as JSON, not HTML 500
        raise BadRequest(str(e))
    return obs, seed


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
        return jsonify({p: terminator.list_scenarios(p)
                        for p in terminator.list_planets()})

    @app.get("/api/observe")
    def observe():
        obs, seed = _validated_observe(request.args)
        return jsonify(
            planet=obs.planet, scenario=obs.scenario,
            n_transits=obs.n_transits, R=obs.R, seed=seed,
            wavelength=obs.wavelength.tolist(),
            depth=obs.depth.tolist(),
            depth_err=obs.depth_err.tolist(),
            points=obs.simulate_points(seed=seed).tolist(),
            model_wavelength=obs.model_wavelength.tolist(),
            model_depth=obs.model_depth.tolist(),
            flat_depth=obs.flat_depth,
            detections=obs.detect(),
        )

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
