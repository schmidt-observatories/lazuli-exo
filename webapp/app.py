"""Local Flask demo UI for terminator (requires the [web] extra).

Run from the repo root:  flask --app webapp/app run
Not the deployable frontend — no packaging, persistence, or auth.
"""
from flask import Flask, jsonify, render_template, request

import terminator

N_TRANSITS_MAX = 1000
R_MIN, R_MAX = 10, 500


def _int_param(args, name, default):
    raw = args.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")


def create_app():
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/targets")
    def targets():
        return jsonify({p: terminator.list_scenarios(p)
                        for p in terminator.list_planets()})

    @app.get("/api/observe")
    def observe():
        planet = request.args.get("planet")
        if planet is None:
            return jsonify(error="planet is required"), 400
        if planet not in terminator.list_planets():
            return jsonify(error=f"unknown planet {planet!r}"), 400
        scenario = request.args.get("scenario", "clear")
        if scenario not in terminator.list_scenarios(planet):
            return jsonify(error=f"unknown scenario {scenario!r} "
                                 f"for {planet}"), 400
        try:
            n_transits = _int_param(request.args, "n_transits", 1)
            R = _int_param(request.args, "R", 100)
            seed = _int_param(request.args, "seed", 0)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        if not 1 <= n_transits <= N_TRANSITS_MAX:
            return jsonify(error=f"n_transits must be in "
                                 f"[1, {N_TRANSITS_MAX}]"), 400
        if not R_MIN <= R <= R_MAX:
            return jsonify(error=f"R must be in [{R_MIN}, {R_MAX}]"), 400

        obs = terminator.observe(planet, scenario=scenario,
                                 n_transits=n_transits, R=R)
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

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
