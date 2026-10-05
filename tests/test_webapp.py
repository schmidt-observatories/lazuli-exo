"""Tests for the local Flask demo UI (webapp/)."""
import pathlib
import sys

import numpy as np
import pytest

import terminator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

flask = pytest.importorskip("flask")
from webapp.app import create_app  # noqa: E402


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_index_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"plotly" in r.data.lower()


def test_targets_shape(client):
    r = client.get("/api/targets")
    assert r.status_code == 200
    targets = r.get_json()
    assert set(targets) == set(terminator.list_planets())
    assert targets["WASP-39 b"] == terminator.list_scenarios("WASP-39 b")


def test_observe_matches_direct_call(client):
    r = client.get("/api/observe", query_string={
        "planet": "WASP-39 b", "scenario": "cloudy",
        "n_transits": 5, "R": 100, "seed": 3})
    assert r.status_code == 200
    j = r.get_json()

    obs = terminator.observe("WASP-39 b", scenario="cloudy",
                             n_transits=5, R=100)
    np.testing.assert_allclose(j["wavelength"], obs.wavelength)
    np.testing.assert_allclose(j["depth"], obs.depth)
    np.testing.assert_allclose(j["depth_err"], obs.depth_err)
    np.testing.assert_allclose(j["points"], obs.simulate_points(seed=3))
    np.testing.assert_allclose(j["model_wavelength"], obs.model_wavelength)
    np.testing.assert_allclose(j["model_depth"], obs.model_depth)
    assert j["flat_depth"] == pytest.approx(obs.flat_depth)
    direct = obs.detect()
    assert "detections" not in j          # renamed: not detection sigmas
    fsnr = j["feature_snr"]
    assert fsnr["template_snr_diagonal"] == pytest.approx(
        direct["template_snr_diagonal"])
    assert set(fsnr["band_snr"]) == set(direct["band_snr"])
    for mol, snr in direct["band_snr"].items():
        assert fsnr["band_snr"][mol] == pytest.approx(snr)
    assert (j["planet"], j["scenario"]) == ("WASP-39 b", "cloudy")
    assert (j["n_transits"], j["R"], j["seed"]) == (5, 100, 3)


def test_observe_carries_status_caveats(client):
    j = client.get("/api/observe", query_string={
        "planet": "WASP-39 b"}).get_json()
    assert j["noise_validated"] is True
    joined = " ".join(j["caveats"]).lower()
    assert "toy" in joined
    assert "statistical" in joined or "photon" in joined
    assert "saturat" not in joined

    j = client.get("/api/observe", query_string={
        "planet": "HD 209458 b"}).get_json()
    assert j["noise_validated"] is False
    assert any("saturat" in c.lower() for c in j["caveats"])


def test_observe_defaults(client):
    r = client.get("/api/observe", query_string={"planet": "WASP-39 b"})
    assert r.status_code == 200
    j = r.get_json()
    assert (j["scenario"], j["n_transits"], j["R"], j["seed"]) == \
        ("clear", 1, 100, 0)


def test_seed_reproducibility(client):
    q = {"planet": "GJ 1214 b", "scenario": "hazy", "seed": 7}
    a = client.get("/api/observe", query_string=q).get_json()
    b = client.get("/api/observe", query_string=q).get_json()
    assert a["points"] == b["points"]
    c = client.get("/api/observe",
                   query_string={**q, "seed": 8}).get_json()
    assert a["points"] != c["points"]


@pytest.mark.parametrize("query", [
    {},                                                    # missing planet
    {"planet": "Vulcan b"},                                # unknown planet
    {"planet": "WASP-39 b", "scenario": "lava"},           # unknown scenario
    {"planet": "WASP-39 b", "n_transits": 0},              # < 1
    {"planet": "WASP-39 b", "n_transits": 1001},           # > cap
    {"planet": "WASP-39 b", "n_transits": "2.5"},          # non-integer
    {"planet": "WASP-39 b", "R": 5},                       # R too low
    {"planet": "WASP-39 b", "R": 501},                     # R too high
    {"planet": "WASP-39 b", "seed": "abc"},                # bad seed
])
def test_observe_rejects_bad_inputs(client, query):
    r = client.get("/api/observe", query_string=query)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_lightcurve_matches_direct_call(client):
    r = client.get("/api/lightcurve", query_string={
        "planet": "WASP-39 b", "scenario": "cloudy",
        "n_transits": 2, "R": 100, "seed": 4, "groups": 3})
    assert r.status_code == 200
    j = r.get_json()

    obs = terminator.observe("WASP-39 b", scenario="cloudy",
                             n_transits=2, R=100)
    lc = obs.light_curves(n_groups=3, seed=4)
    np.testing.assert_allclose(j["time_hr"], lc.time_hr)
    assert len(j["groups"]) == 3
    for i, g in enumerate(j["groups"]):
        np.testing.assert_allclose(g["flux_model"], lc.flux_model[i])
        np.testing.assert_allclose(g["flux"], lc.flux[i])
        assert g["depth"] == pytest.approx(lc.depth[i])
        assert g["sigma_point"] == pytest.approx(lc.sigma_point[i])
        assert g["label"] == lc.labels[i]
    assert j["t14_s"] == pytest.approx(lc.t14_s)
    assert j["cadence_s"] == pytest.approx(lc.cadence_s)
    assert j["shape"] == lc.meta["shape"] == "quadratic-ld"
    assert j["b"] == pytest.approx(lc.meta["b"])
    assert j["a_over_rs"] == pytest.approx(lc.meta["a_over_rs"])
    for i, g in enumerate(j["groups"]):
        assert g["u1"] == pytest.approx(lc.u1[i])
        assert g["u2"] == pytest.approx(lc.u2[i])


@pytest.mark.parametrize("query", [
    {"planet": "WASP-39 b", "groups": 0},
    {"planet": "WASP-39 b", "groups": 13},
    {"planet": "WASP-39 b", "groups": "two"},
    {"planet": "Vulcan b"},
])
def test_lightcurve_rejects_bad_inputs(client, query):
    r = client.get("/api/lightcurve", query_string=query)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_target_endpoint(client):
    r = client.get("/api/target", query_string={"planet": "WASP-39 b"})
    assert r.status_code == 200
    j = r.get_json()
    for f in ("R_p_RJ", "M_p_MJ", "period_d", "a_over_rs", "inc_deg",
              "T_eff_K", "log_g_star", "J_mag", "t14_s", "depth"):
        assert f in j["target"], f
    # derived quantities consistent with the light-curve geometry
    obs = terminator.observe("WASP-39 b")
    lc = obs.light_curves(seed=0)
    assert j["derived"]["b"] == pytest.approx(lc.meta["b"])
    assert j["derived"]["H_km"] > 0
    assert j["derived"]["feature_amp_ppm"] > 0
    # observing setup from the noise-table meta
    o = j["observing"]
    assert o["instrument"] == "lazuli_cbe"
    assert o["t_obs_s"] > 0
    assert isinstance(o["fully_desaturated"], bool)
    # derived readout diagnostics: lazuli_cbe frame time is 2.83 s and the
    # ramp timing must be self-consistent with the recorded nmd
    n, m, d = o["nmd_used"]
    assert o["macc"] == f"{n}:{m}:{d}"
    assert o["nframes_per_ramp"] == n * (m + d) - d
    assert o["tframe_s"] == pytest.approx(2.83, rel=1e-2)
    assert o["ramp_s"] == pytest.approx(o["tframe_s"] * o["nframes_per_ramp"])
    assert o["ramps_per_transit"] == pytest.approx(o["t14_s"] / o["ramp_s"])
    dn, dm, dd = o["default_nmd"]
    assert o["default_macc"] == f"{dn}:{dm}:{dd}"
    assert o["default_ramp_s"] == pytest.approx(
        (dn * (dm + dd) - dd) * o["tframe_s"])
    # equal in/out-of-transit baseline assumption
    assert o["t_in_s"] == o["t_out_s"] == o["t14_s"]


def test_target_endpoint_rejects_unknown(client):
    r = client.get("/api/target", query_string={"planet": "Vulcan b"})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_r_beyond_native_noise_sampling_is_json_400(client):
    """R within the accepted range can still exceed the noise tables'
    native sampling; the library's ValueError must surface as a JSON 400,
    never an HTML 500 (the front end parses every response as JSON)."""
    r = client.get("/api/observe",
                   query_string={"planet": "WASP-39 b", "R": 500})
    assert r.status_code == 400
    assert "native noise sampling" in r.get_json()["error"]


# --- parametric model endpoints (thin wrappers over terminator.forms/report) ---

def _fake_simulate(params, R=100, n_transits=1, tables_dir=None, instrument="lazuli_cbe"):
    wl = np.geomspace(0.4, 1.7, 40)
    return terminator.SimulatedSpectrum(
        planet=params["name"], scenario=params["scenario"], n_transits=n_transits, R=R,
        wavelength=wl, depth=np.full(40, 0.01), depth_err=np.full(40, 1e-4) / np.sqrt(n_transits),
        model_wavelength=wl, model_depth=np.full(40, 0.01), flat_depth=0.01,
        meta={"atmosphere": {"g_ms2": 10.0, "H_m": 5e5, "H_over_Rp": 0.007, "mu_used": 2.3,
                             "mu_source": "chemistry", "T_clamped": False, "T_chem_K": 1200.0,
                             "feature_amplitude_ppm": 500.0, "column_slope_median": 0.8,
                             "zeta_ref": 4.0, "zeta_shift": 0.1},
              "star": {"fallback": None, "clamped": {}, "grid": "phoenix"},
              "noise": {"mag_input": params["mag"], "mag_system": params["mag_system"], "mag_ab": params["mag"] + 0.9,
                        "vega_ab_offset": 0.9, "nsaturated_detpx": 0, "fully_desaturated": True,
                        "t_obs_s": 60.0, "nmd_used": [16, 8, 0], "default_nmd": [16, 8, 0], "nramps": 1,
                        "slicersim_version": "fake", "instrument": instrument, "trimmed_channels": 0,
                        "t14_s": params["t14_s"]},
              "params": params})


def test_preset_endpoint(client):
    j = client.get("/api/preset", query_string={"planet": "WASP-39 b", "scenario": "cloudy"}).get_json()
    assert j["values"]["mag"] == 10.663 and j["values"]["log_p_cloud"] == -3.0 and j["values"]["mu"] is None
    assert "planets" not in j
    r = client.get("/api/preset", query_string={"planet": "Vulcan b"})
    assert r.status_code == 400 and "error" in r.get_json()


def test_presets_listing(client):
    j = client.get("/api/presets").get_json()
    assert "WASP-39 b" in j["planets"] and "hazy" in j["scenarios"]
    assert j["defaults"]["teff"] == "5500"


def test_simulate_endpoint_uses_terminator_and_report(client, monkeypatch):
    monkeypatch.setattr(terminator, "simulate", _fake_simulate)
    j = client.get("/api/simulate", query_string={"mag": "10", "n_transits": "4", "resolution": "50",
                                                  "teff": "abc", "seed": "7"}).get_json()
    assert j["n_transits"] == 4 and j["R"] == 50 and j["seed"] == 7
    assert len(j["points"]) == 40 and j["feature_snr"]["template_snr_diagonal"] >= 0
    assert j["result"]["median_err_ppm"] == pytest.approx(50.0)
    assert any("teff" in w for w in j["warnings"])          # garbage fell back, run proceeded
    assert any(r["label"] == "mu" for r in j["derived"])


def test_simulate_endpoint_fatal_errors_are_json_400(client, monkeypatch):
    monkeypatch.setattr(terminator, "simulate", _fake_simulate)
    r = client.get("/api/simulate", query_string={"teff": "5000"})          # no mag
    assert r.status_code == 400 and "mag" in r.get_json()["error"]
    r = client.get("/api/simulate", query_string={"mag": "10", "rp": "-1"})
    assert r.status_code == 400 and "rp" in r.get_json()["error"]

    def boom(*a, **k):
        raise ValueError("T_eff=9000 K above the PHOENIX grid ceiling 8000 K")
    monkeypatch.setattr(terminator, "simulate", boom)
    r = client.get("/api/simulate", query_string={"mag": "10"})
    assert r.status_code == 400 and "PHOENIX" in r.get_json()["error"]


def test_simulate_canary_real_slicersim(client):
    pytest.importorskip("slicersim")
    v = client.get("/api/preset", query_string={"planet": "WASP-39 b", "scenario": "clear"}).get_json()["values"]
    q = {k: ("" if val is None else str(val)) for k, val in v.items()}
    q.update(resolution="50")
    j = client.get("/api/simulate", query_string=q).get_json()
    assert j["result"]["template_snr_diagonal"] > 1.0 and j["warnings"] == []
    assert {r["label"]: r for r in j["derived"]}["magnitude"]["value"].startswith("10.66 J vega → 11.56")
