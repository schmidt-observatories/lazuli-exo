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
    assert set(j["detections"]) == set(direct)
    for mol, sig in direct.items():
        assert j["detections"][mol] == pytest.approx(sig)
    assert (j["planet"], j["scenario"]) == ("WASP-39 b", "cloudy")
    assert (j["n_transits"], j["R"], j["seed"]) == (5, 100, 3)


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


def test_r_beyond_native_noise_sampling_is_json_400(client):
    """R within the accepted range can still exceed the noise tables'
    native sampling; the library's ValueError must surface as a JSON 400,
    never an HTML 500 (the front end parses every response as JSON)."""
    r = client.get("/api/observe",
                   query_string={"planet": "WASP-39 b", "R": 500})
    assert r.status_code == 400
    assert "native noise sampling" in r.get_json()["error"]
