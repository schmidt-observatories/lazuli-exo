import json

import numpy as np
import pytest

import terminator
from terminator import forms, presets, report


def test_defaults_when_empty():
    res = forms.parse({})
    assert res.values["teff"] == 5500.0 and res.values["band"] == "J"
    assert res.values["mag_system"] == "vega" and res.values["instrument_config"] == "cbe"
    assert res.values["mu"] is None and res.values["log_p_cloud"] is None and res.values["log_a_haze"] is None
    assert any("mag" in e for e in res.errors)          # magnitude is required -> fatal
    assert res.warnings == []


def test_garbage_falls_back_and_warns():
    res = forms.parse({"mag": "10", "teff": "abc", "log_met": "9", "n_transits": "0",
                       "resolution": "5000", "mag_system": "weird", "band": "Q"})
    v = res.values
    assert res.errors == []
    assert v["teff"] == 5500.0 and v["log_met"] == 3.0 and v["n_transits"] == 1
    assert v["resolution"] == 500 and v["mag_system"] == "vega" and v["band"] == "J"
    joined = " ".join(res.warnings)
    for name in ("teff", "log_met", "n_transits", "resolution", "mag_system", "band"):
        assert name in joined


def test_blank_optionals_and_to_params():
    res = forms.parse({"mag": "10.5", "t14_h": "2.0", "mu": "5", "log_p_cloud": "-2",
                       "log_a_haze": "3", "haze_slope": "-5", "planet_name": "X b"})
    assert res.errors == [] and res.warnings == []
    p = forms.to_params(res.values)
    assert p["t14_s"] == 7200.0 and p["mu"] == 5.0 and p["log_P_cloud_bar"] == -2.0
    assert (p["log_a_haze"], p["haze_slope"]) == (3.0, -5.0) and p["mag_system"] == "vega"
    assert p["name"] == "X b"
    p0 = forms.to_params(forms.parse({"mag": "10"}).values)
    assert p0["mu"] is None and p0["log_P_cloud_bar"] is None and p0["log_a_haze"] is None


def test_nonpositive_physical_values_are_fatal():
    res = forms.parse({"mag": "10", "rp": "-1", "t14_h": "0"})
    assert any("rp" in e for e in res.errors) and any("t14_h" in e for e in res.errors)


def test_from_preset_round_trips_to_showcase_params():
    v = forms.from_preset("WASP-39 b", "hazy")
    assert v["planet_name"] == "WASP-39 b" and v["mag"] == 10.663 and v["log_a_haze"] == 2.0
    assert v["mu"] is None and v["log_p_cloud"] is None
    want = presets.showcase("WASP-39 b", "hazy")
    got = forms.to_params(v)
    for k in ("T_eq_K", "R_p_RJ", "M_p_MJ", "R_s_RSun", "teff_K", "logg", "feh", "mag", "band",
              "mag_system", "t14_s", "log_met", "c_to_o", "mu", "log_P_cloud_bar", "log_a_haze", "haze_slope"):
        assert got[k] == pytest.approx(want[k]) if want[k] is not None else got[k] is None, k
    # a preset serialised through form strings parses back without warnings
    res = forms.parse({k: ("" if val is None else str(val)) for k, val in v.items()})
    assert res.errors == [] and res.warnings == []


def _fake_obs(nsat=0):
    wl = np.geomspace(0.4, 1.7, 60)
    return terminator.SimulatedSpectrum(
        planet="X b", scenario="custom", n_transits=4, R=50, wavelength=wl,
        depth=np.full(60, 0.01), depth_err=np.full(60, 1e-4), model_wavelength=wl,
        model_depth=np.full(60, 0.01), flat_depth=0.01,
        meta={"atmosphere": {"g_ms2": 10.0, "H_m": 5e5, "H_over_Rp": 0.007, "mu_used": 2.3,
                             "mu_source": "chemistry", "T_clamped": True, "T_chem_K": 300.0,
                             "feature_amplitude_ppm": 500.0, "column_slope_median": 0.8,
                             "zeta_ref": 4.0, "zeta_shift": 0.1},
              "star": {"fallback": "blackbody", "clamped": {"logg": (5.3, 5.0)}, "grid": "blackbody"},
              "noise": {"mag_input": 10.0, "mag_system": "vega", "mag_ab": 10.9, "vega_ab_offset": 0.9,
                        "nsaturated_detpx": nsat, "fully_desaturated": nsat == 0, "t_obs_s": 60.0,
                        "nmd_used": [2, 1, 0], "default_nmd": [16, 8, 0], "nramps": 1,
                        "slicersim_version": "fake", "instrument": "lazuli_cbe", "trimmed_channels": 0,
                        "t14_s": 7200.0},
              "params": {}})


def test_report_summary_is_json_and_matches_observe_keys():
    s = report.summary(_fake_obs(), seed=3)
    json.dumps(s)                                   # JSON-serialisable (no numpy types)
    for k in ("planet", "scenario", "n_transits", "R", "seed", "wavelength", "depth", "depth_err",
              "points", "model_wavelength", "model_depth", "flat_depth", "feature_snr",
              "caveats", "noise_validated", "derived", "result"):
        assert k in s, k
    assert s["seed"] == 3 and len(s["points"]) == 60 and s["noise_validated"] is True
    assert s["result"]["feature_amplitude_ppm"] == 500.0 and s["result"]["median_err_ppm"] == pytest.approx(100.0)
    labels = {row["label"]: row for row in s["derived"]}
    assert labels["mu"]["value"].startswith("2.3") and "chemistry" in labels["mu"]["value"]
    assert labels["T for chemistry"]["cls"] == "warn"        # T_clamped
    assert labels["stellar SED"]["cls"] == "warn"            # blackbody fallback + clamp
    assert labels["saturated pixels"]["cls"] == ""


def test_report_flags_saturation():
    s = report.summary(_fake_obs(nsat=784), seed=0)
    assert s["noise_validated"] is False
    assert any("saturat" in c for c in s["caveats"])
    assert {r["label"]: r for r in s["derived"]}["saturated pixels"]["cls"] == "bad"
