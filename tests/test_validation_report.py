"""Run the model-vs-POSEIDON gate (issue #22) on the current code.

Both reference sets are committed under precompute/tables; a missing file is
a failure, not a skip. Runtime ~1 min for the 70 showcase cases.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_zeta_grid", ROOT / "precompute" / "validate_zeta_grid.py")
vz = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vz)


def _assert_gate(rows):
    unwaived, stale, _ = vz.summarize(rows)
    assert unwaived == [], f"un-waived gate failures: {unwaived}"
    assert stale == [], f"stale waivers (gate passes now, remove from WAIVED): {stale}"


def test_reference_sets_are_committed():
    models = sorted((vz.MODELS / "models").glob("*.npz"))
    assert len(models) == 14 * 5, f"expected 70 POSEIDON tables in {vz.MODELS / 'models'}"
    assert vz.HOLDOUTS.exists(), f"{vz.HOLDOUTS} missing"


def test_models_gate():
    rows = vz.validate()
    assert sum(not r["like_for_like"] for r in rows) == 14 * 5
    flat = [r for r in rows if r["scenario"] == "flat"]
    assert len(flat) >= 14 and all(r["amp_model_ppm"] < vz.FLAT_MAX_PPM for r in flat)
    assert all(r["gates"] == ["G2", "G5", "G3"] for r in rows if r["scenario"] != "flat")
    _assert_gate(rows)


def test_holdouts_gate():
    rows = vz.validate_holdouts()
    assert len(rows) == 15
    _assert_gate(rows)
