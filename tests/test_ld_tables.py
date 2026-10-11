"""Shipped quadratic LD tables are physical (Kipping 2013) and smooth; the
constrained refit in precompute/make_ld_tables.py behaves (issue #10)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lazuli_exo import tables

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "make_ld_tables", ROOT / "precompute" / "make_ld_tables.py")
make_ld_tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_ld_tables)

TABLES = tables.resolve_tables_dir(None)
PLANETS = sorted(p for p in tables.load_index(TABLES)["planets"]
                 if tables.load_index(TABLES)["planets"][p]["has_ld"])
TOL = 1e-9
# Max chunk-to-chunk |du|. TRAPPIST-1 e's PHOENIX (2500-2600 K) profile jumps
# at 0.42 -> 0.43 um (flux-less blue end; 0.37 unconstrained, 0.41 on the
# u1 + u2 = 1 edge), so it gets a looser regression guard (issue #10).
JUMP_MAX = {"TRAPPIST-1 e": 0.45}


@pytest.mark.parametrize("planet", PLANETS)
def test_shipped_ld_tables_are_physical(planet):
    _, u1, u2, _ = tables.load_ld(TABLES, planet)
    assert u1.shape == u2.shape == (48,)
    assert np.all(u1 >= -TOL), planet
    assert np.all(u1 + u2 <= 1 + TOL), planet       # I(limb) >= 0
    assert np.all(u1 + 2 * u2 >= -TOL), planet      # dI/dmu >= 0 at limb


@pytest.mark.parametrize("planet", PLANETS)
def test_shipped_ld_tables_are_smooth(planet):
    _, u1, u2, meta = tables.load_ld(TABLES, planet)
    jump = JUMP_MAX.get(planet, 0.3)
    assert np.max(np.abs(np.diff(u1))) < jump, planet
    assert np.max(np.abs(np.diff(u2))) < jump, planet
    assert meta["mu_min"] == make_ld_tables.MU_MIN
    assert isinstance(meta["constrained_chunks"], list)


def test_constrain_quadratic_leaves_physical_pairs_unchanged():
    # (1.3, -0.3) sits exactly on the u1 + u2 = 1 edge: boundary, but physical
    for u1, u2 in [(0.3, 0.2), (0.0, 0.5), (0.9, 0.1), (0.5, -0.25),
                   (1.3, -0.3)]:
        assert make_ld_tables.constrain_quadratic(u1, u2) == (u1, u2)


def test_constrain_quadratic_projects_onto_boundary():
    u1, u2 = make_ld_tables.constrain_quadratic(1.3, -0.1)   # u1 + u2 = 1.2
    assert u1 + u2 == pytest.approx(1.0, abs=TOL)
    assert u1 >= -TOL and u1 + 2 * u2 >= -TOL
    # least-squares projection in I(mu) along the edge, not a clip of u2 alone
    assert u1 != pytest.approx(1.3)
