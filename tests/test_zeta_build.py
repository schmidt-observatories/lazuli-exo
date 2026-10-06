"""Exercise the build lifecycle without POSEIDON/opacities or production writes."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from lazuli_exo import tables


@pytest.fixture
def builder(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'precompute' / 'make_zeta_grid.py'
    # The script imports its sibling scenarios; restore sys.path after loading.
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('zeta_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'LOG', tmp_path / 'build.log')
    monkeypatch.setattr(module.scenarios, 'make_wl_grid', lambda: np.array([0.4, 1., 1.7]))
    for name in ('POSEIDON', 'POSEIDON.core', 'POSEIDON.chemistry', 'POSEIDON.constants'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    chemistry = sys.modules['POSEIDON.chemistry']
    chemistry.load_chemistry_grid = lambda *a, **k: {}
    constants = sys.modules['POSEIDON.constants']
    constants.R_J, constants.R_Sun = 7.1492e7, 6.957e8
    core = sys.modules['POSEIDON.core']
    core.create_planet = lambda name, radius, **kw: radius
    core.create_star = lambda radius, *a, **kw: radius
    core.define_model = lambda *a, **kw: {}
    core.read_opacities = lambda *a, **kw: {}
    core.make_atmosphere = lambda planet, model, P, *a, **kw: {
        'P': P, 'r': planet - 1e5 * np.log(P),
        'mu': np.full(P.shape, 2.3 * module.AMU_KG)}
    core.compute_spectrum = lambda planet, star, model, atmo, opac, wl, **kw: np.full(
        len(wl), (planet / star)**2)
    return module


def seed_production(tdir):
    tables.rebuild_index(tdir)
    partial = tdir / 'atmo/zeta_grid.partial.npz'
    partial.write_bytes(b'production checkpoint must survive')
    paths = [tdir / tables.ZETA_GRID, tdir / 'index.json', partial]
    return {p: p.read_bytes() for p in paths}


@pytest.mark.parametrize('resume', [False, True])
def test_smoke_preserves_production_grid_index_checkpoint_and_cache(
        builder, zeta_fixture, monkeypatch, resume):
    monkeypatch.setattr(builder, 'TABLES', zeta_fixture)
    before = seed_production(zeta_fixture)
    cached = tables.load_zeta_grid(zeta_fixture)
    smoke = zeta_fixture / 'atmo/zeta_grid.smoke.npz'
    partial = smoke.with_suffix('.partial.npz')
    args = ['--smoke']
    if resume:
        np.savez(partial, zeta=np.full((1, 1, 1, 2, 3), 7., np.float16),
                 mu=np.full((1, 1, 1), 2.3), done=[True], out_of_column='{}')
        def unexpected(*a, **kw):
            pytest.fail('completed smoke checkpoint was not resumed')
        monkeypatch.setattr(sys.modules['POSEIDON.core'], 'read_opacities', unexpected)
        args.append('--resume')
    builder.main(args)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert tables.load_zeta_grid(zeta_fixture) is cached
    assert not partial.exists()
    with np.load(smoke) as f:
        assert f['zeta'].shape == (1, 1, 1, 2, 3)
        assert json.loads(str(f['meta']))['smoke'] is True
        if resume:
            np.testing.assert_array_equal(f['zeta'], 7.)


def test_failed_smoke_keeps_production_and_resume_checkpoint(
        builder, zeta_fixture, monkeypatch):
    monkeypatch.setattr(builder, 'TABLES', zeta_fixture)
    before = seed_production(zeta_fixture)
    def fail(*a, **kw):
        raise OSError('disk full')
    monkeypatch.setattr(tables, 'write_zeta_grid', fail)
    with pytest.raises(OSError, match='disk full'):
        builder.main(['--smoke'])
    assert all(p.read_bytes() == data for p, data in before.items())
    assert (zeta_fixture / 'atmo/zeta_grid.smoke.partial.npz').exists()


def test_production_build_publishes_grid_and_clears_own_checkpoint(
        builder, zeta_fixture, monkeypatch):
    monkeypatch.setattr(builder, 'TABLES', zeta_fixture)
    monkeypatch.setattr(builder, 'ZETA_T_K', [1000.])
    monkeypatch.setattr(builder, 'ZETA_LOG_MET', [0.])
    monkeypatch.setattr(builder, 'ZETA_C_TO_O', [0.55])
    cached = tables.load_zeta_grid(zeta_fixture)
    smoke_partial = zeta_fixture / 'atmo/zeta_grid.smoke.partial.npz'
    smoke_partial.write_bytes(b'smoke checkpoint')
    builder.main([])
    result = tables.load_zeta_grid(zeta_fixture)
    assert result is not cached
    assert result['zeta'].shape == (1, 1, 1, 2, 3)
    assert result['meta']['smoke'] is False
    assert not (zeta_fixture / 'atmo/zeta_grid.partial.npz').exists()
    assert smoke_partial.read_bytes() == b'smoke checkpoint'
    assert tables.load_index(zeta_fixture)['atmo']['zeta_grid'] is True


def test_failed_serialization_preserves_published_grid(zeta_fixture, monkeypatch):
    grid = tables.load_zeta_grid(zeta_fixture)
    path = zeta_fixture / tables.ZETA_GRID
    before = path.read_bytes()
    def partial_write(path, *args, **kwargs):
        Path(path).write_bytes(b'incomplete archive')
        raise OSError('interrupted write')
    monkeypatch.setattr(tables, '_write', partial_write)
    with pytest.raises(OSError, match='interrupted write'):
        tables.write_zeta_grid(zeta_fixture, **grid)
    assert path.read_bytes() == before
    assert tables.load_zeta_grid(zeta_fixture) is grid
    assert not list(path.parent.glob('.grid-*.npz'))
