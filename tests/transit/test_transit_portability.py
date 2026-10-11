from pathlib import Path

import lazuli_exo.transit as transit


def test_no_instrument_package_dependency_in_source():
    """The transit models stay instrument-agnostic: no wcc_etc imports."""
    pkg_dir = Path(transit.__file__).parent
    offenders = [p.name for p in pkg_dir.rglob("*.py") if "wcc_etc" in p.read_text()]
    assert not offenders, f"lazuli_exo.transit must not reference wcc_etc: {offenders}"
