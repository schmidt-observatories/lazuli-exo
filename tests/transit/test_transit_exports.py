import lazuli_exo.transit as transit


def test_top_level_exports():
    for name in (
        "FluxModel",
        "TransitModel",
        "plot_flux_model",
        "download_exoplanet_archive",
        "load_exoplanet_archive",
        "default_archive_path",
        "ARCHIVE_COLUMNS",
    ):
        assert hasattr(transit, name), name
        assert name in transit.__all__


def test_transit_subpackage_imports_with_numpy_only(monkeypatch):
    """`import lazuli_exo.transit` must not need pandas, astropy, jax or matplotlib."""
    import importlib
    import sys

    for mod in ("pandas", "astropy", "jax", "jaxoplanet", "matplotlib"):
        monkeypatch.setitem(sys.modules, mod, None)
    for name in [m for m in sys.modules if m.startswith("lazuli_exo.transit")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.import_module("lazuli_exo.transit")
