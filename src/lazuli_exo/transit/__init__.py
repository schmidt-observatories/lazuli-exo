"""Instrument-agnostic transit models + NASA Exoplanet Archive helpers.

Formerly the standalone ``lazuli-transit`` package. ``FluxModel`` maps time to
normalized relative flux; instrument simulators (the WCC ETC, the IFS) consume
it through ``relative_flux(time)`` and never the other way round: nothing here
may import an instrument package.
"""
from .archive import (
    ARCHIVE_COLUMNS,
    default_archive_path,
    download_exoplanet_archive,
    load_exoplanet_archive,
)
from .models import FluxModel, MoonTransitModel, TransitModel

# Safe to import eagerly only because plotting.py keeps matplotlib out of its
# module top level (lazy-imported inside the functions). Keep it that way, or
# `import lazuli_exo.transit` will start requiring matplotlib.
from .plotting import plot_flux_model, plot_moon_transit

__all__ = [
    "FluxModel",
    "TransitModel",
    "MoonTransitModel",
    "plot_flux_model",
    "plot_moon_transit",
    "download_exoplanet_archive",
    "load_exoplanet_archive",
    "default_archive_path",
    "ARCHIVE_COLUMNS",
]
