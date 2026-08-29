"""terminator: Lazuli IFS exoplanet transmission spectroscopy simulator."""
from . import binning, tables
from .observe import SimulatedSpectrum, observe
from .tables import list_planets, list_scenarios

__all__ = ["observe", "SimulatedSpectrum", "list_planets", "list_scenarios",
           "tables", "binning"]
