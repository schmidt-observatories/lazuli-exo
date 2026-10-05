"""terminator: Lazuli IFS exoplanet transmission spectroscopy simulator."""
from . import atmosphere, binning, lightcurve, presets, slicer, star, tables
from .observe import SimulatedSpectrum, observe
from .simulate import simulate
from .tables import list_planets, list_scenarios

__all__ = ["observe", "simulate", "SimulatedSpectrum", "list_planets",
           "list_scenarios", "tables", "binning", "lightcurve", "atmosphere",
           "star", "slicer", "presets"]
