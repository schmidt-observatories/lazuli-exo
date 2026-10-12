"""lazuli-exo: Lazuli IFS exoplanet transmission spectroscopy simulator."""
__version__ = "0.1.1"

from . import atmosphere, binning, forms, lightcurve, presets, report, slicer, star, tables
from .observe import SimulatedSpectrum, observe
from .simulate import simulate
from .tables import list_planets, list_scenarios

__all__ = ["__version__", "observe", "simulate", "SimulatedSpectrum", "list_planets",
           "list_scenarios", "tables", "binning", "lightcurve", "atmosphere",
           "star", "slicer", "presets", "forms", "report"]
