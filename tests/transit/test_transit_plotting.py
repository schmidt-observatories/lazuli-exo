import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for CI

from lazuli_exo.transit import FluxModel, plot_flux_model


class BoxModel(FluxModel):
    """Flat 1% dip for |t| < 0.1; no backend needed."""

    def relative_flux(self, time):
        return np.where(np.abs(np.asarray(time)) < 0.1, 0.99, 1.0)


def test_plot_flux_model_returns_axes_with_a_line():
    ax = plot_flux_model(BoxModel(), np.linspace(-0.3, 0.3, 50))
    assert len(ax.lines) == 1


def test_fluxmodel_plot_delegates_to_plot_flux_model():
    ax = BoxModel().plot(np.linspace(-0.3, 0.3, 50))
    assert ax.get_ylabel() == "Relative flux"
