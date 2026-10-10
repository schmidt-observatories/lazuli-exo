import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from lazuli_exo.transit import MoonTransitModel, plot_moon_transit, plot_flux_model


def _model():
    return MoonTransitModel(
        R_star=1.0, u=(0.4089, 0.2556), t0=0.0, per=10.0, Rp=11.2, a=0.092,
        b=0.3, M_planet=318.0, Rm=3.9, per_moon=2.5, M_moon=17.0,
        tau=0.5, Omega=0.0, i_moon=90.0,
    )


def test_default_time_is_monotonic_and_centered():
    pytest.importorskip("pandoramoon")
    t = _model().default_time(n=200, span=1.5)
    assert t.size == 200
    assert np.all(np.diff(t) > 0)
    assert np.isfinite(t).all()
    assert t.min() < 0.0 < t.max()  # centered on t0=0


def test_model_plot_returns_axes_with_lines():
    pytest.importorskip("pandoramoon")
    ax = _model().plot()  # time=None -> auto window
    assert ax is not None
    assert len(ax.lines) >= 1


def test_plot_moon_transit_components_draws_three_lines():
    pytest.importorskip("pandoramoon")
    t = np.linspace(-0.3, 0.3, 300)
    ax = plot_moon_transit(_model(), time=t, show_components=True)
    assert len(ax.lines) >= 3  # total + planet + moon


def test_plot_flux_model_generic():
    pytest.importorskip("pandoramoon")
    t = np.linspace(-0.3, 0.3, 300)
    ax = plot_flux_model(_model(), t)
    assert len(ax.lines) >= 1
