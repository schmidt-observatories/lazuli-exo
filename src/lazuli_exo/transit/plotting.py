"""Reusable matplotlib plotting helpers for lazuli_exo.transit flux models.

matplotlib is an optional dependency, lazy-imported here so the package core
stays numpy/astropy/pandas-only. Each function takes an optional ``ax`` and
returns the matplotlib ``Axes`` so results compose and embed (e.g. in an applet).
"""
import numpy as np

_MPL_HINT = (
    "Plotting requires 'matplotlib'. "
    "Install it with: pip install lazuli-exo[plot]"
)


def _get_ax(ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via hint
        raise ImportError(_MPL_HINT) from exc
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_flux_model(model, time, ax=None, **kw):
    """Plot any FluxModel's ``relative_flux(time)`` curve. Returns the Axes."""
    ax = _get_ax(ax)
    time = np.asarray(time, dtype=float)
    ax.plot(time, model.relative_flux(time), label="flux", **kw)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative flux")
    return ax


def plot_moon_transit(model, time=None, show_components=True, ax=None, **kw):
    """Plot a MoonTransitModel light curve.

    When ``time`` is None, uses ``model.default_time()``. With
    ``show_components`` (default True) overlays planet-only and moon-only
    curves from ``model.components`` so the moon's dip is visible.
    Returns the Axes.
    """
    ax = _get_ax(ax)
    if time is None:
        time = model.default_time()
    time = np.asarray(time, dtype=float)
    total, planet, moon = model.components(time)
    ax.plot(time, total, label="total (planet+moon)", **kw)
    if show_components:
        ax.plot(time, planet, "--", label="planet only")
        ax.plot(time, moon, ":", label="moon only")
        ax.legend()
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative flux")
    return ax
