import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import lazuli_exo
from conftest import PLANET  # pytest puts tests/ on sys.path


def test_plot_returns_axes(fixture_tables):
    obs = lazuli_exo.observe(PLANET, R=50, tables_dir=fixture_tables)
    ax = obs.plot(seed=1)
    assert ax.get_xlabel()
    assert len(ax.lines) + len(ax.containers) >= 2  # model line + errorbars
    assert ax.get_ylabel() == "Transit depth (%)"
    assert PLANET in ax.get_title()


def test_plot_overlays_on_shared_axes_and_units(fixture_tables):
    _, ax = plt.subplots()
    for R, s in ((30, "a"), (60, "b")):
        o = lazuli_exo.observe(PLANET, R=R, tables_dir=fixture_tables)
        assert o.plot(ax=ax, units="ppm", show_points=False, label=s) is ax
    assert ax.get_title() == ""                       # not overridden on a given ax
    assert ax.get_ylabel() == "Transit depth (ppm)"
    assert [l.get_label() for l in ax.lines] == ["a", "b"]
    with pytest.raises(ValueError):
        o.plot(ax=ax, units="mmag")


def test_plot_noise(fixture_tables):
    obs = lazuli_exo.observe(PLANET, R=50, n_transits=4, tables_dir=fixture_tables)
    ax = obs.plot_noise()
    assert ax.get_yscale() == "log"
    assert "error" in ax.get_ylabel()
    (y,) = [l.get_ydata() for l in ax.lines]
    assert y.shape == obs.depth_err.shape


def test_light_curve_plot(fixture_tables):
    obs = lazuli_exo.observe(PLANET, R=50, tables_dir=fixture_tables)
    lcb = obs.light_curves(n_groups=3, seed=0)
    ax = lcb.plot()
    assert len(ax.lines) == 6                         # 3 models + 3 point sets
    assert "offset" in ax.get_ylabel()
    assert [t.get_text() for t in ax.texts] == lcb.labels
    ax2 = obs.light_curves(n_groups=1, seed=0).plot()
    assert [l.get_label() for l in ax2.lines] == ["simulated (60 s cadence)", "model"]
    assert ax2.get_ylabel() == "Normalized flux"
    obs.light_curves(n_groups=1, seed=0).plot(ax=ax2, show_points=False, label="again")
    assert ax2.lines[-1].get_label() == "again"


def test_shared_axes_keep_log_limits(fixture_tables):
    _, axes = plt.subplots(2, 1, sharex=True)
    for ax, R in zip(axes, (30, 60)):
        lazuli_exo.observe(PLANET, R=R, tables_dir=fixture_tables).plot(ax=ax, label="x")
    lo, hi = axes[1].get_xlim()
    assert lo > 0.3 and hi < 2.0                      # not squeezed by linear autoscale
    assert axes[1].lines[0].get_label() == "model"
