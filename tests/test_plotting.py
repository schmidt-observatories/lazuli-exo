import dataclasses
import warnings

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
    # an empty Axes passed in is titled too; a populated one is left alone
    _, ax2 = plt.subplots()
    obs.plot(ax=ax2)
    assert ax2.get_title() == ax.get_title()
    obs.plot(ax=ax2, title="")
    assert ax2.get_title() == ax.get_title()


def test_plot_overlays_on_shared_axes_and_units(fixture_tables):
    _, ax = plt.subplots()
    for R, s in ((30, "a"), (60, "b")):
        o = lazuli_exo.observe(PLANET, R=R, tables_dir=fixture_tables)
        assert o.plot(ax=ax, units="ppm", show_points=False, label=s) is ax
    assert ax.get_ylabel() == "Transit depth (ppm)"
    assert [l.get_label() for l in ax.lines] == ["a", "b"]
    with pytest.raises(ValueError):
        o.plot(ax=ax, units="mmag")


def test_plot_labels_and_colour_with_points(fixture_tables):
    obs = lazuli_exo.observe(PLANET, R=30, tables_dir=fixture_tables)
    _, ax = plt.subplots()
    obs.plot(ax=ax, label="x", c="r")
    obs.plot(ax=ax, label="y")
    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["model (x)", "model (y)", "x", "y"]
    assert ax.lines[0].get_color() == "r"            # c= reaches the model line
    assert ax.containers[0][0].get_color() == "r"


def test_plot_noise(fixture_tables):
    obs = lazuli_exo.observe(PLANET, R=50, n_transits=4, tables_dir=fixture_tables)
    ax = obs.plot_noise()
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel() == "1-sigma depth error (ppm)"
    (y,) = [l.get_ydata() for l in ax.lines]
    assert y.shape == obs.depth_err.shape


def test_light_curve_plot(fixture_tables):
    obs = lazuli_exo.observe(PLANET, R=50, tables_dir=fixture_tables)
    lcb = obs.light_curves(n_groups=3, seed=0)
    ax = lcb.plot()
    assert len(ax.lines) == 6                         # 3 models + 3 point sets
    assert "offset" in ax.get_ylabel()
    assert [t.get_text() for t in ax.texts] == lcb.labels
    lc = obs.light_curves(n_groups=1, seed=0)
    ax2 = lc.plot()
    assert [l.get_label() for l in ax2.lines] == ["simulated, 60 s cadence", "model"]
    assert ax2.get_ylabel() == "Normalized flux"
    assert ax2.get_title() == f"{PLANET}, clear, 1 transit, 60 s cadence"
    lc.plot(ax=ax2, label="again")
    assert ax2.get_legend_handles_labels()[1][-2:] == ["simulated (again), 60 s cadence",
                                                       "model (again)"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # no 'No artists' warning
        ax3 = lc.plot(show_points=False, show_model=False)
    assert ax3.get_legend() is None
    bare = dataclasses.replace(lc, meta={})
    assert bare.plot().get_title() == "60 s cadence"
    flat = dataclasses.replace(lcb, depth=lcb.depth * 0.0 - 1e-5)
    assert "offset" in flat.plot().get_ylabel()       # falls back to the flux range


def test_shared_axes_keep_log_limits(fixture_tables):
    _, axes = plt.subplots(2, 1, sharex=True)
    obs = lazuli_exo.observe(PLANET, R=30, tables_dir=fixture_tables)
    obs.plot(ax=axes[0])
    obs.plot(ax=axes[1], show_model=False)            # narrower range last
    lo, hi = axes[1].get_xlim()
    assert lo <= obs.model_wavelength.min() and hi >= obs.model_wavelength.max()
    assert lo > 0.3 and hi < 2.0                      # not squeezed by linear autoscale


def test_transit_still_and_animation_fallback(fixture_tables):
    # fixture planet: no registry orbit, so the trapezoid path and a uniform disk
    lcb = lazuli_exo.observe(PLANET, R=50, tables_dir=fixture_tables).light_curves(n_groups=3, seed=0)
    fig = lcb.plot_transit(t_hr=0.0)
    assert len(fig.axes) == 4                          # 3 disks + the light curves
    planet = [p for p in fig.axes[0].patches if p.get_facecolor()[:3] == (0.0, 0.0, 0.0)][-1]
    assert planet.center == pytest.approx((0.0, 0.0), abs=0.02)   # central chord, within half a cadence of mid-transit
    anim = lcb.animate(n_frames=4)
    assert anim._save_count == 4 or anim.save_count == 4
    plt.close("all")


def test_transit_registry_planet_uses_orbit():
    lcb = lazuli_exo.observe("WASP-39 b", R=50).light_curves(n_groups=2, seed=0)
    fig = lcb.plot_transit(t_hr=0.0)
    planet = [p for p in fig.axes[0].patches if p.get_facecolor()[:3] == (0.0, 0.0, 0.0)][-1]
    assert planet.center[1] == pytest.approx(lcb.meta["b"])        # impact parameter
    assert "u1 = 0." in fig.axes[0].get_title() and "u1 = 0.00" not in fig.axes[0].get_title()
    plt.close("all")
