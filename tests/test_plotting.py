import matplotlib
matplotlib.use("Agg")

import slixer
from conftest import PLANET  # pytest puts tests/ on sys.path


def test_plot_returns_axes(fixture_tables):
    obs = slixer.observe(PLANET, R=50, tables_dir=fixture_tables)
    ax = obs.plot(seed=1)
    assert ax.get_xlabel()
    assert len(ax.lines) + len(ax.containers) >= 2  # model line + errorbars
