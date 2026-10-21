"""Matplotlib plotting for SimulatedSpectrum and LightCurveSet ([plot] extra).

Every function takes an optional ``ax`` and returns the Axes it drew on, so
plots compose: pass the same ``ax`` to overlay scenarios, or pass one Axes of
a subplot grid. matplotlib is imported inside the functions, never at module
import. The functions do not touch the global style; set
``plt.style.use(...)`` yourself if you want one.
"""
import numpy as np

_MPL_HINT = "Plotting requires matplotlib: pip install lazuli-exo[plot]"
DEPTH_UNITS = {"percent": (100.0, "%"), "ppm": (1e6, "ppm"), "fraction": (1.0, "")}
WAVELENGTH_TICKS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7)


def _get_ax(ax, **subplots_kw):
    """(Axes, fresh): ``fresh`` is True for a new figure or an Axes with no artists."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MPL_HINT) from exc
    if ax is None:
        _, ax = plt.subplots(**subplots_kw)
        return ax, True
    return ax, not ax.has_data()


def _depth_scale(units):
    try:
        scale, unit = DEPTH_UNITS[units]
    except KeyError:
        raise ValueError(f"units must be one of {list(DEPTH_UNITS)}, got {units!r}") from None
    return scale, f" ({unit})" if unit else ""


def _legend(ax, **kw):
    """Legend only when something is labelled (no 'No artists' warning)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, **kw)


def wavelength_axis(ax, ticks=WAVELENGTH_TICKS, margin=1.03):
    """Log wavelength axis with the Lazuli band ticks and a plain-number formatter.

    Call it after plotting: the x-limits are fitted to the lines on the Axes
    and on any Axes sharing its x axis (times ``margin`` each side), which
    sidesteps matplotlib's errorbar data limits on a log axis and keeps
    shared panels on one range.
    """
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Wavelength (µm)")
    xs = [np.asarray(l.get_xdata(), float)
          for a in ax.get_shared_x_axes().get_siblings(ax) for l in a.lines]
    xs = [x[np.isfinite(x) & (x > 0)] for x in xs]
    xs = [x for x in xs if x.size]
    if xs:
        ax.set_xlim(min(x.min() for x in xs) / margin, max(x.max() for x in xs) * margin)
    return ax


def plot_spectrum(obs, ax=None, seed=0, show_model=True, show_points=True,
                  units="percent", label=None, title=None, **errorbar_kw):
    """Model spectrum and one noisy realization of the binned observation.

    ``show_model`` draws the native-resolution model as a line; ``show_points``
    draws ``simulate_points(seed)`` with the per-bin error bars. The legend
    names the points "simulated (scenario, R=...)" and the line "model
    (scenario)"; with ``show_points=False`` the line is labelled by scenario.
    ``label`` replaces the scenario text in those entries. ``units`` is
    "percent", "ppm" or "fraction". ``**errorbar_kw`` goes to ``ax.errorbar``;
    ``color``/``c`` also colours the model line. A title is set on a new
    figure or an empty Axes unless ``title`` is given.
    """
    scale, unit = _depth_scale(units)
    ax, fresh = _get_ax(ax, figsize=(9, 4.5))
    name = label if label is not None else obs.scenario
    if show_points:
        model_label = f"model ({name})"
        points_label = label if label is not None else f"simulated ({name}, R={obs.R})"
    else:
        model_label = name
    if show_model:
        ax.plot(obs.model_wavelength, obs.model_depth * scale, lw=0.8, alpha=0.7,
                color=errorbar_kw.get("color", errorbar_kw.get("c")),
                label=model_label)
    if show_points:
        points = obs.simulate_points(seed=seed)
        kw = dict(fmt="o", ms=3.5, lw=1, capsize=2, label=points_label)
        kw.update(errorbar_kw)
        ax.errorbar(obs.wavelength, points * scale, yerr=obs.depth_err * scale, **kw)
    wavelength_axis(ax)
    ax.set_ylabel(f"Transit depth{unit}")
    if title is None and fresh:
        title = (f"{obs.planet}, {obs.scenario}, {obs.n_transits} "
                 f"transit{'s' if obs.n_transits > 1 else ''}")
    if title:
        ax.set_title(title)
    _legend(ax)
    return ax


def plot_noise(obs, ax=None, units="ppm", label=None, **plot_kw):
    """Per-bin 1-sigma depth uncertainty against wavelength (log y).

    This is ``obs.depth_err`` as binned, with ``n_transits`` included, so two
    observations at different R or N compare directly on one Axes.
    """
    scale, unit = _depth_scale(units)
    ax, fresh = _get_ax(ax, figsize=(9, 4))
    kw = dict(drawstyle="steps-mid", lw=1.2,
              label=label if label is not None else
              f"{obs.planet}, R={obs.R}, N={obs.n_transits}")
    kw.update(plot_kw)
    ax.plot(obs.wavelength, obs.depth_err * scale, **kw)
    wavelength_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel(f"1-sigma depth error{unit}")
    _legend(ax)
    return ax


def plot_light_curves(lc, ax=None, offset=None, show_points=True, show_model=True,
                      label=None, title=None):
    """Light curves of a LightCurveSet, one per band, stacked by ``offset``.

    A single band (white light) draws the noisy realization as points and
    the noiseless model as a line in a second colour, with a legend;
    ``label`` names the set in both entries (useful when overlaying two).
    Several bands are stacked ``offset`` apart (default 1.25 x the largest
    depth, or the flux range if no depth is positive), points and model in
    one colour per band, each annotated with its wavelength range at the
    left edge. A title is set on a new figure or an empty Axes unless
    ``title`` is given.
    """
    n = lc.flux_model.shape[0]
    ax, fresh = _get_ax(ax, figsize=(9, 4.5 if n == 1 else 3 + 0.8 * n))
    if n == 1:
        offset = 0.0
        tag = f" ({label})" if label is not None else ""
        if show_points:
            ax.plot(lc.time_hr, lc.flux[0], ".", ms=2.5, alpha=0.5,
                    label=f"simulated{tag}, {lc.cadence_s:.0f} s cadence")
        if show_model:
            ax.plot(lc.time_hr, lc.flux_model[0], lw=1.6, label=f"model{tag}")
    else:
        if offset is None:
            offset = 1.25 * float(np.max(lc.depth))
            if offset <= 0:
                offset = 1.25 * float(np.ptp(lc.flux_model))
        for i, lab in enumerate(lc.labels):
            dy = -i * offset
            color = None
            if show_model:
                line, = ax.plot(lc.time_hr, lc.flux_model[i] + dy, lw=1.6)
                color = line.get_color()
            if show_points:
                pts, = ax.plot(lc.time_hr, lc.flux[i] + dy, ".", ms=2.5, alpha=0.45,
                               color=color)
                color = color or pts.get_color()
            ax.annotate(lab, (lc.time_hr[0], 1.0 + dy), xytext=(0, 3),
                        textcoords="offset points", fontsize=9, color=color)
    ax.set_xlabel("Hours from mid-transit")
    ax.set_ylabel("Normalized flux" + (" + offset" if offset else ""))
    if title is None and fresh:
        m = lc.meta
        n_tr = m.get("n_transits")
        parts = [m.get("planet"), m.get("scenario"),
                 None if n_tr is None else f"{n_tr} transit{'s' if n_tr > 1 else ''}",
                 f"{lc.cadence_s:.0f} s cadence"]
        title = ", ".join(str(p) for p in parts if p)
    if title:
        ax.set_title(title)
    _legend(ax)
    return ax


# ---------------------------------------------------------------------------
# The transit in colour: one limb-darkened stellar disk per band with the
# planet crossing it, over the band light curves filling in with time.

def _transit_scene(lc, offset, cmap, npix, show_points):
    """Build the figure and return (fig, draw) where draw(idx) renders time sample idx."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle

    from . import orbit
    from .targets import get_target

    n = lc.flux_model.shape[0]
    time_s = lc.time_hr * 3600.0
    k = np.sqrt(np.clip(lc.depth, 0.0, None))
    m = lc.meta
    if m.get("shape") == "quadratic-ld":
        t = get_target(m["planet"])
        b = float(m["b"])
        z = orbit.projected_separation(time_s * m["time_stretch"], t["period_d"],
                                       m["a_over_rs"], t["inc_deg"],
                                       t["ecc"] or 0.0, t["w_deg"] or 90.0)
        x_pl = np.sign(time_s) * np.sqrt(np.clip(z ** 2 - b ** 2, 0.0, None))
        u1, u2 = lc.u1, lc.u2
    else:   # trapezoid fallback: central chord at constant speed, uniform disk
        b = 0.0
        k_med = float(np.sqrt(max(np.median(lc.depth), 0.0)))
        x_pl = time_s / (lc.t14_s / 2.0) * (1.0 + k_med)
        u1 = u2 = np.zeros(n)
    wl_mid = np.sqrt(lc.wl_lo * lc.wl_hi)
    colors = plt.get_cmap(cmap)((wl_mid - 0.40) / (1.70 - 0.40))
    if offset is None:
        offset = 0.0 if n == 1 else 1.25 * float(lc.depth.max())

    g = np.linspace(-1.15, 1.15, npix)
    X, Y = np.meshgrid(g, g)
    r2 = X ** 2 + Y ** 2
    mu = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    fig = plt.figure(figsize=(max(2.3 * n, 6.0), 7.2))
    gs = fig.add_gridspec(2, n, height_ratios=[1.6, 1.0] if n == 1 else [1.0, 1.6],
                          hspace=0.3, wspace=0.08)
    planets, points = [], []
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        disk = np.ma.masked_where(r2 > 1.0, 1.0 - u1[i] * (1.0 - mu) - u2[i] * (1.0 - mu) ** 2)
        ax.imshow(disk, extent=[g[0], g[-1], g[0], g[-1]], origin="lower", vmin=0.0, vmax=1.0,
                  cmap=LinearSegmentedColormap.from_list("tint", ["black", colors[i]]),
                  interpolation="bilinear")
        ax.set_facecolor("black")
        ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, ec="0.45", lw=0.6, zorder=4))
        planets.append(ax.add_patch(Circle((x_pl[0], b), k[i], color="black", zorder=5)))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{lc.labels[i]}\nu1 = {u1[i]:.2f}, depth {lc.depth[i] * 100:.3f} %",
                     fontsize=9)

    ax_lc = fig.add_subplot(gs[1, :])
    for i in range(n):
        dy = -i * offset
        ax_lc.plot(lc.time_hr, lc.flux_model[i] + dy, lw=1.2, color=colors[i], alpha=0.5)
        pt, = ax_lc.plot([], [], ".", ms=3, color=colors[i])
        points.append(pt)
        if n > 1:
            ax_lc.annotate(lc.labels[i], (lc.time_hr[0], 1.0 + dy), xytext=(0, 3),
                           textcoords="offset points", fontsize=9, color=colors[i])
    cursor = ax_lc.axvline(lc.time_hr[0], color="0.4", lw=0.8, ls="--")
    ax_lc.set_xlim(lc.time_hr[0], lc.time_hr[-1])
    dmax = float(lc.depth.max())
    ax_lc.set_ylim(1.0 - (n - 1) * offset - 1.3 * dmax, 1.0 + 0.4 * dmax)
    ax_lc.set_xlabel("Hours from mid-transit")
    ax_lc.set_ylabel("Normalized flux" + (" + offset" if offset else ""))
    n_tr = m.get("n_transits", 1)
    fig.suptitle(f"{m.get('planet', '')}, {m.get('scenario', '')}, {n_tr} transit"
                 f"{'s' if n_tr > 1 else ''}: the transit in {n} band{'s' if n > 1 else ''}",
                 y=0.98)

    def draw(idx):
        for i in range(n):
            planets[i].center = (x_pl[idx], b)
            if show_points:
                points[i].set_data(lc.time_hr[:idx + 1], lc.flux[i, :idx + 1] - i * offset)
        cursor.set_xdata([lc.time_hr[idx], lc.time_hr[idx]])
        return planets + points + [cursor]

    return fig, draw


def plot_transit(lc, t_hr=0.0, offset=None, cmap="plasma", npix=241, show_points=True):
    """Figure of the transit at time ``t_hr`` (hours from mid-transit), one
    limb-darkened stellar disk per band with the planet on it, over the band
    light curves drawn up to that time. Returns the Figure.

    Registry planets use their orbit and limb darkening; a custom planet
    (trapezoid light curves) gets a uniform disk and a central chord. Bands
    are tinted by wavelength through ``cmap``.
    """
    fig, draw = _transit_scene(lc, offset, cmap, npix, show_points)
    draw(int(np.argmin(np.abs(lc.time_hr - t_hr))))
    return fig


def animate_transit(lc, n_frames=60, fps=12, offset=None, cmap="plasma", npix=241,
                    show_points=True):
    """``matplotlib.animation.FuncAnimation`` of the transit, ``n_frames``
    evenly spaced over the observation window; see :func:`plot_transit` for
    the layout. Save it with ``anim.save("transit.gif")`` (pillow) or
    ``anim.save("transit.mp4")`` (ffmpeg); in a notebook, display it with
    ``IPython.display.HTML(anim.to_jshtml())`` or save and show the GIF.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, draw = _transit_scene(lc, offset, cmap, npix, show_points)
    frames = np.linspace(0, lc.time_hr.size - 1, n_frames).astype(int)
    anim = FuncAnimation(fig, lambda j: draw(frames[j]), frames=n_frames,
                         interval=1000.0 / fps, blit=False)
    plt.close(fig)        # the animation keeps its own reference
    return anim
