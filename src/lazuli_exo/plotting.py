"""Matplotlib plotting for SimulatedSpectrum and LightCurveSet ([plot] extra).

Every function takes an optional ``ax`` and returns the Axes it drew on, so
plots compose: pass the same ``ax`` to overlay scenarios, or pass one Axes of
a subplot grid. matplotlib is imported inside the functions, never at module
import. The functions do not touch the global style; set
``plt.style.use(...)`` yourself if you want one.
"""
import numpy as np

_MPL_HINT = "Plotting requires matplotlib: pip install lazuli-exo[plot]"
DEPTH_UNITS = {"percent": (100.0, "Transit depth (%)"),
               "ppm": (1e6, "Transit depth (ppm)"),
               "fraction": (1.0, "Transit depth")}
WAVELENGTH_TICKS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7)


def _get_ax(ax, **subplots_kw):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MPL_HINT) from exc
    if ax is None:
        _, ax = plt.subplots(**subplots_kw)
        return ax, True
    return ax, False


def _depth_scale(units):
    try:
        return DEPTH_UNITS[units]
    except KeyError:
        raise ValueError(f"units must be one of {list(DEPTH_UNITS)}, got {units!r}") from None


def wavelength_axis(ax, ticks=WAVELENGTH_TICKS, margin=1.03):
    """Log wavelength axis with the Lazuli band ticks and a plain-number formatter.

    Call it after plotting: the x-limits are fitted to the lines already on
    the Axes (times ``margin`` each side), which sidesteps matplotlib's
    errorbar data limits on a log axis and keeps shared axes consistent.
    """
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Wavelength (µm)")
    xs = [np.asarray(l.get_xdata(), float) for l in ax.lines]
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
    ``label`` replaces that text verbatim (the points' label, or the line's
    when there are no points). ``units`` is "percent", "ppm" or "fraction".
    ``**errorbar_kw`` goes to ``ax.errorbar`` (e.g. ``color=``). A title is
    set only on a new figure unless ``title`` is given.
    """
    scale, ylabel = _depth_scale(units)
    ax, fresh = _get_ax(ax, figsize=(9, 4.5))
    if show_model:
        ax.plot(obs.model_wavelength, obs.model_depth * scale, lw=0.8, alpha=0.7,
                color=errorbar_kw.get("color"),
                label=("model" if label is not None else f"model ({obs.scenario})")
                if show_points else (label if label is not None else obs.scenario))
    if show_points:
        points = obs.simulate_points(seed=seed)
        kw = dict(fmt="o", ms=3.5, lw=1, capsize=2,
                  label=label if label is not None else
                  f"simulated ({obs.scenario}, R={obs.R})")
        kw.update(errorbar_kw)
        ax.errorbar(obs.wavelength, points * scale, yerr=obs.depth_err * scale, **kw)
    wavelength_axis(ax)
    ax.set_ylabel(ylabel)
    if title is None and fresh:
        title = (f"{obs.planet}, {obs.scenario}, {obs.n_transits} "
                 f"transit{'s' if obs.n_transits > 1 else ''}")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def plot_noise(obs, ax=None, units="ppm", label=None, **plot_kw):
    """Per-bin 1-sigma depth uncertainty against wavelength (log y).

    This is ``obs.depth_err`` as binned, with ``n_transits`` included, so two
    observations at different R or N compare directly on one Axes.
    """
    scale, ylabel = _depth_scale(units)
    ax, fresh = _get_ax(ax, figsize=(9, 4))
    kw = dict(drawstyle="steps-mid", lw=1.2,
              label=label if label is not None else
              f"{obs.planet}, R={obs.R}, N={obs.n_transits}")
    kw.update(plot_kw)
    ax.plot(obs.wavelength, obs.depth_err * scale, **kw)
    wavelength_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel.replace("Transit depth", "1-sigma depth error"))
    ax.legend(frameon=False)
    return ax


def plot_light_curves(lc, ax=None, offset=None, show_points=True, show_model=True,
                      label=None, title=None):
    """Light curves of a LightCurveSet, one per band, stacked by ``offset``.

    A single band (white light) draws the noisy realization as points and
    the noiseless model as a line in a second colour, with a legend;
    ``label`` renames the model line (useful when overlaying two sets).
    Several bands are stacked ``offset`` apart (default 1.25 x the largest
    depth), points and model in one colour per band, each annotated with its
    wavelength range at the left edge.
    """
    n = lc.flux_model.shape[0]
    if offset is None:
        offset = 0.0 if n == 1 else 1.25 * float(np.max(lc.depth))
    ax, fresh = _get_ax(ax, figsize=(9, 4.5 if n == 1 else 3 + 0.8 * n))
    for i, lab in enumerate(lc.labels):
        dy = -i * offset
        color = None
        if n == 1:
            if show_points:
                ax.plot(lc.time_hr, lc.flux[0], ".", ms=2.5, alpha=0.5,
                        label=f"simulated ({lc.cadence_s:.0f} s cadence)")
            if show_model:
                ax.plot(lc.time_hr, lc.flux_model[0], lw=1.6,
                        label=label if label is not None else "model")
            continue
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
        title = (f"{m.get('planet', '')}, {m.get('scenario', '')}, "
                 f"{m.get('n_transits', 1)} transit"
                 f"{'s' if m.get('n_transits', 1) > 1 else ''}, "
                 f"{lc.cadence_s:.0f} s cadence")
    if title:
        ax.set_title(title)
    if n == 1:
        ax.legend(frameon=False)
    return ax
