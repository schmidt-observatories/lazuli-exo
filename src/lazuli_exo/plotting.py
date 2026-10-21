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


BAND_COLORS = {"Na": "#edae49", "K": "#66a182", "H2O": "#30638e",
               "CH4": "#d1495b", "CO2": "#8e6c8a"}
BAND_NAMES = {"Na": "sodium", "K": "potassium", "H2O": "water",
              "CH4": "methane", "CO2": "carbon dioxide"}


def shade_bands(ax, molecules=None, names="both", alpha=0.15, label="text"):
    """Shade the wavelength bands of :data:`lazuli_exo.detect.BAND_MASKS` on
    ``ax``, one colour per molecule, and name them.

    ``molecules`` selects a subset (default: all). ``names`` is "chemical"
    ("H2O"), "common" ("water") or "both" ("H2O, water"). ``label`` is "text"
    (a label at the top of each molecule's widest band), "legend" (one legend
    entry per molecule; call ``ax.legend()`` afterwards) or None. Returns the
    list of shaded patches. The bands overlap (H2O and CH4 near 1.15 µm) and
    are the masks of the template statistic, not line lists.
    """
    from matplotlib.transforms import blended_transform_factory
    from .detect import BAND_MASKS

    molecules = list(BAND_MASKS) if molecules is None else list(molecules)
    patches = []
    for mol in molecules:
        spans = BAND_MASKS[mol]
        color = BAND_COLORS.get(mol, "0.5")
        name = {"chemical": mol, "common": BAND_NAMES.get(mol, mol),
                "both": f"{mol}, {BAND_NAMES.get(mol, mol)}"}[names]
        widest = max(spans, key=lambda s: s[1] - s[0])
        for lo, hi in spans:
            patches.append(ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0, zorder=0,
                                      label=name if (label == "legend" and (lo, hi) == widest) else None))
        if label == "text":
            ax.text(np.sqrt(widest[0] * widest[1]), 0.985, name, color=color, fontsize=8,
                    ha="center", va="top", rotation=90 if widest[1] - widest[0] < 0.06 else 0,
                    transform=blended_transform_factory(ax.transData, ax.transAxes))
    return patches


def plot_spectrum(obs, ax=None, seed=0, show_model=True, show_points=True,
                  units="percent", label=None, title=None, bands=False,
                  **errorbar_kw):
    """Model spectrum and one noisy realization of the binned observation.

    ``show_model`` draws the native-resolution model as a line; ``show_points``
    draws ``simulate_points(seed)`` with the per-bin error bars. The legend
    names the points "simulated (scenario, R=...)" and the line "model
    (scenario)"; with ``show_points=False`` the line is labelled by scenario.
    ``label`` replaces the scenario text in those entries. ``units`` is
    "percent", "ppm" or "fraction". ``**errorbar_kw`` goes to ``ax.errorbar``;
    ``color``/``c`` also colours the model line. A title is set on a new
    figure or an empty Axes unless ``title`` is given. ``bands=True`` shades
    the molecular bands (:func:`shade_bands`); a list of molecule names
    shades only those.
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
    if bands:
        shade_bands(ax, None if bands is True else bands)
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


# ---------------------------------------------------------------------------
# The orbit: the planet going round its star, from a chosen vantage point.

def _star_color(teff_K):
    """A plausible star colour from T_eff (interpolated between a few anchors)."""
    anchors = [(2500, (1.0, 0.45, 0.25)), (3800, (1.0, 0.70, 0.40)),
               (5000, (1.0, 0.88, 0.60)), (5800, (1.0, 0.97, 0.85)),
               (7000, (0.90, 0.93, 1.0)), (10000, (0.70, 0.80, 1.0))]
    T = np.array([a[0] for a in anchors], float)
    rgb = np.array([a[1] for a in anchors], float)
    return tuple(np.interp(float(teff_K or 5800.0), T, rgb[:, j]) for j in range(3))


def _orbit_params(planet):
    """Registry name, SimulatedSpectrum or dict -> (name, period_d, a_over_rs,
    inc_deg, ecc, w_deg, k, teff_K)."""
    from .targets import get_target
    if isinstance(planet, dict):
        t = planet
    else:
        name = getattr(planet, "planet", planet)
        try:
            t = get_target(name)
        except KeyError:
            raise ValueError(f"{name!r} has no registry orbit; pass a dict with period_d, "
                             "a_over_rs, inc_deg (ecc, w_deg, depth, name, T_eff_K optional)") from None
    k = float(t["k"]) if "k" in t else float(np.sqrt(t.get("depth", 0.01)))
    return (t.get("name", "planet"), float(t["period_d"]), float(t["a_over_rs"]),
            float(t.get("inc_deg", 90.0)), float(t.get("ecc") or 0.0),
            float(t.get("w_deg") or 90.0), k, t.get("T_eff_K"))


def _orbit_scene(planet, view, n_frames, npix):
    """Build the figure; return (fig, draw, time_d) with draw(idx) at time_d[idx]."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    from . import orbit

    name, period, a_rs, inc, ecc, w, k, teff = _orbit_params(planet)
    inc_view = {"observer": inc, "top": 0.0}.get(view, view)
    label = {"observer": "seen from Earth", "top": "face-on"}.get(view, f"inclined {inc_view:.0f} deg")
    time_d = np.linspace(0.0, period, n_frames, endpoint=False)
    X, Y, Z = orbit.sky_position(time_d * 86400.0, period, a_rs, inc_view, ecc, w)
    t_path = np.linspace(0.0, period, 720)
    Xp, Yp, Zp = orbit.sky_position(t_path * 86400.0, period, a_rs, inc_view, ecc, w)

    g = np.linspace(-1.0, 1.0, npix)
    gx, gy = np.meshgrid(g, g)
    mu = np.sqrt(np.clip(1.0 - gx ** 2 - gy ** 2, 0.0, 1.0))
    disk = np.ma.masked_where(gx ** 2 + gy ** 2 > 1.0,
                              1.0 - 0.4 * (1.0 - mu) - 0.25 * (1.0 - mu) ** 2)

    half = 1.12 * a_rs * (1.0 + ecc)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_facecolor("black")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    behind = Zp < 0
    ax.plot(np.where(behind, np.nan, Xp), np.where(behind, np.nan, Yp), color="0.6", lw=0.8, zorder=5)
    ax.plot(np.where(behind, Xp, np.nan), np.where(behind, Yp, np.nan), color="0.35", lw=0.8, zorder=1)
    ax.imshow(disk, extent=[-1, 1, -1, 1], origin="lower", vmin=0, vmax=1, zorder=3,
              cmap=LinearSegmentedColormap.from_list("star", ["black", _star_color(teff)]))
    pl = ax.add_patch(Circle((X[0], Y[0]), k, color="0.2", ec="0.6", lw=0.5, zorder=6))
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white", va="top", fontsize=10)
    ax.set_title(f"{name}: orbit {label} (sizes to scale)")

    def draw(idx):
        pl.center = (X[idx], Y[idx])
        pl.set_zorder(6 if Z[idx] >= 0 else 2)
        near = np.hypot(X[idx], Y[idx]) < 1.0 + k
        txt.set_text(f"day {time_d[idx]:.2f} of {period:.2f}"
                     + ("   transit" if (Z[idx] > 0 and near) else
                        "   behind the star" if (Z[idx] < 0 and near) else ""))
        return [pl, txt]

    return fig, draw, time_d


def plot_orbit(planet, t_d=0.0, view="observer", n_frames=360, npix=201):
    """Figure of the planet on its orbit ``t_d`` days after mid-transit.

    ``planet`` is a registry name, a :class:`~lazuli_exo.SimulatedSpectrum`
    or a dict (``period_d``, ``a_over_rs``, ``inc_deg``; optional ``ecc``,
    ``w_deg``, ``depth`` or ``k``, ``name``, ``T_eff_K``). ``view`` is
    "observer" (Earth's line of sight, the registry inclination), "top"
    (face-on) or an inclination in degrees. Star and planet are to scale in
    stellar radii; the star's colour follows its temperature.
    """
    fig, draw, time_d = _orbit_scene(planet, view, n_frames, npix)
    period = time_d[1] * n_frames if n_frames > 1 else 1.0
    draw(int(np.argmin(np.abs(time_d - (t_d % period)))))
    return fig


def animate_orbit(planet, n_frames=120, fps=15, view="observer", npix=201):
    """``FuncAnimation`` of one full orbit, starting at mid-transit; see
    :func:`plot_orbit` for the arguments and ``anim.save("orbit.gif")`` to
    write it out.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, draw, _ = _orbit_scene(planet, view, n_frames, npix)
    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000.0 / fps, blit=False)
    plt.close(fig)
    return anim
