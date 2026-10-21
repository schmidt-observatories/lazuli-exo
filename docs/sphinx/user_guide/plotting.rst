Plotting
========

The result objects draw themselves. :meth:`~lazuli_exo.SimulatedSpectrum.plot`,
:meth:`~lazuli_exo.SimulatedSpectrum.plot_noise` and
:meth:`~lazuli_exo.lightcurve.LightCurveSet.plot` wrap the functions in
:mod:`lazuli_exo.plotting`. Each takes an optional ``ax`` and returns the
matplotlib Axes it drew on, so plots overlay on one Axes or tile into a
subplot grid. They need the ``[plot]`` extra (matplotlib).

Every figure on this page is drawn when the documentation is built, from the
code shown above it, so the plots match the code that produced them.

The functions do not set a matplotlib style. The tutorial notebooks apply one
in their preamble; use whatever style you like before calling them.

.. plot::
   :context: reset
   :nofigs:

   import matplotlib.pyplot as plt
   import lazuli_exo

   obs = lazuli_exo.observe("WASP-39 b", scenario="clear", n_transits=5, R=100)

A spectrum
----------

``obs.plot()`` draws the native-resolution model as a line and one noisy
realization of the binned spectrum as points with error bars. ``seed`` picks
the realization; ``units`` is ``"percent"`` (default), ``"ppm"`` or
``"fraction"``.

.. plot::
   :context: close-figs

   obs.plot(seed=0)

Overlaying scenarios
--------------------

Pass the same ``ax`` to several calls. ``show_points=False`` draws only the
model lines, and the legend entry is the scenario (or ``label`` if given).
A title is set only on a new figure or an empty Axes, so set your own after
an overlay.

.. plot::
   :context: close-figs

   fig, ax = plt.subplots(figsize=(9, 4.5))
   for s in ["clear", "cloudy", "hazy", "high_metallicity"]:
       lazuli_exo.observe("WASP-39 b", scenario=s).plot(ax=ax, show_points=False, units="ppm")
   ax.set_title("WASP-39 b: scenario presets, 1 transit")

Two observations with points can share an Axes as well; ``label`` names the
points and the matching model line (``model (<label>)``).

.. plot::
   :context: close-figs

   fig, ax = plt.subplots(figsize=(9, 4.5))
   for R in (30, 100):
       o = lazuli_exo.observe("WASP-39 b", scenario="clear", R=R)
       o.plot(ax=ax, label=f"R = {R}, {o.wavelength.size} bins")
   ax.set_title("WASP-39 b, clear, 1 transit: two binnings")

Which band is which molecule
----------------------------

``bands=True`` shades the wavelength bands of
:data:`lazuli_exo.detect.BAND_MASKS` and names them; a list such as
``bands=["H2O", "Na"]`` shades only those. The bands overlap (water and
methane both cover 1.10 to 1.20 µm) and are the masks of the template
statistic, so they say where a molecule absorbs, not that it was detected.
:func:`~lazuli_exo.plotting.shade_bands` does the same on any Axes, with
``names="common"`` for plain-language labels or ``label="legend"`` for
legend entries instead of text.

.. plot::
   :context: close-figs

   obs.plot(seed=0, bands=True)

The error bar against wavelength
--------------------------------

``obs.plot_noise()`` shows where the instrument is precise: the per-bin
1-sigma uncertainty, with ``n_transits`` included, on a log scale. Overlay
several binnings or transit counts to compare them directly.

.. plot::
   :context: close-figs

   fig, ax = plt.subplots(figsize=(9, 4))
   for R in (30, 100):
       lazuli_exo.observe("WASP-39 b", R=R).plot_noise(ax=ax)
   ax.set_title("WASP-39 b: per-bin error, 1 transit")

Light curves
------------

``obs.light_curves(n_groups=1)`` is the white-light transit; its ``plot()``
draws the noisy samples and the noiseless model with a legend.

.. plot::
   :context: close-figs

   obs.light_curves(n_groups=1, seed=0).plot()

With several bands the curves stack, ``offset`` apart, each annotated with
its wavelength range. The depth and the limb darkening change from band to
band; the geometry is shared.

.. plot::
   :context: close-figs

   obs.light_curves(n_groups=5, seed=0).plot()

The transit in colour
---------------------

``lc.plot_transit(t_hr)`` draws the transit at one moment: a stellar disk per
band, with that band's limb darkening and tinted by wavelength, the planet
on the registry orbit, and the band light curves filled in up to that time.
``lc.animate()`` returns a matplotlib ``FuncAnimation`` of the same scene
over the whole window; save it with ``anim.save("transit.gif")`` (pillow) or
``anim.save("transit.mp4")`` (ffmpeg), or show it in a notebook with
``IPython.display.HTML(anim.to_jshtml())``. A custom planet without a
registry orbit gets a uniform disk and a central chord.

.. plot::
   :context: close-figs

   lcb = obs.light_curves(n_groups=5, seed=0)
   lcb.plot_transit(t_hr=-0.8)

.. code-block:: python

   anim = lcb.animate(n_frames=60, fps=12)
   anim.save("wasp39b_transit.gif")

The orbit
---------

``obs.plot_orbit(t_d)`` draws the planet on its registry orbit ``t_d`` days
after mid-transit, star and planet to scale in stellar radii and the star
coloured by its temperature; ``obs.animate_orbit()`` is the same scene over
one full orbit as a ``FuncAnimation``. ``view`` is ``"observer"`` (Earth's
line of sight, so the planet transits and then passes behind the star),
``"top"`` (face-on) or an inclination in degrees. A planet without a
registry entry can be given as a dict with ``period_d``, ``a_over_rs`` and
``inc_deg``.

.. plot::
   :context: close-figs

   obs.plot_orbit(t_d=0.6)

.. code-block:: python

   anim = obs.animate_orbit(n_frames=120, fps=15, view=60)
   anim.save("wasp39b_orbit.gif")

Tiling
------

Because each call draws on the Axes you give it, a subplot grid works the
same way. Shared x axes keep one wavelength range across the panels.

.. plot::
   :context: close-figs

   fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
   for ax, n in zip(axes, (1, 5, 20)):
       lazuli_exo.observe("WASP-39 b", n_transits=n).plot(ax=ax, seed=0, title=f"{n} transits")
   for ax in axes[:-1]:
       ax.set_xlabel("")

Keywords
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Keywords
   * - ``obs.plot``
     - ``ax``, ``seed``, ``show_model``, ``show_points``, ``units``,
       ``label``, ``title``, ``bands``; anything else goes to ``ax.errorbar``
       (``color`` or ``c`` also colours the model line).
   * - ``obs.plot_noise``
     - ``ax``, ``units``, ``label``; anything else goes to ``ax.plot``.
   * - ``lc.plot``
     - ``ax``, ``offset`` (band spacing in normalized flux), ``show_points``,
       ``show_model``, ``label`` (white light only), ``title``.
   * - ``obs.plot_orbit`` / ``obs.animate_orbit``
     - ``t_d`` (still only), ``n_frames`` and ``fps`` (animation only),
       ``view`` ("observer", "top" or degrees), ``npix``.
   * - ``lc.plot_transit`` / ``lc.animate``
     - ``t_hr`` (still only), ``n_frames`` and ``fps`` (animation only),
       ``offset``, ``cmap`` (wavelength tint), ``npix`` (disk resolution),
       ``show_points``.
   * - ``plotting.wavelength_axis(ax)``
     - Log wavelength axis with the Lazuli band ticks; call it after
       plotting on an Axes you drew yourself.
