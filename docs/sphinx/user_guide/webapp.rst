Web applet and the web contract
===============================

Local Flask applet
------------------

``webapp/`` is a single-page Flask + Plotly UI over the runtime, for local
exploration only:

.. code-block:: bash

   pip install -e ".[web]"
   ./run.sh            # then open http://127.0.0.1:5000

*Showcase tables* mode: pick a planet, scenario, number of transits and R;
get the spectrum with one noisy realization, the template S/N, transit
light curves and the slicersim readout diagnostics. *Custom planet* mode:
prefill from a showcase preset, edit the star, planet and atmosphere, run
the parametric model; a Derived table reports :math:`g`, :math:`H`,
:math:`\mu` and its source, the temperature clamp, the Vega→AB conversion,
the readout used and any residual saturation.

The web contract
----------------

Every UI, including the public page on the Lazuli simulators site, drives
the model through two small modules so defaults, ranges and messages live in
one place:

- :func:`lazuli_exo.forms.parse` takes the raw form/query strings and returns
  typed values, fatal ``errors`` (missing magnitude, non-positive radius) and
  non-fatal ``warnings`` (a value was replaced or clamped; the run proceeds).
  Blank ``mu`` means chemistry, blank ``log_p_cloud`` means clear, blank
  ``log_a_haze`` means no haze.
- :func:`lazuli_exo.forms.to_params` maps the values onto
  :func:`lazuli_exo.simulate`; :func:`lazuli_exo.forms.from_preset` prefills
  from a showcase planet and scenario.
- :func:`lazuli_exo.report.summary` turns the resulting
  :class:`~lazuli_exo.observe.SimulatedSpectrum` into plain JSON: the arrays
  to plot, ``result`` scalars, ``derived`` rows (label, value, css class) and
  the caveat strings.

The Flask endpoints ``/api/presets``, ``/api/preset`` and ``/api/simulate``
are a dozen lines each on top of these. Notebook
:doc:`../notebooks/07_custom_planet` walks through the same path in Python.
